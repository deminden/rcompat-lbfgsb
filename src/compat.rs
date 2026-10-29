use crate::backend::{BackendControl, LbfgsbBackend, NativeBackend};
use crate::bounds::Bounds;
use crate::control::OptimControl;
use crate::diagnostics::{NoObserver, OptimEvent, OptimObserver, TerminationReason};
use crate::error::OptimError;
use crate::objective::ScaledProblem;
use crate::profile::ArithmeticProfile;
use crate::result::{OptimCounts, OptimResult};
use crate::scaling::Scaling;

type NoGradient = fn(&[f64]) -> Vec<f64>;

/// Optimizes an objective using R-compatible L-BFGS-B wrapper semantics.
///
/// When no gradient is supplied, gradients are approximated with `ndeps` in the
/// scaled parameter space.
pub fn optim_lbfgsb<F>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    control: OptimControl,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
{
    let gradient: Option<NoGradient> = None;
    run(
        par,
        bounds,
        objective,
        gradient,
        control,
        ArithmeticProfile::legacy(),
        NoObserver,
    )
}

/// Optimizes an objective with a user-supplied gradient.
///
/// The supplied gradient must be the derivative of the user-scale objective with
/// respect to user-scale parameters. The compatibility layer converts it to the
/// scaled internal coordinate system.
pub fn optim_lbfgsb_with_gradient<F, G>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    gradient: G,
    control: OptimControl,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    run(
        par,
        bounds,
        objective,
        Some(gradient),
        control,
        ArithmeticProfile::legacy(),
        NoObserver,
    )
}

fn run<F, G, O>(
    par: Vec<f64>,
    bounds: Bounds,
    mut objective: F,
    gradient: Option<G>,
    control: OptimControl,
    profile: ArithmeticProfile,
    mut observer: O,
) -> Result<OptimResult, OptimError>
where
    O: OptimObserver,
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let n = par.len();
    control.validate_for_dimension(n)?;
    bounds.validate_for_dimension(n)?;
    validate_initial_parameters(&par, &bounds)?;
    validate_gradient_availability(&bounds, gradient.is_some())?;

    if n == 0 {
        let value = objective(&[]);
        if !value.is_finite() {
            if O::ENABLED {
                observer.observe(OptimEvent::EvaluationFailed {
                    counts: OptimCounts::default(),
                });
            }
            return Err(OptimError::NonFiniteInitialValue { value });
        }
        if O::ENABLED {
            observer.observe(OptimEvent::Finished {
                iterations: 0,
                reason: TerminationReason::NothingToDo,
                counts: OptimCounts {
                    function: 1,
                    gradient: 0,
                },
            });
        }
        return Ok(OptimResult {
            par,
            value,
            counts: OptimCounts {
                function: 1,
                gradient: 0,
            },
            convergence: 0,
            message: TerminationReason::NothingToDo.message().to_owned(),
        });
    }

    let scaling = Scaling::new(control.fnscale, control.parscale.clone());
    let projected: Vec<_> = par
        .iter()
        .zip(&bounds.lower)
        .zip(&bounds.upper)
        .map(|((&value, &lower), &upper)| value.clamp(lower, upper))
        .collect();
    let initial = scaling.user_to_internal_par(&projected)?;
    let scaled_bounds = scaling.scale_bounds(&bounds)?;
    let backend_control = BackendControl {
        profile,
        maxit: control.maxit,
        factr: control.factr,
        pgtol: control.pgtol,
        lmm: control.lmm,
        trace: control.trace,
        report: control.report,
        #[cfg(test)]
        has_user_gradient: gradient.is_some(),
    };

    let mut problem = ScaledProblem::new(
        objective,
        gradient,
        scaling,
        scaled_bounds.clone(),
        control.ndeps.clone(),
        observer,
    );
    let mut backend = NativeBackend;
    let backend_result = backend.minimize(
        &mut problem,
        &initial,
        &scaled_bounds.lower,
        &scaled_bounds.upper,
        backend_control,
    )?;

    let converted: Result<_, OptimError> = (|| {
        Ok((
            problem.user_parameters(&backend_result.x)?,
            problem.user_value_from_internal(backend_result.value)?,
        ))
    })();
    let (par, value) = match converted {
        Ok(values) => values,
        Err(error) => {
            problem.emit(OptimEvent::OutputConversionFailed {
                counts: backend_result.counts,
            });
            return Err(error);
        }
    };
    problem.emit(OptimEvent::Finished {
        iterations: backend_result.iterations,
        reason: backend_result.reason,
        counts: backend_result.counts,
    });
    Ok(OptimResult {
        par,
        value,
        counts: backend_result.counts,
        convergence: backend_result.convergence,
        message: backend_result.message,
    })
}

fn validate_initial_parameters(par: &[f64], bounds: &Bounds) -> Result<(), OptimError> {
    bounds.validate_for_dimension(par.len())?;
    for (index, &value) in par.iter().enumerate() {
        if value.is_nan() {
            return Err(OptimError::NonFiniteInitialParameter { index, value });
        }
        let projected = value.max(bounds.lower[index]).min(bounds.upper[index]);
        if !projected.is_finite() {
            return Err(OptimError::NonFiniteInitialParameter { index, value });
        }
    }
    Ok(())
}

fn validate_gradient_availability(
    bounds: &Bounds,
    has_user_gradient: bool,
) -> Result<(), OptimError> {
    if has_user_gradient {
        return Ok(());
    }

    for (index, (&lower, &upper)) in bounds.lower.iter().zip(bounds.upper.iter()).enumerate() {
        if lower == upper {
            return Err(OptimError::InvalidBounds {
                index: Some(index),
                lower: Some(lower),
                upper: Some(upper),
                reason: "R-compatible finite-difference optimization requires a user gradient for fixed parameters".to_string(),
            });
        }
    }

    Ok(())
}

/// Optimize with numerical gradients using an explicitly selected arithmetic target.
/// Selection is local to this run and independent of Cargo features.
pub fn optim_lbfgsb_with_profile<F>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    control: OptimControl,
    profile: ArithmeticProfile,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
{
    let gradient: Option<NoGradient> = None;
    run(
        par, bounds, objective, gradient, control, profile, NoObserver,
    )
}

/// Optimize with a supplied gradient and an explicit arithmetic target.
/// Callbacks use user-scale coordinates, as in the original entry points.
pub fn optim_lbfgsb_with_gradient_and_profile<F, G>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    gradient: G,
    control: OptimControl,
    profile: ArithmeticProfile,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    run(
        par,
        bounds,
        objective,
        Some(gradient),
        control,
        profile,
        NoObserver,
    )
}

/// Optimize with numerical gradients, an explicit profile and a structured observer.
/// Events use internal scaled units and do not cause extra callback evaluations.
pub fn optim_lbfgsb_observed<F, O>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    control: OptimControl,
    profile: ArithmeticProfile,
    observer: O,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
    O: OptimObserver,
{
    let gradient: Option<NoGradient> = None;
    run(par, bounds, objective, gradient, control, profile, observer)
}

/// Supplied-gradient variant of [`optim_lbfgsb_observed`].
pub fn optim_lbfgsb_with_gradient_observed<F, G, O>(
    par: Vec<f64>,
    bounds: Bounds,
    objective: F,
    gradient: G,
    control: OptimControl,
    profile: ArithmeticProfile,
    observer: O,
) -> Result<OptimResult, OptimError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
    O: OptimObserver,
{
    run(
        par,
        bounds,
        objective,
        Some(gradient),
        control,
        profile,
        observer,
    )
}
