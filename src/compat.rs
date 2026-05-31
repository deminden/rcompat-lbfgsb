use crate::backend::{BackendControl, LbfgsbBackend, NativeBackend};
use crate::bounds::Bounds;
use crate::control::OptimControl;
use crate::error::OptimError;
use crate::objective::ScaledProblem;
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
    run(par, bounds, objective, gradient, control)
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
    run(par, bounds, objective, Some(gradient), control)
}

fn run<F, G>(
    par: Vec<f64>,
    bounds: Bounds,
    mut objective: F,
    gradient: Option<G>,
    control: OptimControl,
) -> Result<OptimResult, OptimError>
where
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
            return Err(OptimError::NonFiniteInitialValue { value });
        }
        return Ok(OptimResult {
            par,
            value,
            counts: OptimCounts {
                function: 1,
                gradient: 0,
            },
            convergence: 0,
            message: "NOTHING TO DO".to_string(),
        });
    }

    let scaling = Scaling::new(control.fnscale, control.parscale.clone());
    let initial = scaling.user_to_internal_par(&par);
    let scaled_bounds = scaling.scale_bounds(&bounds);
    let backend_control = BackendControl {
        maxit: control.maxit,
        factr: control.factr,
        pgtol: control.pgtol,
        lmm: control.lmm,
        trace: control.trace,
        report: control.report,
        has_user_gradient: gradient.is_some(),
    };

    let mut problem = ScaledProblem::new(
        objective,
        gradient,
        scaling,
        scaled_bounds.clone(),
        control.ndeps.clone(),
    );
    let mut backend = NativeBackend;
    let backend_result = backend.minimize(
        &mut problem,
        &initial,
        &scaled_bounds.lower,
        &scaled_bounds.upper,
        backend_control,
    )?;

    Ok(OptimResult {
        par: problem.user_parameters(&backend_result.x),
        value: problem.user_value_from_internal(backend_result.value),
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
