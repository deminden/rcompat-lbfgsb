use super::{BackendControl, BackendProblem, BackendResult, LbfgsbBackend};
use crate::diagnostics::{
    HistoryUpdate, LineSearchFailure, OptimEvent, RecoveryReason, StepAcceptance, TerminationReason,
};
use crate::error::OptimError;
use crate::result::OptimCounts;

mod compact;
mod iteration;
mod line_search;
use line_search::*;
#[cfg(test)]
mod reference;
use crate::profile::ArithmeticProfile;
#[cfg(test)]
use reference::*;

const BOUND_TOL: f64 = 1e-12;
const MAIN_PATH_MIN_STEP: f64 = 1e-16;

/// Native in-tree bounded limited-memory quasi-Newton backend.
#[derive(Debug, Default)]
pub(crate) struct NativeBackend;

impl LbfgsbBackend for NativeBackend {
    fn minimize<P>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
    ) -> Result<BackendResult, OptimError>
    where
        P: BackendProblem,
    {
        self.minimize_compact(problem, initial, lower, upper, control)
    }
}

fn exact_bound_count(x: &[f64], lower: &[f64], upper: &[f64]) -> usize {
    x.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .filter(|&((&value, &lower), &upper)| {
            (lower.is_finite() && value == lower) || (upper.is_finite() && value == upper)
        })
        .count()
}

#[derive(Debug, Clone)]
struct Evaluation {
    value: f64,
    gradient: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DirectionChoice {
    direction: Vec<f64>,
    unit_step_target: Option<Vec<f64>>,
}

fn evaluate<P>(
    problem: &mut P,
    x: &[f64],
    counts: &mut OptimCounts,
) -> Result<Evaluation, OptimError>
where
    P: BackendProblem,
{
    let (value, gradient) = match problem.value_and_gradient(x) {
        Ok(evaluation) => evaluation,
        Err(error) => {
            if P::OBSERVED {
                problem.observe(OptimEvent::EvaluationFailed { counts: *counts });
            }
            return Err(error);
        }
    };
    crate::error::checked_finite(value, "backend objective", None)?;
    if gradient.len() != x.len() {
        return Err(OptimError::InvalidGradient {
            index: None,
            value: None,
            reason: "backend gradient dimension mismatch".to_owned(),
        });
    }
    for (index, &component) in gradient.iter().enumerate() {
        crate::error::checked_finite(component, "backend gradient", Some(index))?;
    }
    counts.function += 1;
    counts.gradient += 1;
    Ok(Evaluation { value, gradient })
}

fn initial_evaluation_error(error: OptimError) -> OptimError {
    match error {
        OptimError::NonFiniteObjective { value } => OptimError::NonFiniteInitialValue { value },
        other => other,
    }
}

fn success(
    x: Vec<f64>,
    value: f64,
    counts: OptimCounts,
    reason: TerminationReason,
    iterations: usize,
) -> BackendResult {
    BackendResult {
        x,
        value,
        counts,
        reason,
        iterations,
        convergence: reason.convergence(),
        message: reason.message().to_owned(),
    }
}

fn iteration_limit(
    x: Vec<f64>,
    value: f64,
    counts: OptimCounts,
    iterations: usize,
) -> BackendResult {
    success(
        x,
        value,
        counts,
        TerminationReason::IterationLimit,
        iterations,
    )
}

fn project(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|((&value, &lower), &upper)| value.clamp(lower, upper))
        .collect()
}

fn project_direction(x: &[f64], lower: &[f64], upper: &[f64], direction: &mut [f64]) {
    for index in 0..x.len() {
        let fixed = lower[index] == upper[index];
        let below_lower = x[index] <= lower[index] + BOUND_TOL && direction[index] < 0.0;
        let above_upper = x[index] >= upper[index] - BOUND_TOL && direction[index] > 0.0;
        if fixed || below_lower || above_upper {
            direction[index] = 0.0;
        }
    }
}

fn steepest_projected_direction(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Vec<f64> {
    let mut direction: Vec<f64> = gradient.iter().map(|&value| -value).collect();
    project_direction(x, lower, upper, &mut direction);
    direction
}

fn projected_gradient_norm(x: &[f64], gradient: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut max_norm: f64 = 0.0;
    for index in 0..x.len() {
        let mut component = gradient[index];
        if !component.is_finite() {
            return f64::NAN;
        }
        if component < 0.0 && upper[index].is_finite() {
            component = component.max(x[index] - upper[index]);
        } else if component > 0.0 && lower[index].is_finite() {
            component = component.min(x[index] - lower[index]);
        }
        max_norm = max_norm.max(component.abs());
    }
    max_norm
}

fn relative_objective_reduction(previous: f64, current: f64) -> f64 {
    (previous - current) / previous.abs().max(current.abs()).max(1.0)
}

fn difference(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left - right)
        .collect()
}

fn norm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn maybe_trace(
    iteration: usize,
    value: f64,
    relative_reduction: f64,
    projected_norm: f64,
    step: &Step,
    control: BackendControl,
) {
    if control.trace > 0 && control.report > 0 && iteration.is_multiple_of(control.report) {
        eprintln!(
            "iter={iteration} f={value:.6e} rel_red={relative_reduction:.3e} ||proj_grad||_inf={projected_norm:.3e} line_search={} alpha={:.6e} max_alpha={:.6e} step_norm={:.6e} wolfe_curv={} curvature_ratio={:.3e}",
            step.line_search_trials,
            step.alpha,
            step.max_alpha,
            step.step_norm,
            step.wolfe_curvature_satisfied,
            step.curvature_ratio
        );
        if control.trace > 1 {
            eprintln!("x={:?}", step.x);
            eprintln!("gradient={:?}", step.gradient);
        }
    }
}

#[cfg(test)]
mod tests;
