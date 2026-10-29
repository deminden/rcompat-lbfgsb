//! Production More–Thuente search and step geometry.
use super::*;

mod interpolation;
pub(super) use interpolation::*;
mod search;
pub(super) use search::*;

pub(super) const BACKTRACK: f64 = 0.4995;
pub(super) const MAX_LINE_SEARCH_TRIALS: usize = 20;
pub(super) const MORE_THUENTE_BRACKET_SHRINK: f64 = 0.66;
pub(super) const MORE_THUENTE_FTOL: f64 = 1e-3;
pub(super) const MORE_THUENTE_XTRAPU: f64 = 4.0;
pub(super) const MORE_THUENTE_RETRY_RESTORE_BRACKET_RATIO: f64 = 0.1;
pub(super) const MORE_THUENTE_RETRY_WARNING_BRACKET_RATIO: f64 = 0.5;
pub(super) const WOLFE_CURVATURE: f64 = 0.9;
pub(super) const MORE_THUENTE_MAX_STEP: f64 = 1e10;

#[derive(Debug, Clone)]
pub(super) struct Step {
    pub(super) acceptance: StepAcceptance,
    pub(super) initial_derivative: f64,
    pub(super) final_derivative: f64,
    pub(super) x: Vec<f64>,
    pub(super) value: f64,
    pub(super) gradient: Vec<f64>,
    pub(super) line_search_trials: usize,
    pub(super) alpha: f64,
    pub(super) max_alpha: f64,
    pub(super) step_norm: f64,
    pub(super) curvature_ratio: f64,
    pub(super) wolfe_curvature_satisfied: bool,
}

pub(super) struct LineSearchRequest<'a> {
    pub(super) profile: ArithmeticProfile,
    pub(super) x: &'a [f64],
    pub(super) value: f64,
    pub(super) gradient: &'a [f64],
    pub(super) direction: &'a [f64],
    pub(super) unit_step_target: Option<&'a [f64]>,
    pub(super) lower: &'a [f64],
    pub(super) upper: &'a [f64],
    pub(super) max_step_cap: Option<f64>,
    pub(super) cap_initial_unbounded_step: bool,
    pub(super) allow_retry_warning_accept: bool,
    pub(super) min_step: f64,
    pub(super) exact_operations: bool,
}

pub(super) fn geometric_feasible_step(
    x: &[f64],
    direction: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let mut alpha = f64::INFINITY;
    for index in 0..x.len() {
        let d = direction[index];
        if d > 0.0 && upper[index].is_finite() {
            alpha = alpha.min((upper[index] - x[index]) / d);
        } else if d < 0.0 && lower[index].is_finite() {
            alpha = alpha.min((lower[index] - x[index]) / d);
        }
    }
    alpha.max(0.0)
}

pub(super) fn line_search_candidate(request: &LineSearchRequest<'_>, alpha: f64) -> Vec<f64> {
    if alpha == 1.0 {
        if let Some(target) = request.unit_step_target {
            return target.to_vec();
        }
    }

    if request.exact_operations {
        return request
            .x
            .iter()
            .zip(request.direction.iter())
            .map(|(&x_i, &d_i)| request.profile.mul_add(alpha, d_i, x_i))
            .collect();
    }

    bounded_step(
        request.profile,
        request.x,
        request.direction,
        alpha,
        request.lower,
        request.upper,
    )
}

pub(super) fn bounded_step(
    profile: ArithmeticProfile,
    x: &[f64],
    direction: &[f64],
    alpha: f64,
    lower: &[f64],
    upper: &[f64],
) -> Vec<f64> {
    x.iter()
        .zip(direction.iter())
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|(((&x_i, &d_i), &lower_i), &upper_i)| {
            (profile.mul_add(alpha, d_i, x_i)).clamp(lower_i, upper_i)
        })
        .collect()
}
