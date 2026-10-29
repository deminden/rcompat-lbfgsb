//! More-Thuente search and isolated historical search probes.
use super::*;

#[cfg(test)]
pub(super) const ARMIJO: f64 = 1e-4;
pub(super) const BACKTRACK: f64 = 0.4995;
pub(super) const MAX_LINE_SEARCH_TRIALS: usize = 20;
pub(super) const MORE_THUENTE_BRACKET_SHRINK: f64 = 0.66;
pub(super) const MORE_THUENTE_FTOL: f64 = 1e-3;
pub(super) const MORE_THUENTE_XTRAPU: f64 = 4.0;
pub(super) const MORE_THUENTE_RETRY_RESTORE_BRACKET_RATIO: f64 = 0.1;
pub(super) const MORE_THUENTE_RETRY_WARNING_BRACKET_RATIO: f64 = 0.5;
#[cfg(test)]
pub(super) const STRONG_WOLFE_UNBOUNDED_MAX_STEP: f64 = 1.0e20;
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
    #[cfg(test)]
    pub(super) used_multidimensional_interpolation: bool,
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
    pub(super) initial_step_cap: Option<f64>,
    #[cfg(test)]
    pub(super) allow_quadratic_interpolation: bool,
    pub(super) allow_retry_warning_accept: bool,
    pub(super) min_step: f64,
    pub(super) exact_operations: bool,
    #[cfg(test)]
    pub(super) quadratic_interpolation_damping: f64,
}

#[cfg(test)]
pub(super) fn line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let mut max_alpha =
        max_feasible_step(request.x, request.direction, request.lower, request.upper);
    if let Some(cap) = request.max_step_cap {
        max_alpha = max_alpha.min(cap);
    }
    let mut alpha = max_alpha.min(1.0);
    if request.cap_initial_unbounded_step {
        alpha = alpha.min(1.0 / norm_inf(request.direction).max(1.0));
    }
    if let Some(cap) = request.initial_step_cap {
        alpha = alpha.min(cap);
    }
    if alpha <= 0.0 || !alpha.is_finite() {
        return Ok(None);
    }
    let initial_alpha = alpha;
    let initial_directional_derivative = request.profile.dot(
        request.exact_operations,
        request.gradient,
        request.direction,
    );
    let mut tried_quadratic_step = false;
    let mut used_multidimensional_interpolation = false;
    let mut evaluated_candidates = 0_usize;

    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        let mut candidate = line_search_candidate(&request, alpha);
        for (index, coordinate) in candidate.iter_mut().enumerate() {
            let direction = request.direction[index];
            if direction < 0.0
                && request.lower[index].is_finite()
                && alpha == (request.lower[index] - request.x[index]) / direction
            {
                *coordinate = request.lower[index];
            } else if direction > 0.0
                && request.upper[index].is_finite()
                && alpha == (request.upper[index] - request.x[index]) / direction
            {
                *coordinate = request.upper[index];
            }
        }
        let step = difference(&candidate, request.x);

        if norm_inf(&step) <= request.min_step {
            alpha *= BACKTRACK;
            continue;
        }

        let directional_decrease =
            request
                .profile
                .dot(request.exact_operations, request.gradient, &step);
        if directional_decrease < 0.0 {
            let evaluation = evaluate(problem, &candidate, counts)?;
            evaluated_candidates += 1;
            if evaluation.value <= request.value + ARMIJO * directional_decrease {
                let directional_derivative = request.profile.dot(
                    request.exact_operations,
                    &evaluation.gradient,
                    request.direction,
                );
                let curvature_ratio = directional_derivative.abs()
                    / (-initial_directional_derivative).max(f64::MIN_POSITIVE);
                return Ok(Some(Step {
                    acceptance: StepAcceptance::Armijo,
                    initial_derivative: initial_directional_derivative,
                    final_derivative: directional_derivative,
                    x: candidate,
                    value: evaluation.value,
                    gradient: evaluation.gradient,
                    line_search_trials: evaluated_candidates.saturating_sub(1),
                    alpha,
                    max_alpha,
                    step_norm: norm2(&step),
                    curvature_ratio,
                    wolfe_curvature_satisfied: curvature_ratio <= WOLFE_CURVATURE,
                    used_multidimensional_interpolation,
                }));
            }
            if !tried_quadratic_step
                && (request.x.len() == 1
                    || (request.allow_quadratic_interpolation && initial_alpha >= 0.1))
            {
                tried_quadratic_step = true;
                if let Some(interpolated_alpha) = quadratic_trial_alpha(
                    request.value,
                    initial_directional_derivative,
                    initial_alpha,
                    evaluation.value,
                    request.quadratic_interpolation_damping,
                ) {
                    used_multidimensional_interpolation = request.x.len() > 1;
                    alpha = interpolated_alpha;
                    continue;
                }
            }
        }

        alpha *= BACKTRACK;
    }

    Ok(None)
}

#[cfg(test)]
pub(super) fn line_search_with_mode<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
    mode: LineSearchMode,
    iteration: usize,
) -> Result<Result<Step, LineSearchFailure>, OptimError>
where
    P: BackendProblem,
{
    let _ = iteration;
    match mode {
        LineSearchMode::BacktrackingArmijo => {
            Ok(line_search(problem, request, counts)?.ok_or(LineSearchFailure::TrialLimit))
        }
        LineSearchMode::MoreThuente => more_thuente_line_search(problem, request, counts),
        #[cfg(test)]
        LineSearchMode::StrongWolfe => Ok(strong_wolfe_line_search(problem, request, counts)?
            .ok_or(LineSearchFailure::TrialLimit)),
        #[cfg(test)]
        LineSearchMode::MoreThuenteFirstThenBacktracking => {
            if iteration == 1 {
                more_thuente_line_search(problem, request, counts)
            } else {
                Ok(line_search(problem, request, counts)?.ok_or(LineSearchFailure::TrialLimit))
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoreThuentePoint {
    pub(super) alpha: f64,
    pub(super) value: f64,
    pub(super) derivative: f64,
}

pub(super) fn more_thuente_line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Result<Step, LineSearchFailure>, OptimError>
where
    P: BackendProblem,
{
    let mut max_alpha =
        geometric_feasible_step(request.x, request.direction, request.lower, request.upper)
            .min(MORE_THUENTE_MAX_STEP);
    if let Some(cap) = request.max_step_cap {
        max_alpha = max_alpha.min(cap);
    }
    let mut alpha = if request.cap_initial_unbounded_step {
        (1.0 / request
            .profile
            .dot(
                request.exact_operations,
                request.direction,
                request.direction,
            )
            .sqrt())
        .min(max_alpha)
    } else {
        max_alpha.min(1.0)
    };
    if let Some(cap) = request.initial_step_cap {
        alpha = alpha.min(cap);
    }

    let initial_derivative = request.profile.dot(
        request.exact_operations,
        request.gradient,
        request.direction,
    );
    if alpha <= 0.0 || !alpha.is_finite() {
        return Ok(Err(LineSearchFailure::InvalidInitialStep));
    }
    if initial_derivative >= 0.0 {
        return Ok(Err(LineSearchFailure::NonDescentDirection));
    }

    let mut best = MoreThuentePoint {
        alpha: 0.0,
        value: request.value,
        derivative: initial_derivative,
    };
    let mut other = best;
    let mut bracketed = false;
    let mut stage_one = true;
    let mut width = max_alpha;
    let mut previous_width = 2.0 * width;
    let mut evaluated_candidates = 0_usize;
    let decrease_test = MORE_THUENTE_FTOL * initial_derivative;

    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        // This is the L-BFGS-B 2.3 More-Thuente task update: evaluate a trial,
        // accept strong-Wolfe or feasible-cap cases, then update the safeguarded
        // bracket using the modified stage-one objective when needed.
        let candidate = line_search_candidate(&request, alpha);
        let step = difference(&candidate, request.x);
        // The exact finite-box task still submits trials whose coordinates round
        // back to x. Those evaluations advance the line-search warning state and
        // are observable through R's function/gradient counts.
        if !request.exact_operations && norm_inf(&step) <= request.min_step {
            alpha = if bracketed {
                0.5 * (best.alpha + other.alpha)
            } else {
                (alpha * BACKTRACK).max(request.min_step)
            };
            continue;
        }

        let evaluation = evaluate(problem, &candidate, counts)?;
        evaluated_candidates += 1;
        let derivative = request.profile.dot(
            request.exact_operations,
            &evaluation.gradient,
            request.direction,
        );
        let point = MoreThuentePoint {
            alpha,
            value: evaluation.value,
            derivative,
        };
        let curvature_ratio = derivative.abs() / (-initial_derivative).max(f64::MIN_POSITIVE);
        if evaluation.value <= more_thuente_sufficient_decrease(request.value, decrease_test, alpha)
            && curvature_ratio <= WOLFE_CURVATURE
        {
            return Ok(Ok(Step {
                acceptance: StepAcceptance::StrongWolfe,
                initial_derivative,
                final_derivative: derivative,
                x: candidate,
                value: evaluation.value,
                gradient: evaluation.gradient,
                line_search_trials: evaluated_candidates.saturating_sub(1),
                alpha,
                max_alpha,
                step_norm: norm2(&step),
                curvature_ratio,
                wolfe_curvature_satisfied: true,
                #[cfg(test)]
                used_multidimensional_interpolation: false,
            }));
        }

        let sufficient_decrease =
            more_thuente_sufficient_decrease(request.value, decrease_test, alpha);
        if alpha == max_alpha
            && evaluation.value <= sufficient_decrease
            && derivative <= decrease_test
        {
            return Ok(Ok(Step {
                acceptance: StepAcceptance::StepLimit,
                initial_derivative,
                final_derivative: derivative,
                x: candidate,
                value: evaluation.value,
                gradient: evaluation.gradient,
                line_search_trials: evaluated_candidates.saturating_sub(1),
                alpha,
                max_alpha,
                step_norm: norm2(&step),
                curvature_ratio,
                wolfe_curvature_satisfied: false,
                #[cfg(test)]
                used_multidimensional_interpolation: false,
            }));
        }
        if evaluation.value <= sufficient_decrease
            && more_thuente_accepts_warning(
                request.exact_operations,
                request.allow_retry_warning_accept,
                best.alpha,
                other.alpha,
                bracketed,
            )
        {
            return Ok(Ok(Step {
                acceptance: StepAcceptance::BracketWarning,
                initial_derivative,
                final_derivative: derivative,
                x: candidate,
                value: evaluation.value,
                gradient: evaluation.gradient,
                line_search_trials: evaluated_candidates.saturating_sub(1),
                alpha,
                max_alpha,
                step_norm: norm2(&step),
                curvature_ratio,
                wolfe_curvature_satisfied: false,
                #[cfg(test)]
                used_multidimensional_interpolation: false,
            }));
        }

        if more_thuente_enters_stage_two(
            stage_one,
            evaluation.value,
            sufficient_decrease,
            derivative,
        ) {
            stage_one = false;
        }

        let (step_min, step_max) =
            more_thuente_safeguard_interval(best.alpha, other.alpha, alpha, max_alpha, bracketed);

        let next_alpha = if stage_one
            && evaluation.value <= best.value
            && evaluation.value > sufficient_decrease
        {
            let modified_best = MoreThuentePoint {
                alpha: best.alpha,
                value: best.value - best.alpha * decrease_test,
                derivative: best.derivative - decrease_test,
            };
            let modified_other = MoreThuentePoint {
                alpha: other.alpha,
                value: other.value - other.alpha * decrease_test,
                derivative: other.derivative - decrease_test,
            };
            let modified_trial = MoreThuentePoint {
                alpha,
                value: evaluation.value - alpha * decrease_test,
                derivative: derivative - decrease_test,
            };
            let mut step_state = MoreThuenteStepState {
                best: modified_best,
                other: modified_other,
                bracketed,
            };
            let next_alpha = more_thuente_step(
                request.profile,
                &mut step_state,
                modified_trial,
                step_min,
                step_max,
            );
            best = MoreThuentePoint {
                alpha: step_state.best.alpha,
                value: step_state.best.value + step_state.best.alpha * decrease_test,
                derivative: step_state.best.derivative + decrease_test,
            };
            other = MoreThuentePoint {
                alpha: step_state.other.alpha,
                value: step_state.other.value + step_state.other.alpha * decrease_test,
                derivative: step_state.other.derivative + decrease_test,
            };
            bracketed = step_state.bracketed;
            next_alpha
        } else {
            let mut step_state = MoreThuenteStepState {
                best,
                other,
                bracketed,
            };
            let next_alpha =
                more_thuente_step(request.profile, &mut step_state, point, step_min, step_max);
            best = step_state.best;
            other = step_state.other;
            bracketed = step_state.bracketed;
            next_alpha
        };

        alpha = next_alpha.clamp(0.0, max_alpha);
        if bracketed {
            if (other.alpha - best.alpha).abs() >= MORE_THUENTE_BRACKET_SHRINK * previous_width {
                alpha = best.alpha + 0.5 * (other.alpha - best.alpha);
            }
            previous_width = width;
            width = (other.alpha - best.alpha).abs();
        }
        if more_thuente_restores_best(
            request.exact_operations,
            request.allow_retry_warning_accept,
            best.alpha,
            other.alpha,
            bracketed,
        ) {
            // The exact task and a restarted search restore the best point once
            // the bracket reaches warning width. The following task evaluates
            // that point before accepting the warning as a successful iterate.
            alpha = best.alpha;
        }

        if alpha <= 0.0 || !alpha.is_finite() {
            return Ok(Err(LineSearchFailure::InvalidUpdatedStep));
        }
    }

    Ok(Err(LineSearchFailure::TrialLimit))
}

pub(super) fn more_thuente_sufficient_decrease(
    value_at_zero: f64,
    decrease_test: f64,
    alpha: f64,
) -> f64 {
    value_at_zero + alpha * decrease_test
}

pub(super) fn more_thuente_enters_stage_two(
    stage_one: bool,
    value: f64,
    sufficient_decrease: f64,
    derivative: f64,
) -> bool {
    stage_one && value <= sufficient_decrease && derivative >= 0.0
}

pub(super) fn more_thuente_retry_warning(
    best_alpha: f64,
    other_alpha: f64,
    bracketed: bool,
) -> bool {
    if !bracketed {
        return false;
    }
    let step_max = best_alpha.max(other_alpha);
    step_max > 0.0
        && (other_alpha - best_alpha).abs() <= MORE_THUENTE_RETRY_WARNING_BRACKET_RATIO * step_max
}

pub(super) fn more_thuente_retry_restore(
    best_alpha: f64,
    other_alpha: f64,
    bracketed: bool,
) -> bool {
    if !bracketed {
        return false;
    }
    let step_max = best_alpha.max(other_alpha);
    step_max > 0.0
        && (other_alpha - best_alpha).abs() <= MORE_THUENTE_RETRY_RESTORE_BRACKET_RATIO * step_max
}

pub(super) fn more_thuente_accepts_warning(
    exact_operations: bool,
    allow_retry_warning_accept: bool,
    best_alpha: f64,
    other_alpha: f64,
    bracketed: bool,
) -> bool {
    (exact_operations && more_thuente_retry_restore(best_alpha, other_alpha, bracketed))
        || (!exact_operations
            && allow_retry_warning_accept
            && more_thuente_retry_warning(best_alpha, other_alpha, bracketed))
}

pub(super) fn more_thuente_restores_best(
    exact_operations: bool,
    allow_retry_warning_accept: bool,
    best_alpha: f64,
    other_alpha: f64,
    bracketed: bool,
) -> bool {
    (exact_operations || allow_retry_warning_accept)
        && more_thuente_retry_restore(best_alpha, other_alpha, bracketed)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct MoreThuenteStepState {
    pub(super) best: MoreThuentePoint,
    pub(super) other: MoreThuentePoint,
    pub(super) bracketed: bool,
}

pub(super) fn more_thuente_safeguard_interval(
    best_alpha: f64,
    other_alpha: f64,
    alpha: f64,
    max_alpha: f64,
    bracketed: bool,
) -> (f64, f64) {
    if bracketed {
        (best_alpha.min(other_alpha), best_alpha.max(other_alpha))
    } else if alpha >= best_alpha {
        (
            0.0,
            (alpha + MORE_THUENTE_XTRAPU * (alpha - best_alpha)).min(max_alpha),
        )
    } else {
        (
            (alpha + MORE_THUENTE_XTRAPU * (alpha - best_alpha)).max(0.0),
            max_alpha,
        )
    }
}

pub(super) fn more_thuente_step(
    profile: ArithmeticProfile,
    state: &mut MoreThuenteStepState,
    trial: MoreThuentePoint,
    step_min: f64,
    step_max: f64,
) -> f64 {
    let best = state.best;
    let delta = trial.alpha - best.alpha;
    if delta == 0.0 {
        return trial.alpha;
    }
    let signed_derivative = trial.derivative * (best.derivative / best.derivative.abs());
    let mut bounded = false;

    let mut next_alpha = if trial.value > best.value {
        state.bracketed = true;
        bounded = true;
        let cubic = more_thuente_cubic_case_one(profile, best, trial).unwrap_or(trial.alpha);
        let quadratic = more_thuente_quadratic_case_one(best, trial).unwrap_or(cubic);
        if (cubic - best.alpha).abs() < (quadratic - best.alpha).abs() {
            cubic
        } else {
            cubic + 0.5 * (quadratic - cubic)
        }
    } else if signed_derivative < 0.0 {
        state.bracketed = true;
        let cubic = more_thuente_cubic_case_two(profile, best, trial).unwrap_or(trial.alpha);
        let secant = more_thuente_secant(best, trial).unwrap_or(cubic);
        if (cubic - trial.alpha).abs() > (secant - trial.alpha).abs() {
            cubic
        } else {
            secant
        }
    } else if trial.derivative.abs() < best.derivative.abs() {
        bounded = true;
        let cubic = more_thuente_cubic_case_three(profile, best, trial);
        let secant = more_thuente_secant(best, trial);
        match (cubic, secant) {
            (Some(cubic), Some(secant)) if state.bracketed => {
                if (cubic - trial.alpha).abs() < (secant - trial.alpha).abs() {
                    cubic
                } else {
                    secant
                }
            }
            (Some(cubic), Some(secant)) => {
                if (cubic - trial.alpha).abs() > (secant - trial.alpha).abs() {
                    cubic
                } else {
                    secant
                }
            }
            (Some(cubic), None) => cubic,
            (None, Some(secant)) => secant,
            (None, None) if trial.alpha > best.alpha => step_max,
            (None, None) => step_min,
        }
    } else if state.bracketed {
        more_thuente_cubic_case_four(profile, state.other, trial)
            .unwrap_or(0.5 * (trial.alpha + state.other.alpha))
    } else if trial.alpha > best.alpha {
        step_max
    } else {
        step_min
    };

    if trial.value > best.value {
        state.other = trial;
    } else {
        if signed_derivative < 0.0 {
            state.other = best;
        }
        state.best = trial;
    }

    next_alpha = next_alpha.clamp(step_min, step_max);
    if state.bracketed && bounded {
        let bracket_limit =
            state.best.alpha + MORE_THUENTE_BRACKET_SHRINK * (state.other.alpha - state.best.alpha);
        if state.other.alpha > state.best.alpha {
            next_alpha = next_alpha.min(bracket_limit);
        } else {
            next_alpha = next_alpha.max(bracket_limit);
        }
    }

    if next_alpha.is_finite() {
        next_alpha
    } else {
        0.5 * (step_min + step_max)
    }
}

pub(super) fn more_thuente_cubic_case_one(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    let p = (gamma - best.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - best.derivative) + gamma) + trial.derivative;
    finite_ratio(p, q).map(|ratio| best.alpha + ratio * delta)
}

pub(super) fn more_thuente_quadratic_case_one(
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let denominator = (best.value - trial.value) / delta + best.derivative;
    finite_ratio(0.5 * best.derivative, denominator).map(|ratio| best.alpha + ratio * delta)
}

pub(super) fn more_thuente_cubic_case_two(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + best.derivative;
    finite_ratio(p, q).map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

pub(super) fn more_thuente_cubic_case_three(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(best, trial, delta);
    let q = (gamma + (best.derivative - trial.derivative)) + gamma;
    finite_ratio(p, q)
        .filter(|ratio| *ratio < 0.0 && gamma != 0.0)
        .map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

pub(super) fn more_thuente_cubic_case_four(
    profile: ArithmeticProfile,
    other: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - other.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        other.value,
        other.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(other, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + other.derivative;
    finite_ratio(p, q).map(|ratio| trial.alpha + ratio * (other.alpha - trial.alpha))
}

pub(super) fn more_thuente_secant(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    finite_ratio(trial.derivative, trial.derivative - best.derivative)
        .map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

pub(super) fn more_thuente_theta(
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
    delta: f64,
) -> f64 {
    3.0 * (best.value - trial.value) / delta + best.derivative + trial.derivative
}

pub(super) fn more_thuente_gamma(
    profile: ArithmeticProfile,
    best_value: f64,
    best_derivative: f64,
    trial_value: f64,
    trial_derivative: f64,
    delta: f64,
) -> Option<f64> {
    let theta = 3.0 * (best_value - trial_value) / delta + best_derivative + trial_derivative;
    let scale = theta
        .abs()
        .max(best_derivative.abs())
        .max(trial_derivative.abs());
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }

    let normalized_theta = theta / scale;
    let discriminant = profile.mul_add(
        normalized_theta,
        normalized_theta,
        -((best_derivative / scale) * (trial_derivative / scale)),
    );
    if !discriminant.is_finite() {
        return None;
    }
    let mut gamma = scale * discriminant.max(0.0).sqrt();
    if delta < 0.0 {
        gamma = -gamma;
    }
    Some(gamma)
}

pub(super) fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let ratio = numerator / denominator;
    ratio.is_finite().then_some(ratio)
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct StrongWolfeCandidate {
    pub(super) point: LineSearchPoint1D,
    pub(super) x: Vec<f64>,
    pub(super) step: Vec<f64>,
    pub(super) evaluation: Evaluation,
}

#[cfg(test)]
pub(super) fn strong_wolfe_line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let max_alpha = max_feasible_step_allowing_unbounded(
        request.x,
        request.direction,
        request.lower,
        request.upper,
    );
    let mut initial_alpha = max_alpha.min(1.0);
    if request.cap_initial_unbounded_step {
        initial_alpha = initial_alpha.min(1.0 / norm_inf(request.direction).max(1.0));
    }
    if let Some(cap) = request.initial_step_cap {
        initial_alpha = initial_alpha.min(cap);
    }

    let initial_directional_derivative = dot(request.gradient, request.direction);
    if initial_alpha <= 0.0 || initial_directional_derivative >= 0.0 {
        return Ok(None);
    }

    let mut last_candidate = None;
    let mut evaluations = 0_usize;
    let result = strong_wolfe_search(
        initial_alpha,
        max_alpha,
        request.value,
        initial_directional_derivative,
        |alpha| {
            let candidate = line_search_candidate(&request, alpha);
            let step = difference(&candidate, request.x);
            let evaluation = evaluate(problem, &candidate, counts)?;
            evaluations += 1;
            let derivative = dot(&evaluation.gradient, request.direction);
            let point = LineSearchPoint1D {
                alpha,
                value: evaluation.value,
                derivative,
            };
            last_candidate = Some(StrongWolfeCandidate {
                point,
                x: candidate,
                step,
                evaluation,
            });
            Ok(point)
        },
    )?;

    let Some(result) = result else {
        let Some(candidate) = last_candidate else {
            return Ok(None);
        };
        if candidate.point.alpha == max_alpha
            && candidate.point.value
                <= armijo_value(
                    request.value,
                    initial_directional_derivative,
                    candidate.point.alpha,
                )
        {
            return Ok(Some(strong_wolfe_step_from_candidate(
                candidate,
                max_alpha,
                evaluations.saturating_sub(1),
                initial_directional_derivative,
            )));
        }
        return Ok(None);
    };

    let Some(candidate) = last_candidate else {
        return Ok(None);
    };
    debug_assert_eq!(candidate.point, result.point);

    Ok(Some(strong_wolfe_step_from_candidate(
        candidate,
        max_alpha,
        result.evaluations.saturating_sub(1),
        initial_directional_derivative,
    )))
}

#[cfg(test)]
pub(super) fn strong_wolfe_step_from_candidate(
    candidate: StrongWolfeCandidate,
    max_alpha: f64,
    line_search_trials: usize,
    initial_directional_derivative: f64,
) -> Step {
    let curvature_ratio =
        candidate.point.derivative.abs() / (-initial_directional_derivative).max(f64::MIN_POSITIVE);

    Step {
        acceptance: if curvature_ratio <= WOLFE_CURVATURE {
            StepAcceptance::StrongWolfe
        } else {
            StepAcceptance::StepLimit
        },
        initial_derivative: initial_directional_derivative,
        final_derivative: candidate.point.derivative,
        x: candidate.x,
        value: candidate.evaluation.value,
        gradient: candidate.evaluation.gradient,
        line_search_trials,
        alpha: candidate.point.alpha,
        max_alpha,
        step_norm: norm2(&candidate.step),
        curvature_ratio,
        wolfe_curvature_satisfied: curvature_ratio <= WOLFE_CURVATURE,
        #[cfg(test)]
        used_multidimensional_interpolation: false,
    }
}

#[cfg(test)]
pub(super) fn quadratic_trial_alpha(
    value_at_zero: f64,
    derivative_at_zero: f64,
    trial_alpha: f64,
    trial_value: f64,
    damping: f64,
) -> Option<f64> {
    let denominator = 2.0 * (trial_value - value_at_zero - derivative_at_zero * trial_alpha);
    if denominator <= 0.0 || !denominator.is_finite() {
        return None;
    }

    let alpha = -derivative_at_zero * trial_alpha * trial_alpha / denominator;
    if alpha > 0.0 && alpha < trial_alpha && alpha.is_finite() {
        if trial_alpha >= 1.0 {
            Some(alpha * damping)
        } else {
            Some(alpha)
        }
    } else {
        None
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LineSearchPoint1D {
    pub(super) alpha: f64,
    pub(super) value: f64,
    pub(super) derivative: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct StrongWolfeResult {
    pub(super) point: LineSearchPoint1D,
    pub(super) evaluations: usize,
}

#[cfg(test)]
pub(super) fn strong_wolfe_search<F>(
    initial_alpha: f64,
    max_alpha: f64,
    value_at_zero: f64,
    derivative_at_zero: f64,
    mut evaluate: F,
) -> Result<Option<StrongWolfeResult>, OptimError>
where
    F: FnMut(f64) -> Result<LineSearchPoint1D, OptimError>,
{
    if initial_alpha <= 0.0
        || max_alpha <= 0.0
        || derivative_at_zero >= 0.0
        || !initial_alpha.is_finite()
        || !max_alpha.is_finite()
    {
        return Ok(None);
    }

    let mut evaluations = 0;
    let mut previous_alpha = 0.0;
    let mut previous_value = value_at_zero;
    let mut alpha = initial_alpha.min(max_alpha);

    for iteration in 0..MAX_LINE_SEARCH_TRIALS {
        let point = evaluate(alpha)?;
        evaluations += 1;

        if !point.value.is_finite() || !point.derivative.is_finite() {
            return Ok(None);
        }

        if point.value > armijo_value(value_at_zero, derivative_at_zero, alpha)
            || (iteration > 0 && point.value >= previous_value)
        {
            return strong_wolfe_zoom(
                previous_alpha,
                alpha,
                value_at_zero,
                derivative_at_zero,
                previous_value,
                evaluate,
                evaluations,
            );
        }

        if point.derivative.abs() <= -WOLFE_CURVATURE * derivative_at_zero {
            return Ok(Some(StrongWolfeResult { point, evaluations }));
        }

        if point.derivative >= 0.0 {
            return strong_wolfe_zoom(
                alpha,
                previous_alpha,
                value_at_zero,
                derivative_at_zero,
                point.value,
                evaluate,
                evaluations,
            );
        }

        if alpha >= max_alpha {
            return Ok(None);
        }

        previous_alpha = alpha;
        previous_value = point.value;
        alpha = (2.0 * alpha).min(max_alpha);
    }

    Ok(None)
}

#[cfg(test)]
pub(super) fn strong_wolfe_zoom<F>(
    mut low_alpha: f64,
    mut high_alpha: f64,
    value_at_zero: f64,
    derivative_at_zero: f64,
    mut low_value: f64,
    mut evaluate: F,
    mut evaluations: usize,
) -> Result<Option<StrongWolfeResult>, OptimError>
where
    F: FnMut(f64) -> Result<LineSearchPoint1D, OptimError>,
{
    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        let alpha = 0.5 * (low_alpha + high_alpha);
        if (high_alpha - low_alpha).abs() <= MIN_STEP.max(MIN_STEP * alpha.abs()) {
            return Ok(None);
        }

        let point = evaluate(alpha)?;
        evaluations += 1;

        if !point.value.is_finite() || !point.derivative.is_finite() {
            return Ok(None);
        }

        if point.value > armijo_value(value_at_zero, derivative_at_zero, alpha)
            || point.value >= low_value
        {
            high_alpha = alpha;
            continue;
        }

        if point.derivative.abs() <= -WOLFE_CURVATURE * derivative_at_zero {
            return Ok(Some(StrongWolfeResult { point, evaluations }));
        }

        if point.derivative * (high_alpha - low_alpha) >= 0.0 {
            high_alpha = low_alpha;
        }
        low_alpha = alpha;
        low_value = point.value;
    }

    Ok(None)
}

#[cfg(test)]
pub(super) fn armijo_value(value_at_zero: f64, derivative_at_zero: f64, alpha: f64) -> f64 {
    value_at_zero + ARMIJO * alpha * derivative_at_zero
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

#[cfg(test)]
pub(super) fn max_feasible_step(x: &[f64], direction: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let alpha = geometric_feasible_step(x, direction, lower, upper);
    if alpha.is_infinite() {
        1.0
    } else {
        alpha
    }
}

#[cfg(test)]
pub(super) fn max_feasible_step_allowing_unbounded(
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
    if alpha.is_infinite() {
        STRONG_WOLFE_UNBOUNDED_MAX_STEP
    } else {
        alpha.max(0.0)
    }
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
