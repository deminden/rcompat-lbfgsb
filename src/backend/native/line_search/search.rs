use super::*;

pub(in crate::backend::native) fn more_thuente_line_search<P>(
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

pub(in crate::backend::native) fn more_thuente_sufficient_decrease(
    value_at_zero: f64,
    decrease_test: f64,
    alpha: f64,
) -> f64 {
    value_at_zero + alpha * decrease_test
}

pub(in crate::backend::native) fn more_thuente_enters_stage_two(
    stage_one: bool,
    value: f64,
    sufficient_decrease: f64,
    derivative: f64,
) -> bool {
    stage_one && value <= sufficient_decrease && derivative >= 0.0
}

pub(in crate::backend::native) fn more_thuente_retry_warning(
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

pub(in crate::backend::native) fn more_thuente_retry_restore(
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

pub(in crate::backend::native) fn more_thuente_accepts_warning(
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

pub(in crate::backend::native) fn more_thuente_restores_best(
    exact_operations: bool,
    allow_retry_warning_accept: bool,
    best_alpha: f64,
    other_alpha: f64,
    bracketed: bool,
) -> bool {
    (exact_operations || allow_retry_warning_accept)
        && more_thuente_retry_restore(best_alpha, other_alpha, bracketed)
}
