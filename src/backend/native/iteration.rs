//! Production compact-model iteration and recovery policy.
use super::*;

impl NativeBackend {
    pub(super) fn minimize_compact<P: BackendProblem>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
    ) -> Result<BackendResult, OptimError> {
        let mut x = project(initial, lower, upper);
        let mut counts = OptimCounts::default();
        let Evaluation {
            mut value,
            mut gradient,
        } = evaluate(problem, &x, &mut counts).map_err(initial_evaluation_error)?;
        let initial_projected_norm = projected_gradient_norm(&x, &gradient, lower, upper);
        if P::OBSERVED {
            problem.observe(OptimEvent::Started {
                profile: control.profile,
                value,
                projected_gradient: initial_projected_norm,
                counts,
            });
        }
        // A stationary initial point does not need correction columns or
        // compact factors. Preserve its evaluation/events before returning.
        if initial_projected_norm.is_finite() && initial_projected_norm <= control.pgtol {
            return Ok(success(
                x,
                value,
                counts,
                TerminationReason::ProjectedGradient,
                0,
            ));
        }
        let has_finite_bound = lower.iter().chain(upper).any(|value| value.is_finite());
        let finite_box = lower
            .iter()
            .zip(upper)
            .all(|(l, u)| l.is_finite() && u.is_finite());
        let mut state = compact::ExactCompactState::new(
            control.profile,
            initial.len(),
            control.lmm,
            lower,
            upper,
        );

        for iteration in 1..=control.maxit.saturating_add(1) {
            let projected_norm = projected_gradient_norm(&x, &gradient, lower, upper);
            if projected_norm.is_finite() && projected_norm <= control.pgtol {
                return Ok(success(
                    x,
                    value,
                    counts,
                    TerminationReason::ProjectedGradient,
                    iteration - 1,
                ));
            }
            let mut retried_search = false;
            let (step, direction, initial_derivative) = loop {
                let choice = state
                    .direction(
                        &x,
                        &gradient,
                        lower,
                        upper,
                        projected_norm,
                        MAIN_PATH_MIN_STEP,
                    )
                    .or_else(|reason| {
                        if P::OBSERVED {
                            problem.observe(OptimEvent::MemoryReset { iteration, reason });
                        }
                        state.clear_history();
                        state.direction(
                            &x,
                            &gradient,
                            lower,
                            upper,
                            projected_norm,
                            MAIN_PATH_MIN_STEP,
                        )
                    })
                    .unwrap_or_else(|reason| {
                        if P::OBSERVED {
                            problem.observe(OptimEvent::ModelFallback { iteration, reason });
                        }
                        DirectionChoice {
                            direction: steepest_projected_direction(&x, &gradient, lower, upper),
                            unit_step_target: None,
                            refresh_history: false,
                        }
                    });
                let mut direction = choice.direction;
                let mut target = choice.unit_step_target;
                if target.is_some() && norm_inf(&direction) == 0.0 {
                    if P::OBSERVED {
                        problem.observe(OptimEvent::LineSearchFailed {
                            iteration,
                            reason: LineSearchFailure::UnrepresentableTarget,
                            evaluations: 0,
                            counts,
                        });
                    }
                    return Ok(success(
                        x,
                        value,
                        counts,
                        TerminationReason::AbnormalLineSearch,
                        iteration - 1,
                    ));
                }
                let derivative = control.profile.dot(true, &gradient, &direction);
                if derivative >= 0.0 || norm_inf(&direction) == 0.0 {
                    direction = steepest_projected_direction(&x, &gradient, lower, upper);
                    target = None;
                }
                if control.profile.dot(true, &gradient, &direction) >= 0.0
                    || norm_inf(&direction) == 0.0
                {
                    return Ok(success(
                        x,
                        value,
                        counts,
                        TerminationReason::NoDescentDirection,
                        iteration - 1,
                    ));
                }
                if P::OBSERVED {
                    problem.observe(OptimEvent::Direction {
                        iteration,
                        model: target.as_ref().map(|_| state.snapshot()),
                        direction_norm: norm_inf(&direction),
                    });
                }
                if control.trace > 1
                    && control.report > 0
                    && iteration.is_multiple_of(control.report)
                {
                    let target_active = target
                        .as_ref()
                        .map(|point| exact_bound_count(point, lower, upper));
                    eprintln!(
                        "direction_inf={:.3e} target_active={target_active:?} refresh_history={}",
                        norm_inf(&direction),
                        choice.refresh_history
                    );
                }
                let initial_derivative = control.profile.dot(true, &gradient, &direction);
                let request = LineSearchRequest {
                    profile: control.profile,
                    x: &x,
                    value,
                    gradient: &gradient,
                    direction: &direction,
                    unit_step_target: target.as_deref(),
                    lower,
                    upper,
                    max_step_cap: if iteration == 1 && has_finite_bound {
                        Some(1.0)
                    } else {
                        None
                    },
                    cap_initial_unbounded_step: iteration == 1 && !finite_box,
                    initial_step_cap: None,
                    #[cfg(test)]
                    allow_quadratic_interpolation: !state.has_history(),
                    allow_retry_warning_accept: retried_search,
                    min_step: MAIN_PATH_MIN_STEP,
                    exact_operations: true,
                    #[cfg(test)]
                    quadratic_interpolation_damping: quadratic_interpolation_damping(
                        control,
                        x.len(),
                    ),
                };
                let evaluations_before_search = counts.function;
                match more_thuente_line_search(problem, request, &mut counts)? {
                    Ok(step) => break (step, direction, initial_derivative),
                    Err(reason) => {
                        if P::OBSERVED {
                            problem.observe(OptimEvent::LineSearchFailed {
                                iteration,
                                reason,
                                evaluations: counts.function - evaluations_before_search,
                                counts,
                            });
                        }
                        if !retried_search && state.has_history() {
                            if P::OBSERVED {
                                problem.observe(OptimEvent::MemoryReset {
                                    iteration,
                                    reason: RecoveryReason::LineSearch,
                                });
                            }
                            if control.trace > 1 {
                                eprintln!("lbfgs_memory_refresh=line_search");
                            }
                            state.clear_history();
                            retried_search = true;
                            continue;
                        }
                        return Ok(success(
                            x,
                            value,
                            counts,
                            TerminationReason::AbnormalLineSearch,
                            iteration - 1,
                        ));
                    }
                }
            };
            let relative_reduction = relative_objective_reduction(value, step.value);
            let next_projected_norm =
                projected_gradient_norm(&step.x, &step.gradient, lower, upper);
            maybe_trace(
                iteration,
                step.value,
                relative_reduction,
                next_projected_norm,
                &step,
                control,
            );
            if P::OBSERVED {
                problem.observe(OptimEvent::Accepted {
                    iteration,
                    value: step.value,
                    projected_gradient: next_projected_norm,
                    alpha: step.alpha,
                    max_alpha: step.max_alpha,
                    step_norm: step.step_norm,
                    search_evaluations: step.line_search_trials + 1,
                    acceptance: step.acceptance,
                    initial_derivative: step.initial_derivative,
                    final_derivative: step.final_derivative,
                    wolfe_curvature: step.wolfe_curvature_satisfied,
                    counts,
                });
            }
            let gradient_difference = difference(&step.gradient, &gradient);
            let final_derivative = control.profile.dot(true, &step.gradient, &direction);
            let outcome = state.update(
                &direction,
                step.alpha,
                &gradient_difference,
                initial_derivative,
                final_derivative,
            );
            if outcome == HistoryUpdate::ResetFactorization {
                if P::OBSERVED {
                    problem.observe(OptimEvent::MemoryReset {
                        iteration,
                        reason: RecoveryReason::HistoryFactorization,
                    });
                }
                state.clear_history();
            }
            if P::OBSERVED {
                problem.observe(OptimEvent::History {
                    iteration,
                    outcome,
                    history: state.snapshot().history,
                });
            }
            x = step.x;
            value = step.value;
            gradient = step.gradient;
            if iteration > control.maxit {
                return Ok(iteration_limit(x, value, counts, iteration));
            }
            if next_projected_norm.is_finite() && next_projected_norm <= control.pgtol {
                return Ok(success(
                    x,
                    value,
                    counts,
                    TerminationReason::ProjectedGradient,
                    iteration,
                ));
            }
            if relative_reduction <= control.factr * f64::EPSILON {
                return Ok(success(
                    x,
                    value,
                    counts,
                    TerminationReason::RelativeReduction,
                    iteration,
                ));
            }
        }
        Ok(iteration_limit(
            x,
            value,
            counts,
            control.maxit.saturating_add(1),
        ))
    }
}
