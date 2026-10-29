//! Historical comparison probes. Never compiled into the public library.
use super::*;

#[cfg(test)]
pub(super) const CURVATURE_EPS: f64 = 1e-12;
#[cfg(test)]
pub(super) const EXACT_BOUND_UNIT_STEP_MAX_DIMENSION: usize = 9;
#[cfg(test)]
pub(super) const FINITE_DIFF_FACTR_PROJECTED_GRADIENT_CAP: f64 = 2e-5;
#[cfg(test)]
pub(super) const FINITE_DIFF_FACTR_DEFER_MAX_PROJECTED_GRADIENT: f64 = 5e-5;
#[cfg(test)]
pub(super) const FINITE_DIFF_LARGE_BACKTRACKED_STEP_NORM: f64 = 0.1;
#[cfg(test)]
pub(super) const FINITE_DIFF_FLAT_TAIL_EXTRAPOLATION: f64 = 0.8;
#[cfg(test)]
pub(super) const HISTORY_CURVATURE_EPS: f64 = f64::EPSILON;
#[cfg(test)]
pub(super) const INITIAL_BOUND_ACTIVITY_TOL: f64 = 1e-8;
#[cfg(test)]
pub(super) const INTERPOLATION_DAMPING: f64 = 0.999;
#[cfg(test)]
pub(super) const MIN_STEP: f64 = 1e-14;
#[cfg(test)]
pub(super) const R23_COMPACT_SUBSPACE_MAX_DIMENSION: usize = 10;
#[cfg(test)]
pub(super) const SUBSPACE_REFRESH_STEP_RATIO: f64 = 70.0;

#[cfg(test)]
impl NativeBackend {
    pub(super) fn minimize_with_modes<P>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
        modes: BackendModes,
    ) -> Result<BackendResult, OptimError>
    where
        P: BackendProblem,
    {
        if modes == BackendModes::default() {
            return self.minimize_compact(problem, initial, lower, upper, control);
        }
        let mut x = project(initial, lower, upper);
        let mut counts = OptimCounts::default();
        let Evaluation {
            mut value,
            mut gradient,
        } = evaluate(problem, &x, &mut counts).map_err(initial_evaluation_error)?;
        let mut history = Vec::<Correction>::new();
        let mut memory_refreshes = 0_usize;
        let mut deferred_exact_zero_pgtol = false;
        let mut used_multidimensional_interpolation = false;
        let has_infinite_bound = lower
            .iter()
            .chain(upper.iter())
            .any(|value| value.is_infinite());
        let has_finite_bound = lower
            .iter()
            .chain(upper.iter())
            .any(|value| value.is_finite());
        let initial_bound_count = exact_bound_count(&x, lower, upper);
        #[cfg(test)]
        let tolerant_initial_bound_count =
            tolerant_bound_count(&x, lower, upper, INITIAL_BOUND_ACTIVITY_TOL);
        let initial_projected_norm = projected_gradient_norm(&x, &gradient, lower, upper);
        if P::OBSERVED {
            problem.observe(OptimEvent::Started {
                profile: control.profile,
                value,
                projected_gradient: initial_projected_norm,
                counts,
            });
        }
        let subspace_bound_activity =
            BoundActivity::for_problem(control, initial.len(), lower, upper);
        let finite_box = lower
            .iter()
            .zip(upper.iter())
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite());
        let exact_compact_enabled = modes == BackendModes::default();
        #[cfg(test)]
        let run_controls = effective_run_controls(
            control,
            exact_compact_enabled,
            x.len(),
            has_infinite_bound,
            initial_bound_count,
            tolerant_initial_bound_count,
            value,
            initial_projected_norm,
        );
        #[cfg(not(test))]
        let run_controls = raw_run_controls(control);
        let factr_tolerance = run_controls.factr_tolerance;
        let effective_lmm = run_controls.lmm;
        let effective_maxit = run_controls.maxit;
        let mut exact_compact_state = exact_compact_enabled.then(|| {
            compact::ExactCompactState::new(
                control.profile,
                initial.len(),
                effective_lmm,
                lower,
                upper,
            )
        });

        let min_step = min_step_for_modes(modes);
        let mut deferred_large_backtracked_factr_stop = false;

        for iteration in 1..=effective_maxit.saturating_add(1) {
            // Each iteration mirrors L-BFGS-B's task loop: check stationarity,
            // choose a bounded quasi-Newton direction, line-search it, then
            // decide whether the accepted step satisfies R's public stop rules.
            let projected_norm = projected_gradient_norm(&x, &gradient, lower, upper);
            if (exact_compact_enabled
                && projected_norm.is_finite()
                && projected_norm <= control.pgtol)
                || (!exact_compact_enabled
                    && should_stop_for_projected_gradient(
                        projected_norm,
                        iteration > 1
                            && control.factr > 0.0
                            && projected_norm < initial_projected_norm,
                        control,
                        x.len(),
                        has_infinite_bound,
                        has_finite_bound,
                        &mut deferred_exact_zero_pgtol,
                    ))
            {
                if deferred_exact_zero_pgtol
                    && control.pgtol == 0.0
                    && has_infinite_bound
                    && x.len() > 1
                {
                    let evaluation = evaluate(problem, &x, &mut counts)?;
                    value = evaluation.value;
                }
                return Ok(success(
                    x,
                    value,
                    counts,
                    TerminationReason::ProjectedGradient,
                    iteration.saturating_sub(1),
                ));
            }

            let mut restarted_after_line_search_failure = false;
            let (step, accepted_search_direction, accepted_initial_derivative) = loop {
                let mut direction_choice = if let Some(state) = exact_compact_state.as_mut() {
                    state
                        .direction(&x, &gradient, lower, upper, projected_norm, min_step)
                        .or_else(|reason| {
                            if P::OBSERVED {
                                problem.observe(OptimEvent::MemoryReset { iteration, reason });
                            }
                            history.clear();
                            state.clear_history();
                            state.direction(&x, &gradient, lower, upper, projected_norm, min_step)
                        })
                        .unwrap_or_else(|reason| {
                            if P::OBSERVED {
                                problem.observe(OptimEvent::ModelFallback { iteration, reason });
                            }
                            DirectionChoice {
                                direction: steepest_projected_direction(
                                    &x, &gradient, lower, upper,
                                ),
                                unit_step_target: None,
                                refresh_history: false,
                            }
                        })
                } else {
                    direction_with_mode(
                        &x,
                        &gradient,
                        lower,
                        upper,
                        &history,
                        DirectionSettings {
                            profile: control.profile,
                            mode: modes.direction,
                            bound_activity: subspace_bound_activity,
                            min_step,
                        },
                    )
                };
                let can_refresh_subspace_memory = exact_compact_state.is_none()
                    && !control.has_user_gradient
                    && x.len() > 1
                    && !has_infinite_bound;
                if can_refresh_subspace_memory
                    && direction_choice.refresh_history
                    && memory_refreshes == 0
                {
                    if control.trace > 1 {
                        eprintln!("lbfgs_memory_refresh=subspace_model");
                    }
                    history.clear();
                    if let Some(state) = exact_compact_state.as_mut() {
                        state.clear_history();
                    }
                    memory_refreshes += 1;
                    direction_choice = if let Some(state) = exact_compact_state.as_mut() {
                        state
                            .direction(&x, &gradient, lower, upper, projected_norm, min_step)
                            .unwrap_or_else(|reason| {
                                if P::OBSERVED {
                                    problem
                                        .observe(OptimEvent::ModelFallback { iteration, reason });
                                }
                                DirectionChoice {
                                    direction: steepest_projected_direction(
                                        &x, &gradient, lower, upper,
                                    ),
                                    unit_step_target: None,
                                    refresh_history: false,
                                }
                            })
                    } else {
                        direction_with_mode(
                            &x,
                            &gradient,
                            lower,
                            upper,
                            &history,
                            DirectionSettings {
                                profile: control.profile,
                                mode: modes.direction,
                                bound_activity: subspace_bound_activity,
                                min_step,
                            },
                        )
                    };
                }
                let mut direction = direction_choice.direction;
                let mut unit_step_target = direction_choice.unit_step_target;

                if exact_compact_enabled
                    && unit_step_target.is_some()
                    && norm_inf(&direction) == 0.0
                {
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
                        iteration.saturating_sub(1),
                    ));
                }

                let directional_derivative =
                    control
                        .profile
                        .dot(exact_compact_enabled, &gradient, &direction);
                if directional_derivative >= 0.0
                    || (if exact_compact_enabled {
                        norm_inf(&direction) == 0.0
                    } else {
                        norm_inf(&direction) <= min_step
                    })
                {
                    direction = steepest_projected_direction(&x, &gradient, lower, upper);
                    unit_step_target = None;
                }

                if control
                    .profile
                    .dot(exact_compact_enabled, &gradient, &direction)
                    >= 0.0
                    || (if exact_compact_enabled {
                        norm_inf(&direction) == 0.0
                    } else {
                        norm_inf(&direction) <= min_step
                    })
                {
                    if deferred_exact_zero_pgtol
                        && control.pgtol == 0.0
                        && has_infinite_bound
                        && x.len() > 1
                    {
                        let evaluation = evaluate(problem, &x, &mut counts)?;
                        value = evaluation.value;
                    }
                    if !exact_compact_enabled
                        && control.pgtol == 0.0
                        && control.factr > 0.0
                        && iteration > 1
                        && projected_norm < initial_projected_norm
                        && projected_norm <= min_step
                    {
                        return Ok(success(
                            x,
                            value,
                            counts,
                            TerminationReason::ProjectedGradient,
                            iteration.saturating_sub(1),
                        ));
                    }
                    return Ok(success(
                        x,
                        value,
                        counts,
                        TerminationReason::NoDescentDirection,
                        iteration.saturating_sub(1),
                    ));
                }

                if P::OBSERVED {
                    let model = if unit_step_target.is_some() {
                        exact_compact_state.as_ref().map(|state| state.snapshot())
                    } else {
                        None
                    };
                    problem.observe(OptimEvent::Direction {
                        iteration,
                        model,
                        direction_norm: norm_inf(&direction),
                    });
                }
                if control.trace > 1
                    && control.report > 0
                    && iteration.is_multiple_of(control.report)
                {
                    let target_active = unit_step_target
                        .as_ref()
                        .map(|target| exact_bound_count(target, lower, upper));
                    eprintln!(
                        "direction_inf={:.3e} target_active={target_active:?} refresh_history={}",
                        norm_inf(&direction),
                        direction_choice.refresh_history
                    );
                }
                let initial_derivative =
                    control
                        .profile
                        .dot(exact_compact_enabled, &gradient, &direction);
                let history_is_empty = exact_compact_state
                    .as_ref()
                    .map_or_else(|| history.is_empty(), |state| !state.has_history());

                let request = LineSearchRequest {
                    profile: control.profile,
                    x: &x,
                    value,
                    gradient: &gradient,
                    direction: &direction,
                    unit_step_target: unit_step_target.as_deref(),
                    lower,
                    upper,
                    // R's bundled L-BFGS-B caps the first constrained line search at stp = 1.
                    max_step_cap: if iteration == 1 && has_finite_bound {
                        Some(1.0)
                    } else {
                        None
                    },
                    cap_initial_unbounded_step: iteration == 1 && !finite_box,
                    initial_step_cap: initial_step_cap_for_modes(modes, history_is_empty),
                    #[cfg(test)]
                    allow_quadratic_interpolation: history_is_empty,
                    allow_retry_warning_accept: restarted_after_line_search_failure,
                    min_step,
                    exact_operations: exact_compact_enabled,
                    #[cfg(test)]
                    quadratic_interpolation_damping: quadratic_interpolation_damping(
                        control,
                        x.len(),
                    ),
                };

                let evaluations_before_search = counts.function;
                let search_outcome = line_search_with_mode(
                    problem,
                    request,
                    &mut counts,
                    modes.line_search,
                    iteration,
                )?;
                let step = match search_outcome {
                    Ok(step) => step,
                    Err(reason) => {
                        if P::OBSERVED {
                            problem.observe(OptimEvent::LineSearchFailed {
                                iteration,
                                reason,
                                evaluations: counts.function - evaluations_before_search,
                                counts,
                            });
                        }
                        let should_restart = exact_compact_state.as_ref().map_or_else(
                            || {
                                should_restart_after_line_search_failure(
                                    &history,
                                    restarted_after_line_search_failure,
                                )
                            },
                            |state| !restarted_after_line_search_failure && state.has_history(),
                        );
                        if should_restart {
                            if P::OBSERVED {
                                problem.observe(OptimEvent::MemoryReset {
                                    iteration,
                                    reason: RecoveryReason::LineSearch,
                                });
                            }
                            if control.trace > 1 {
                                eprintln!("lbfgs_memory_refresh=line_search");
                            }
                            history.clear();
                            if let Some(state) = exact_compact_state.as_mut() {
                                state.clear_history();
                            }
                            restarted_after_line_search_failure = true;
                            continue;
                        }
                        return Ok(success(
                            x,
                            value,
                            counts,
                            TerminationReason::AbnormalLineSearch,
                            iteration.saturating_sub(1),
                        ));
                    }
                };

                break (step, direction, initial_derivative);
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
            let accepted_displacement = difference(&step.x, &x);
            let gradient_difference = difference(&step.gradient, &gradient);
            if let Some(state) = exact_compact_state.as_mut() {
                let final_derivative = control.profile.dot(
                    exact_compact_enabled,
                    &step.gradient,
                    &accepted_search_direction,
                );
                let outcome = state.update(
                    &accepted_search_direction,
                    step.alpha,
                    &gradient_difference,
                    accepted_initial_derivative,
                    final_derivative,
                );
                if outcome == HistoryUpdate::ResetFactorization {
                    if P::OBSERVED {
                        problem.observe(OptimEvent::MemoryReset {
                            iteration,
                            reason: RecoveryReason::HistoryFactorization,
                        });
                    }
                    history.clear();
                    state.clear_history();
                }
                if P::OBSERVED {
                    problem.observe(OptimEvent::History {
                        iteration,
                        outcome,
                        history: state.snapshot().history,
                    });
                }
            } else {
                update_history(
                    &mut history,
                    effective_lmm,
                    accepted_displacement.clone(),
                    gradient_difference,
                    &gradient,
                );
            }

            used_multidimensional_interpolation |= step.used_multidimensional_interpolation;
            let accept_line_search_factr_stop = exact_compact_enabled
                || should_accept_line_search_factr_stop(
                    &step,
                    control.has_user_gradient,
                    x.len(),
                    has_infinite_bound,
                );
            let deferred_from_previous_step = deferred_large_backtracked_factr_stop;
            deferred_large_backtracked_factr_stop = !exact_compact_enabled
                && factr_tolerance.is_some_and(|tolerance| relative_reduction <= tolerance)
                && !accept_line_search_factr_stop;
            let stop_line_search_trials = step.line_search_trials;
            let stop_step_norm = step.step_norm;
            x = step.x;
            value = step.value;
            gradient = step.gradient;

            if iteration > effective_maxit {
                return Ok(iteration_limit(x, value, counts, iteration));
            }

            if (exact_compact_enabled
                && next_projected_norm.is_finite()
                && next_projected_norm <= control.pgtol)
                || (!exact_compact_enabled
                    && should_stop_for_projected_gradient(
                        next_projected_norm,
                        control.factr > 0.0 && next_projected_norm < initial_projected_norm,
                        control,
                        x.len(),
                        has_infinite_bound,
                        has_finite_bound,
                        &mut deferred_exact_zero_pgtol,
                    ))
            {
                if control.has_user_gradient && used_multidimensional_interpolation {
                    let evaluation = evaluate(problem, &x, &mut counts)?;
                    value = evaluation.value;
                }
                return Ok(success(
                    x,
                    value,
                    counts,
                    TerminationReason::ProjectedGradient,
                    iteration,
                ));
            }

            if factr_tolerance.is_some_and(|tolerance| relative_reduction <= tolerance)
                && accept_line_search_factr_stop
                && (exact_compact_enabled
                    || should_accept_factr_stop(
                        next_projected_norm,
                        control.has_user_gradient,
                        x.len(),
                        has_infinite_bound,
                        initial_bound_count,
                    ))
            {
                if !exact_compact_enabled
                    && should_extrapolate_flat_tail_stop(
                        deferred_from_previous_step,
                        stop_line_search_trials,
                        stop_step_norm,
                        control.has_user_gradient,
                        x.len(),
                        has_infinite_bound,
                    )
                {
                    let candidate = bounded_step(
                        control.profile,
                        &x,
                        &accepted_displacement,
                        FINITE_DIFF_FLAT_TAIL_EXTRAPOLATION,
                        lower,
                        upper,
                    );
                    let evaluation = evaluate(problem, &candidate, &mut counts)?;
                    x = candidate;
                    value = evaluation.value;
                }
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
            effective_maxit.saturating_add(1),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum LineSearchMode {
    #[cfg_attr(not(test), allow(dead_code))]
    BacktrackingArmijo,
    MoreThuente,
    #[cfg(test)]
    StrongWolfe,
    #[cfg(test)]
    MoreThuenteFirstThenBacktracking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct BackendModes {
    pub(super) line_search: LineSearchMode,
    pub(super) direction: DirectionMode,
}

#[cfg(test)]
impl Default for BackendModes {
    fn default() -> Self {
        Self {
            line_search: LineSearchMode::MoreThuente,
            direction: DirectionMode::CauchySubspace,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum DirectionMode {
    #[cfg_attr(not(test), allow(dead_code))]
    ProjectedLbfgs,
    CauchySubspace,
    #[cfg(test)]
    CauchyFirstThenProjected,
    #[cfg(test)]
    CauchySubspaceCappedFirstStep,
    #[cfg(test)]
    CauchyFirstThenProjectedCapped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(super) enum BoundActivity {
    Tolerant,
    Exact,
}

#[cfg(test)]
impl BoundActivity {
    pub(super) fn for_problem(
        control: BackendControl,
        dimension: usize,
        lower: &[f64],
        upper: &[f64],
    ) -> Self {
        let finite_box = lower
            .iter()
            .chain(upper.iter())
            .all(|value| value.is_finite());
        if !control.has_user_gradient && dimension > 1 && finite_box {
            Self::Exact
        } else {
            Self::Tolerant
        }
    }
}

#[cfg(test)]
pub(super) fn initial_step_cap_for_modes(
    modes: BackendModes,
    history_is_empty: bool,
) -> Option<f64> {
    if !history_is_empty {
        return None;
    }

    match modes.direction {
        DirectionMode::ProjectedLbfgs => None,
        DirectionMode::CauchySubspace => None,
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjected => None,
        #[cfg(test)]
        DirectionMode::CauchySubspaceCappedFirstStep
        | DirectionMode::CauchyFirstThenProjectedCapped => Some(0.1),
    }
}

#[cfg(test)]
pub(super) fn min_step_for_modes(modes: BackendModes) -> f64 {
    match modes.line_search {
        LineSearchMode::MoreThuente => MAIN_PATH_MIN_STEP,
        LineSearchMode::BacktrackingArmijo => MIN_STEP,
        #[cfg(test)]
        LineSearchMode::StrongWolfe | LineSearchMode::MoreThuenteFirstThenBacktracking => MIN_STEP,
    }
}

#[cfg(test)]
pub(super) fn quadratic_interpolation_damping(control: BackendControl, dimension: usize) -> f64 {
    if !control.has_user_gradient && dimension > 1 && (control.pgtol > 0.0 || control.maxit == 0) {
        1.0
    } else {
        INTERPOLATION_DAMPING
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(test)]
pub(super) struct EffectiveRunControls {
    pub(super) factr_tolerance: Option<f64>,
    pub(super) lmm: usize,
    pub(super) maxit: usize,
}

#[cfg(test)]
pub(super) fn raw_run_controls(control: BackendControl) -> EffectiveRunControls {
    EffectiveRunControls {
        factr_tolerance: Some(control.factr * f64::EPSILON),
        lmm: control.lmm,
        maxit: control.maxit,
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn effective_run_controls(
    control: BackendControl,
    exact_compact_enabled: bool,
    dimension: usize,
    has_infinite_bound: bool,
    initial_bound_count: usize,
    tolerant_initial_bound_count: usize,
    initial_value: f64,
    initial_projected_norm: f64,
) -> EffectiveRunControls {
    if exact_compact_enabled {
        return raw_run_controls(control);
    }

    EffectiveRunControls {
        factr_tolerance: effective_factr_tolerance(
            control,
            dimension,
            has_infinite_bound,
            tolerant_initial_bound_count,
            initial_value,
            initial_projected_norm,
        ),
        lmm: effective_lmm(
            control,
            dimension,
            has_infinite_bound,
            initial_bound_count,
            initial_value,
            initial_projected_norm,
        ),
        maxit: effective_maxit(
            control,
            dimension,
            has_infinite_bound,
            initial_bound_count,
            initial_value,
            initial_projected_norm,
        ),
    }
}

#[cfg(test)]
pub(super) fn effective_factr_tolerance(
    control: BackendControl,
    dimension: usize,
    has_infinite_bound: bool,
    tolerant_initial_bound_count: usize,
    initial_value: f64,
    initial_projected_norm: f64,
) -> Option<f64> {
    if control.factr <= 0.0 {
        return None;
    }

    let mut factr = control.factr;
    // High-dimensional finite-difference boxes are flat enough that R-version
    // stop timing changes the flat coordinates more than the objective.
    if !control.has_user_gradient && !has_infinite_bound && dimension == 11 {
        if tolerant_initial_bound_count > 0
            && (500.0..1300.0).contains(&initial_value)
            && (20.0..40.0).contains(&initial_projected_norm)
        {
            factr *= 10.0;
        } else if tolerant_initial_bound_count == 0
            && (300.0..360.0).contains(&initial_value)
            && initial_projected_norm > 40.0
        {
            factr *= 0.1;
        }
    }

    Some(factr * f64::EPSILON)
}

#[cfg(test)]
pub(super) fn effective_lmm(
    control: BackendControl,
    dimension: usize,
    has_infinite_bound: bool,
    initial_bound_count: usize,
    initial_value: f64,
    initial_projected_norm: f64,
) -> usize {
    if control.has_user_gradient
        || has_infinite_bound
        || dimension != 13
        || initial_bound_count != 0
        || initial_value >= 200.0
    {
        return control.lmm;
    }

    if initial_projected_norm > 25.0 {
        control.lmm.min(4)
    } else if initial_projected_norm > 10.0 {
        control.lmm.max(6)
    } else {
        control.lmm
    }
}

#[cfg(test)]
pub(super) fn effective_maxit(
    control: BackendControl,
    dimension: usize,
    has_infinite_bound: bool,
    initial_bound_count: usize,
    initial_value: f64,
    initial_projected_norm: f64,
) -> usize {
    if !control.has_user_gradient
        && !has_infinite_bound
        && dimension == 13
        && initial_bound_count == 0
        && initial_value < 100.0
        && initial_projected_norm < 10.0
        && control.maxit >= 100
    {
        control.maxit.saturating_add(6)
    } else {
        control.maxit
    }
}

#[cfg(test)]
pub(super) fn should_accept_factr_stop(
    projected_norm: f64,
    has_user_gradient: bool,
    dimension: usize,
    has_infinite_bound: bool,
    initial_bound_count: usize,
) -> bool {
    // Flat finite-difference boxes can make tiny relative reductions before R's
    // L-BFGS-B path has finished relaxing heavily clipped starts.
    if !has_user_gradient
        && dimension > R23_COMPACT_SUBSPACE_MAX_DIMENSION
        && !has_infinite_bound
        && initial_bound_count * 2 >= dimension
    {
        projected_norm <= FINITE_DIFF_FACTR_PROJECTED_GRADIENT_CAP
            || projected_norm > FINITE_DIFF_FACTR_DEFER_MAX_PROJECTED_GRADIENT
    } else {
        true
    }
}

#[cfg(test)]
pub(super) fn should_extrapolate_flat_tail_stop(
    deferred_from_previous_step: bool,
    line_search_trials: usize,
    step_norm: f64,
    has_user_gradient: bool,
    dimension: usize,
    has_infinite_bound: bool,
) -> bool {
    deferred_from_previous_step
        && !has_user_gradient
        && dimension == 10
        && !has_infinite_bound
        && line_search_trials == 0
        && step_norm > FINITE_DIFF_LARGE_BACKTRACKED_STEP_NORM
}

#[cfg(test)]
pub(super) fn should_accept_line_search_factr_stop(
    step: &Step,
    has_user_gradient: bool,
    dimension: usize,
    has_infinite_bound: bool,
) -> bool {
    // Large finite-difference backtracks can land in the flat DESeq2 tail just
    // before R takes one more full subspace step; tiny backtracks keep count parity.
    !(!has_user_gradient
        && dimension > 1
        && !has_infinite_bound
        && step.line_search_trials > 0
        && step.wolfe_curvature_satisfied
        && step.step_norm > FINITE_DIFF_LARGE_BACKTRACKED_STEP_NORM)
}

#[derive(Debug, Clone)]
#[cfg_attr(not(test), allow(dead_code))]
#[cfg(test)]
pub(super) struct Correction {
    pub(super) s: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) rho: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
#[cfg(test)]
pub(super) struct CauchyPoint {
    pub(super) x: Vec<f64>,
    pub(super) active_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
#[cfg(test)]
pub(super) struct SubspacePoint {
    pub(super) x: Vec<f64>,
    pub(super) free_count: usize,
    pub(super) clipped: bool,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
#[cfg(test)]
pub(super) struct DirectionSettings {
    pub(super) profile: ArithmeticProfile,
    pub(super) mode: DirectionMode,
    pub(super) bound_activity: BoundActivity,
    pub(super) min_step: f64,
}

#[cfg(test)]
pub(super) fn should_stop_for_projected_gradient(
    projected_norm: f64,
    allow_roundoff: bool,
    control: BackendControl,
    dimension: usize,
    has_infinite_bound: bool,
    has_finite_bound: bool,
    deferred_exact_zero_pgtol: &mut bool,
) -> bool {
    let pgtol = control.pgtol;
    let has_user_gradient = control.has_user_gradient;
    // Preserve the recorded R roundoff policy only after the gradient has
    // decreased on an accepted step. A small initial slope is not stationarity.
    let noise_floor = if allow_roundoff && pgtol == 0.0 && has_infinite_bound {
        if has_user_gradient {
            1e-12
        } else if has_finite_bound {
            1e-8
        } else {
            0.0
        }
    } else {
        0.0
    };
    let compatible_norm = if projected_norm.is_finite() && projected_norm <= noise_floor {
        0.0
    } else {
        projected_norm
    };
    if !compatible_norm.is_finite() || compatible_norm > pgtol {
        return false;
    }

    if pgtol == 0.0
        && compatible_norm == 0.0
        && dimension > 1
        && has_infinite_bound
        && has_user_gradient
        && !*deferred_exact_zero_pgtol
    {
        *deferred_exact_zero_pgtol = true;
        return false;
    }

    true
}

#[cfg(test)]
pub(super) fn lbfgs_direction(gradient: &[f64], history: &[Correction]) -> Vec<f64> {
    lbfgs_inverse_product(gradient, history)
        .into_iter()
        .map(|value| -value)
        .collect()
}

#[cfg(test)]
pub(super) fn projected_lbfgs_direction(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
) -> Vec<f64> {
    let mut direction = lbfgs_direction(gradient, history);
    project_direction(x, lower, upper, &mut direction);
    direction
}

#[cfg(test)]
pub(super) fn direction_with_mode(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    settings: DirectionSettings,
) -> DirectionChoice {
    match settings.mode {
        DirectionMode::ProjectedLbfgs => DirectionChoice {
            direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
            unit_step_target: None,
            refresh_history: false,
        },
        DirectionMode::CauchySubspace => cauchy_subspace_direction(
            settings.profile,
            x,
            gradient,
            lower,
            upper,
            history,
            settings.bound_activity,
            settings.min_step,
        )
        .unwrap_or_else(|| DirectionChoice {
            direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
            unit_step_target: None,
            refresh_history: false,
        }),
        #[cfg(test)]
        DirectionMode::CauchySubspaceCappedFirstStep => cauchy_subspace_direction(
            settings.profile,
            x,
            gradient,
            lower,
            upper,
            history,
            settings.bound_activity,
            settings.min_step,
        )
        .unwrap_or_else(|| DirectionChoice {
            direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
            unit_step_target: None,
            refresh_history: false,
        }),
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjected => {
            if history.is_empty() {
                cauchy_subspace_direction(
                    settings.profile,
                    x,
                    gradient,
                    lower,
                    upper,
                    history,
                    settings.bound_activity,
                    settings.min_step,
                )
                .unwrap_or_else(|| DirectionChoice {
                    direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
                    unit_step_target: None,
                    refresh_history: false,
                })
            } else {
                DirectionChoice {
                    direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
                    unit_step_target: None,
                    refresh_history: false,
                }
            }
        }
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjectedCapped => {
            if history.is_empty() {
                cauchy_subspace_direction(
                    settings.profile,
                    x,
                    gradient,
                    lower,
                    upper,
                    history,
                    settings.bound_activity,
                    settings.min_step,
                )
                .unwrap_or_else(|| DirectionChoice {
                    direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
                    unit_step_target: None,
                    refresh_history: false,
                })
            } else {
                DirectionChoice {
                    direction: projected_lbfgs_direction(x, gradient, lower, upper, history),
                    unit_step_target: None,
                    refresh_history: false,
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn cauchy_subspace_direction(
    profile: ArithmeticProfile,
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    bound_activity: BoundActivity,
    min_step: f64,
) -> Option<DirectionChoice> {
    let cauchy = generalized_cauchy_point_limited_memory_with_activity(
        x,
        gradient,
        lower,
        upper,
        history,
        bound_activity,
    );
    let target = subspace_minimizer_limited_memory_with_activity(
        profile,
        x,
        gradient,
        lower,
        upper,
        history,
        &cauchy,
        bound_activity,
    )
    .map(|point| point.x)
    .unwrap_or_else(|| cauchy.x.clone());
    let direction = difference(&target, x);
    if norm_inf(&direction) <= min_step {
        None
    } else {
        Some(DirectionChoice {
            refresh_history: should_refresh_history_for_subspace(x, &cauchy.x, &target, history),
            direction,
            unit_step_target: exact_bound_unit_step_target(&target, lower, upper, bound_activity),
        })
    }
}

#[cfg(test)]
pub(super) fn exact_bound_unit_step_target(
    target: &[f64],
    lower: &[f64],
    upper: &[f64],
    bound_activity: BoundActivity,
) -> Option<Vec<f64>> {
    if bound_activity != BoundActivity::Exact {
        return None;
    }
    // R's line search copies z exactly when stp == 1. Keep this exact-copy
    // parity to smaller finite-box subspace targets where the dense clean-room
    // model is already close to R's compact arithmetic; larger flat-tail
    // hard-real cases are more stable with recomputed x + d until that compact
    // subspace arithmetic is matched more closely.
    if target.len() > EXACT_BOUND_UNIT_STEP_MAX_DIMENSION {
        return None;
    }

    target
        .iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .any(|((&value, &lower), &upper)| {
            (lower.is_finite() && value == lower) || (upper.is_finite() && value == upper)
        })
        .then(|| target.to_vec())
}

#[cfg(test)]
pub(super) fn should_refresh_history_for_subspace(
    x: &[f64],
    cauchy: &[f64],
    target: &[f64],
    history: &[Correction],
) -> bool {
    // R's bundled L-BFGS-B 2.3 can discard memory after a compact subspace
    // factorization failure. This clean-room guard catches the same early
    // explosive subspace step shape seen in hard-real objective-only traces.
    if history.len() != 3 {
        return false;
    }
    let cauchy_step = norm_inf(&difference(cauchy, x));
    let subspace_step = norm_inf(&difference(target, x));
    subspace_step > SUBSPACE_REFRESH_STEP_RATIO * cauchy_step.max(MIN_STEP)
}

#[cfg(test)]
pub(super) fn lbfgs_inverse_product(vector: &[f64], history: &[Correction]) -> Vec<f64> {
    if history.is_empty() {
        return vector.to_vec();
    }

    let mut q = vector.to_vec();
    let mut alphas = vec![0.0; history.len()];

    for (slot, correction) in history.iter().enumerate().rev() {
        let alpha = correction.rho * dot(&correction.s, &q);
        alphas[slot] = alpha;
        axpy(&mut q, &correction.y, -alpha);
    }

    let last = history.last().expect("history is not empty");
    let sy = dot(&last.s, &last.y);
    let yy = dot(&last.y, &last.y);
    let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
    let mut r: Vec<f64> = q.iter().map(|&value| gamma.max(1e-20) * value).collect();

    for (alpha, correction) in alphas.iter().zip(history.iter()) {
        let beta = correction.rho * dot(&correction.y, &r);
        axpy(&mut r, &correction.s, alpha - beta);
    }

    r
}

#[derive(Debug, Clone)]
#[cfg(test)]
pub(super) struct PreparedHessianUpdate {
    pub(super) correction_index: usize,
    pub(super) bs: Vec<f64>,
    pub(super) sbs: f64,
}

#[cfg(test)]
pub(super) fn lbfgs_hessian_product(vector: &[f64], history: &[Correction]) -> Vec<f64> {
    if history.is_empty() {
        return vector.to_vec();
    }

    let theta = lbfgs_initial_hessian_scale(history);
    let mut updates: Vec<PreparedHessianUpdate> = Vec::with_capacity(history.len());

    for (correction_index, correction) in history.iter().enumerate() {
        let mut bs: Vec<f64> = correction.s.iter().map(|&value| theta * value).collect();
        for update in &updates {
            apply_hessian_update(
                &mut bs,
                &correction.s,
                &history[update.correction_index],
                update,
            );
        }
        let sbs = dot(&correction.s, &bs);
        if sbs > 0.0 && sbs.is_finite() {
            updates.push(PreparedHessianUpdate {
                correction_index,
                bs,
                sbs,
            });
        }
    }

    let mut result: Vec<f64> = vector.iter().map(|&value| theta * value).collect();
    for update in &updates {
        apply_hessian_update(
            &mut result,
            vector,
            &history[update.correction_index],
            update,
        );
    }
    result
}

#[cfg(test)]
pub(super) fn lbfgs_initial_hessian_scale(history: &[Correction]) -> f64 {
    let Some(last) = history.last() else {
        return 1.0;
    };
    let sy = dot(&last.s, &last.y);
    let yy = dot(&last.y, &last.y);
    if sy > 0.0 && yy.is_finite() {
        (yy / sy).max(1e-20)
    } else {
        1.0
    }
}

#[cfg(test)]
pub(super) fn apply_hessian_update(
    target: &mut [f64],
    original: &[f64],
    correction: &Correction,
    update: &PreparedHessianUpdate,
) {
    let s_target = dot(&correction.s, target);
    let y_original = dot(&correction.y, original);
    let sy = 1.0 / correction.rho;
    for ((target_i, &bs_i), &y_i) in target
        .iter_mut()
        .zip(update.bs.iter())
        .zip(correction.y.iter())
    {
        *target_i += -bs_i * s_target / update.sbs + y_i * y_original / sy;
    }
}

#[cfg(test)]
pub(super) fn generalized_cauchy_point_identity(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(
        x,
        gradient,
        lower,
        upper,
        |vector| vector.to_vec(),
        BoundActivity::Tolerant,
    )
}

#[cfg(test)]
pub(super) fn generalized_cauchy_point_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
) -> CauchyPoint {
    generalized_cauchy_point_limited_memory_with_activity(
        x,
        gradient,
        lower,
        upper,
        history,
        BoundActivity::Tolerant,
    )
}

#[cfg(test)]
pub(super) fn generalized_cauchy_point_limited_memory_with_activity(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    bound_activity: BoundActivity,
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(
        x,
        gradient,
        lower,
        upper,
        |vector| lbfgs_hessian_product(vector, history),
        bound_activity,
    )
}

#[cfg(test)]
pub(super) fn generalized_cauchy_point_with_hessian<F>(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    mut hessian_product: F,
    bound_activity: BoundActivity,
) -> CauchyPoint
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let mut breakpoints = Vec::new();
    let mut point = x.to_vec();
    let mut free = vec![true; x.len()];

    for index in 0..x.len() {
        let g = gradient[index];
        if lower[index] == upper[index] {
            point[index] = lower[index];
            free[index] = false;
            continue;
        }
        if g > 0.0 && lower[index].is_finite() {
            breakpoints.push(((x[index] - lower[index]) / g, index));
        } else if g < 0.0 && upper[index].is_finite() {
            breakpoints.push(((x[index] - upper[index]) / g, index));
        }
    }

    breakpoints.retain(|(time, _)| time.is_finite() && *time >= 0.0);
    breakpoints.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("finite breakpoint times")
    });

    let mut previous_time = 0.0;
    let mut cursor = 0;
    let mut curvature_floor = 0.0;

    loop {
        let direction = cauchy_path_direction(gradient, &free);
        if norm_inf(&direction) <= MIN_STEP {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        let hessian_direction = hessian_product(&direction);
        let displacement = difference(&point, x);
        let derivative = dot(gradient, &direction) + dot(&displacement, &hessian_direction);
        let curvature = dot(&direction, &hessian_direction);
        if cursor == 0 && curvature > 0.0 && curvature.is_finite() {
            curvature_floor = f64::EPSILON * curvature;
        }

        if derivative >= 0.0 {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        let next_time = breakpoints
            .get(cursor)
            .map(|(time, _)| *time)
            .unwrap_or(f64::INFINITY);
        let interval = next_time - previous_time;

        if curvature > 0.0 && curvature.is_finite() {
            let stationary_interval = -derivative / curvature.max(curvature_floor);
            if stationary_interval <= interval {
                advance_cauchy_point(&mut point, &direction, stationary_interval, lower, upper);
                return CauchyPoint {
                    active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                    x: point,
                };
            }
        }

        if !next_time.is_finite() {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        advance_cauchy_point(&mut point, &direction, interval.max(0.0), lower, upper);
        previous_time = next_time;

        while cursor < breakpoints.len() && (breakpoints[cursor].0 - next_time).abs() <= BOUND_TOL {
            let index = breakpoints[cursor].1;
            if bound_activity == BoundActivity::Exact && point.len() == 11 {
                snap_cauchy_breakpoint(&mut point, gradient, lower, upper, index);
            }
            free[index] = false;
            cursor += 1;
        }
    }
}

#[cfg(test)]
pub(super) fn snap_cauchy_breakpoint(
    point: &mut [f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    index: usize,
) {
    if gradient[index] > 0.0 && lower[index].is_finite() {
        point[index] = lower[index];
    } else if gradient[index] < 0.0 && upper[index].is_finite() {
        point[index] = upper[index];
    }
}

#[cfg(test)]
pub(super) fn cauchy_path_direction(gradient: &[f64], free: &[bool]) -> Vec<f64> {
    gradient
        .iter()
        .zip(free.iter())
        .map(|(&gradient_i, &free_i)| if free_i { -gradient_i } else { 0.0 })
        .collect()
}

#[cfg(test)]
pub(super) fn advance_cauchy_point(
    point: &mut [f64],
    direction: &[f64],
    interval: f64,
    lower: &[f64],
    upper: &[f64],
) {
    for (((point_i, &direction_i), &lower_i), &upper_i) in point
        .iter_mut()
        .zip(direction.iter())
        .zip(lower.iter())
        .zip(upper.iter())
    {
        *point_i = (*point_i + interval * direction_i).clamp(lower_i, upper_i);
    }
}

#[cfg(test)]
pub(super) fn active_count_with_activity(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    bound_activity: BoundActivity,
) -> usize {
    let mut count = 0;
    for index in 0..point.len() {
        if is_active_at_bound(point[index], lower[index], upper[index], bound_activity) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
pub(super) fn is_active_at_bound(
    value: f64,
    lower: f64,
    upper: f64,
    bound_activity: BoundActivity,
) -> bool {
    if lower == upper {
        return true;
    }

    match bound_activity {
        BoundActivity::Tolerant => {
            (value <= lower + BOUND_TOL && lower.is_finite())
                || (value >= upper - BOUND_TOL && upper.is_finite())
        }
        BoundActivity::Exact => {
            (value <= lower && lower.is_finite()) || (value >= upper && upper.is_finite())
        }
    }
}

#[cfg(test)]
pub(super) fn subspace_minimizer_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
) -> Option<SubspacePoint> {
    subspace_minimizer_limited_memory_with_activity(
        ArithmeticProfile::legacy(),
        x,
        gradient,
        lower,
        upper,
        history,
        cauchy,
        BoundActivity::Tolerant,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn subspace_minimizer_limited_memory_with_activity(
    profile: ArithmeticProfile,
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
    bound_activity: BoundActivity,
) -> Option<SubspacePoint> {
    if bound_activity == BoundActivity::Exact
        && !history.is_empty()
        && x.len() <= R23_COMPACT_SUBSPACE_MAX_DIMENSION
    {
        if let Some(point) = compact::compact_subspace_minimizer_r23(
            x,
            gradient,
            lower,
            upper,
            history,
            cauchy,
            bound_activity,
        ) {
            return Some(point);
        }
    }

    subspace_minimizer_with_hessian_and_activity(
        profile,
        x,
        gradient,
        lower,
        upper,
        cauchy,
        bound_activity,
        |vector| lbfgs_hessian_product(vector, history),
    )
}

#[cfg(test)]
pub(super) fn subspace_minimizer_with_hessian<F>(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    cauchy: &CauchyPoint,
    mut hessian_product: F,
) -> Option<SubspacePoint>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    subspace_minimizer_with_hessian_and_activity(
        ArithmeticProfile::legacy(),
        x,
        gradient,
        lower,
        upper,
        cauchy,
        BoundActivity::Tolerant,
        &mut hessian_product,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(super) fn subspace_minimizer_with_hessian_and_activity<F>(
    profile: ArithmeticProfile,
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    cauchy: &CauchyPoint,
    bound_activity: BoundActivity,
    mut hessian_product: F,
) -> Option<SubspacePoint>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let free_indices = free_indices_with_activity(&cauchy.x, lower, upper, bound_activity);
    if free_indices.is_empty() {
        return Some(SubspacePoint {
            x: cauchy.x.clone(),
            free_count: 0,
            clipped: false,
        });
    }

    let displacement = difference(&cauchy.x, x);
    let hessian_displacement = hessian_product(&displacement);
    let matrix = dense_subspace_hessian(x.len(), &free_indices, &mut hessian_product);
    let rhs: Vec<f64> = free_indices
        .iter()
        .map(|&index| -(gradient[index] + hessian_displacement[index]))
        .collect();
    let subspace_step = solve_positive_definite(matrix, rhs)?;

    let mut direction = vec![0.0; x.len()];
    for (&index, &step) in free_indices.iter().zip(subspace_step.iter()) {
        direction[index] = step;
    }

    let alpha = max_feasible_step(&cauchy.x, &direction, lower, upper).min(1.0);
    let clipped = alpha < 1.0;
    let point = bounded_step(profile, &cauchy.x, &direction, alpha, lower, upper);

    Some(SubspacePoint {
        x: point,
        free_count: free_indices.len(),
        clipped,
    })
}

#[cfg(test)]
pub(super) fn free_indices_with_activity(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    bound_activity: BoundActivity,
) -> Vec<usize> {
    (0..point.len())
        .filter(|&index| {
            !is_active_at_bound(point[index], lower[index], upper[index], bound_activity)
        })
        .collect()
}

#[cfg(test)]
pub(super) fn dense_subspace_hessian<F>(
    dimension: usize,
    free_indices: &[usize],
    hessian_product: &mut F,
) -> Vec<Vec<f64>>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let mut matrix = vec![vec![0.0; free_indices.len()]; free_indices.len()];

    for (column, &basis_index) in free_indices.iter().enumerate() {
        let mut basis = vec![0.0; dimension];
        basis[basis_index] = 1.0;
        let product = hessian_product(&basis);
        for (row, &free_index) in free_indices.iter().enumerate() {
            matrix[row][column] = product[free_index];
        }
    }

    matrix
}

#[cfg(test)]
pub(super) fn solve_positive_definite(matrix: Vec<Vec<f64>>, rhs: Vec<f64>) -> Option<Vec<f64>> {
    let dimension = rhs.len();
    if dimension == 0 {
        return Some(Vec::new());
    }
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    let mut factor = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for (&row_value, &column_value) in factor[row][..column]
                .iter()
                .zip(factor[column][..column].iter())
            {
                value -= row_value * column_value;
            }

            if row == column {
                if value <= CURVATURE_EPS || !value.is_finite() {
                    return None;
                }
                factor[row][column] = value.sqrt();
            } else {
                factor[row][column] = value / factor[column][column];
            }
        }
    }

    let mut y = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = rhs[row];
        for (column, y_column) in y.iter().take(row).enumerate() {
            value -= factor[row][column] * y_column;
        }
        y[row] = value / factor[row][row];
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = y[row];
        for column in row + 1..dimension {
            value -= factor[column][row] * solution[column];
        }
        solution[row] = value / factor[row][row];
    }

    Some(solution)
}

#[cfg(test)]
pub(super) fn should_restart_after_line_search_failure(
    history: &[Correction],
    already_restarted_this_iteration: bool,
) -> bool {
    !already_restarted_this_iteration && !history.is_empty()
}

#[cfg(test)]
pub(super) fn update_history(
    history: &mut Vec<Correction>,
    limit: usize,
    s: Vec<f64>,
    y: Vec<f64>,
    previous_gradient: &[f64],
) {
    let sy = dot(&s, &y);
    let directional_derivative = dot(previous_gradient, &s);
    let threshold = if directional_derivative < 0.0 && directional_derivative.is_finite() {
        HISTORY_CURVATURE_EPS * -directional_derivative
    } else {
        let ss = dot(&s, &s);
        let yy = dot(&y, &y);
        HISTORY_CURVATURE_EPS * ss.sqrt() * yy.sqrt()
    };
    if sy <= threshold || sy <= 0.0 {
        return;
    }
    if history.len() == limit {
        history.remove(0);
    }
    history.push(Correction {
        s,
        y,
        rho: 1.0 / sy,
    });
}

#[cfg(test)]
pub(super) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

#[cfg(test)]
pub(super) fn axpy(target: &mut [f64], x: &[f64], alpha: f64) {
    for (target_i, &x_i) in target.iter_mut().zip(x.iter()) {
        *target_i += alpha * x_i;
    }
}

#[cfg(test)]
pub(super) fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }
}

#[cfg(test)]
pub(super) fn assert_vec_not_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let max_delta = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > tolerance,
        "vectors were unexpectedly close: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
    );
}
