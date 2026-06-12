use super::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TraceFixture {
    fixture: String,
    source_fixture: String,
    trace_kind: String,
    first_points: Vec<Vec<f64>>,
}

fn assert_mirrored_steps(right: Option<f64>, left: Option<f64>) {
    let right = right.expect("right-hand step");
    let left = left.expect("left-hand mirrored step");
    assert!(
        (right + left).abs() <= 1e-14,
        "right={right:?}, left={left:?}"
    );
}

#[test]
fn relative_reduction_uses_signed_decrease_like_r() {
    assert!((relative_objective_reduction(10.0, 9.0) - 0.1).abs() <= 1e-15);
    assert!((relative_objective_reduction(10.0, 10.5) + (0.5 / 10.5)).abs() <= 1e-15);
    assert!((relative_objective_reduction(0.25, 0.5) + 0.25).abs() <= 1e-15);
}

#[test]
fn factr_stop_defers_small_gradient_band_for_clipped_finite_difference_boxes() {
    assert!(!should_accept_factr_stop(3.0e-5, false, 13, false, 12));
    assert!(should_accept_factr_stop(1.0e-5, false, 13, false, 12));
    assert!(should_accept_factr_stop(6.0e-5, false, 13, false, 12));
    assert!(should_accept_factr_stop(3.0e-5, false, 13, false, 2));
    assert!(should_accept_factr_stop(3.0e-5, true, 13, false, 12));
    assert!(should_accept_factr_stop(3.0e-5, false, 13, true, 12));
}

#[test]
fn factr_stop_defers_only_large_objective_only_backtracked_steps() {
    let mut step = Step {
        x: vec![0.0],
        value: 0.0,
        gradient: vec![0.0],
        line_search_trials: 1,
        alpha: 0.25,
        max_alpha: 1.0,
        step_norm: 0.2,
        curvature_ratio: 1e-4,
        wolfe_curvature_satisfied: true,
        used_multidimensional_interpolation: false,
    };

    assert!(!should_accept_line_search_factr_stop(
        &step, false, 10, false
    ));
    assert!(should_accept_line_search_factr_stop(&step, true, 10, false));
    assert!(should_accept_line_search_factr_stop(&step, false, 10, true));

    step.step_norm = 1e-3;
    assert!(should_accept_line_search_factr_stop(
        &step, false, 10, false
    ));

    step.step_norm = 0.2;
    step.line_search_trials = 0;
    assert!(should_accept_line_search_factr_stop(
        &step, false, 10, false
    ));

    step.line_search_trials = 1;
    step.wolfe_curvature_satisfied = false;
    assert!(should_accept_line_search_factr_stop(
        &step, false, 10, false
    ));
}

#[test]
fn flat_tail_extrapolation_requires_previous_large_backtracked_deferral() {
    assert!(should_extrapolate_flat_tail_stop(
        true, 0, 0.2, false, 10, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        false, 0, 0.2, false, 10, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        true, 1, 0.2, false, 10, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        true, 0, 1e-3, false, 10, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        true, 0, 0.2, false, 13, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        true, 0, 0.2, true, 10, false
    ));
    assert!(!should_extrapolate_flat_tail_stop(
        true, 0, 0.2, false, 10, true
    ));
}

#[test]
fn projected_gradient_norm_caps_components_by_bound_distance_like_r() {
    let norm = projected_gradient_norm(
        &[0.99, -1.99, 0.0, 2.0],
        &[-10.0, 8.0, 0.005, -3.0],
        &[-1.0, -2.0, f64::NEG_INFINITY, 2.0],
        &[1.0, 3.0, f64::INFINITY, 2.0],
    );

    assert!((norm - 0.01).abs() <= 1e-15);
}

#[test]
fn line_search_failure_restart_requires_existing_history_once() {
    let history = vec![Correction {
        s: vec![1.0],
        y: vec![1.0],
        rho: 1.0,
    }];

    assert!(should_restart_after_line_search_failure(&history, false));
    assert!(!should_restart_after_line_search_failure(&history, true));
    assert!(!should_restart_after_line_search_failure(&[], false));
}

#[test]
fn project_direction_keeps_free_quasi_newton_components() {
    let mut direction = vec![1.0, -2.0];
    project_direction(
        &[0.0, 0.0],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &mut direction,
    );
    assert_eq!(direction, vec![1.0, -2.0]);
}

#[test]
fn project_direction_blocks_outward_bound_components() {
    let mut direction = vec![-1.0, 2.0, 1.0];
    project_direction(
        &[0.0, 1.0, 3.0],
        &[0.0, 0.0, 3.0],
        &[5.0, 1.0, 3.0],
        &mut direction,
    );
    assert_eq!(direction, vec![0.0, 0.0, 0.0]);
}

#[test]
fn project_direction_blocks_outward_components_with_roundoff_tolerance() {
    let mut direction = vec![-1.0, 2.0];
    project_direction(
        &[4.0e-16, 1.0 - 4.0e-16],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &mut direction,
    );

    assert_eq!(direction, vec![0.0, 0.0]);
}

#[test]
fn subspace_free_set_requires_exact_bound_hit() {
    let lower = vec![0.0, 0.0, 0.0, 0.0];
    let upper = vec![1.0, 1.0, 1.0, 1.0];
    let point = vec![4.0e-16, 1.0 - 4.0e-16, 0.0, 1.0];

    assert_eq!(
        free_indices_with_activity(&point, &lower, &upper, BoundActivity::Exact),
        vec![0, 1]
    );
    assert_eq!(
        active_count_with_activity(&point, &lower, &upper, BoundActivity::Exact),
        2
    );
}

#[test]
fn exact_bound_activity_is_limited_to_objective_only_finite_boxes() {
    let objective_only = BackendControl {
        has_user_gradient: false,
        maxit: 100,
        factr: 1e7,
        pgtol: 0.0,
        lmm: 5,
        trace: 0,
        report: 10,
    };
    let supplied_gradient = BackendControl {
        has_user_gradient: true,
        maxit: 100,
        factr: 1e7,
        pgtol: 0.0,
        lmm: 5,
        trace: 0,
        report: 10,
    };

    assert_eq!(
        BoundActivity::for_problem(objective_only, 2, &[0.0, 0.0], &[1.0, 1.0]),
        BoundActivity::Exact
    );
    assert_eq!(
        BoundActivity::for_problem(supplied_gradient, 2, &[0.0, 0.0], &[1.0, 1.0]),
        BoundActivity::Tolerant
    );
    assert_eq!(
        BoundActivity::for_problem(objective_only, 2, &[f64::NEG_INFINITY, 0.0], &[1.0, 1.0]),
        BoundActivity::Tolerant
    );
}

#[test]
fn history_update_accepts_machine_epsilon_scale_positive_curvature() {
    let mut history = Vec::new();

    update_history(
        &mut history,
        5,
        vec![1.0, 0.0],
        vec![1e-13, 1.0],
        &[-1.0, 0.0],
    );

    assert_eq!(history.len(), 1);
}

#[test]
fn history_update_uses_directional_algorithm_778_skip_threshold() {
    let mut history = Vec::new();

    update_history(
        &mut history,
        5,
        vec![1.0, 0.0],
        vec![1e-18, 0.0],
        &[-1.0, 0.0],
    );

    assert!(history.is_empty());
}

#[test]
fn more_thuente_stage_two_waits_for_nonnegative_directional_derivative() {
    assert!(!more_thuente_enters_stage_two(true, 0.0, 0.0, -1e-12));
    assert!(more_thuente_enters_stage_two(true, 0.0, 0.0, 0.0));
    assert!(!more_thuente_enters_stage_two(false, 0.0, 0.0, 1.0));
}

#[test]
fn more_thuente_retry_warning_requires_narrow_bracket() {
    assert!(more_thuente_retry_warning(10.0, 14.9, true));
    assert!(!more_thuente_retry_warning(10.0, 21.0, true));
    assert!(!more_thuente_retry_warning(10.0, 14.9, false));
    assert!(!more_thuente_retry_warning(0.0, 0.0, true));
}

#[test]
fn more_thuente_cubic_steps_are_mirror_symmetric() {
    let right_best = MoreThuentePoint {
        alpha: 0.0,
        value: 0.0,
        derivative: -1.0,
    };
    let left_best = MoreThuentePoint {
        alpha: 0.0,
        value: 0.0,
        derivative: 1.0,
    };

    assert_mirrored_steps(
        more_thuente_cubic_case_two(
            right_best,
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.25,
                derivative: 0.2,
            },
        ),
        more_thuente_cubic_case_two(
            left_best,
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.25,
                derivative: -0.2,
            },
        ),
    );

    assert_mirrored_steps(
        more_thuente_cubic_case_three(
            right_best,
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.5,
                derivative: -0.1,
            },
        ),
        more_thuente_cubic_case_three(
            left_best,
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.5,
                derivative: 0.1,
            },
        ),
    );

    assert_mirrored_steps(
        more_thuente_cubic_case_four(
            MoreThuentePoint {
                alpha: 2.0,
                value: 0.5,
                derivative: 0.7,
            },
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.25,
                derivative: -1.2,
            },
        ),
        more_thuente_cubic_case_four(
            MoreThuentePoint {
                alpha: -2.0,
                value: 0.5,
                derivative: -0.7,
            },
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.25,
                derivative: 1.2,
            },
        ),
    );
}

#[test]
fn identity_cauchy_point_is_unconstrained_newton_step() {
    let cauchy = generalized_cauchy_point_identity(
        &[1.0, -2.0],
        &[0.25, -0.5],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
    );
    assert_eq!(cauchy.active_count, 0);
    assert_vec_close(&cauchy.x, &[0.75, -1.5], 1e-15);
}

#[test]
fn identity_cauchy_point_stops_at_box_when_stationary_point_is_beyond_bounds() {
    let cauchy = generalized_cauchy_point_identity(
        &[-1.2, 1.0],
        &[-215.6, -88.0],
        &[-5.0, -5.0],
        &[5.0, 5.0],
    );
    assert_eq!(cauchy.active_count, 2);
    assert_vec_close(&cauchy.x, &[5.0, 5.0], 1e-15);
}

#[test]
fn identity_cauchy_point_handles_partial_activation() {
    let cauchy =
        generalized_cauchy_point_identity(&[0.0, 0.0], &[1.0, 1.0], &[-0.9, -5.0], &[5.0, 5.0]);
    assert_eq!(cauchy.active_count, 1);
    assert_vec_close(&cauchy.x, &[-0.9, -1.0], 1e-15);
}

#[test]
fn direct_hessian_product_inverts_two_loop_inverse_product() {
    let history = vec![
        Correction {
            s: vec![1.0, 0.0],
            y: vec![2.0, 0.0],
            rho: 0.5,
        },
        Correction {
            s: vec![0.0, 1.0],
            y: vec![0.0, 3.0],
            rho: 1.0 / 3.0,
        },
    ];

    let vector = vec![4.0, 9.0];
    let hessian_vector = lbfgs_hessian_product(&vector, &history);
    assert_vec_close(&hessian_vector, &[8.0, 27.0], 1e-14);

    let inverse_hessian_vector = lbfgs_inverse_product(&hessian_vector, &history);
    assert_vec_close(&inverse_hessian_vector, &vector, 1e-14);
}

#[test]
fn direct_hessian_product_satisfies_latest_secant_condition() {
    let history = vec![
        Correction {
            s: vec![1.0, 0.0],
            y: vec![2.0, 1.0],
            rho: 0.5,
        },
        Correction {
            s: vec![0.0, 1.0],
            y: vec![1.0, 3.0],
            rho: 1.0 / 3.0,
        },
    ];

    let latest = history.last().expect("history fixture");
    let hessian_s = lbfgs_hessian_product(&latest.s, &history);
    assert_vec_close(&hessian_s, &latest.y, 1e-14);
}

#[test]
fn direct_hessian_product_inverts_two_loop_for_nonorthogonal_history() {
    let history = vec![
        Correction {
            s: vec![1.0, 0.0],
            y: vec![2.0, 0.5],
            rho: 0.5,
        },
        Correction {
            s: vec![0.25, 1.0],
            y: vec![1.0, 3.0],
            rho: 1.0 / 3.25,
        },
    ];

    for vector in [vec![4.0, 9.0], vec![-1.25, 0.75], vec![0.0, 2.5]] {
        let inverse_hessian_vector = lbfgs_inverse_product(&vector, &history);
        let hessian_inverse_hessian_vector =
            lbfgs_hessian_product(&inverse_hessian_vector, &history);
        assert_vec_close(&hessian_inverse_hessian_vector, &vector, 1e-12);
    }
}

#[test]
fn limited_memory_cauchy_point_uses_model_curvature() {
    let history = vec![Correction {
        s: vec![1.0, 0.0],
        y: vec![2.0, 0.0],
        rho: 0.5,
    }];

    let cauchy = generalized_cauchy_point_limited_memory(
        &[0.0, 0.0],
        &[2.0, 0.0],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &history,
    );

    assert_eq!(cauchy.active_count, 0);
    assert_vec_close(&cauchy.x, &[-1.0, 0.0], 1e-14);
}

#[test]
fn unbounded_cauchy_subspace_step_matches_two_loop_direction() {
    let history = vec![
        Correction {
            s: vec![1.0, 0.0],
            y: vec![2.0, 0.5],
            rho: 0.5,
        },
        Correction {
            s: vec![0.25, 1.0],
            y: vec![1.0, 3.0],
            rho: 1.0 / 3.25,
        },
    ];
    let x = vec![0.2, -0.4];
    let gradient = vec![3.0, -2.0];
    let lower = vec![f64::NEG_INFINITY; 2];
    let upper = vec![f64::INFINITY; 2];

    let cauchy = generalized_cauchy_point_limited_memory(&x, &gradient, &lower, &upper, &history);
    let subspace =
        subspace_minimizer_limited_memory(&x, &gradient, &lower, &upper, &history, &cauchy)
            .expect("positive definite full-space model");
    let model_direction = difference(&subspace.x, &x);
    let two_loop_direction = lbfgs_direction(&gradient, &history);

    assert_eq!(cauchy.active_count, 0);
    assert_eq!(subspace.free_count, 2);
    assert!(!subspace.clipped);
    assert_vec_close(&model_direction, &two_loop_direction, 1e-12);
}

#[test]
fn subspace_refresh_guard_requires_three_corrections_and_large_step_ratio() {
    let history = vec![
        Correction {
            s: vec![1.0],
            y: vec![1.0],
            rho: 1.0,
        };
        3
    ];

    assert!(should_refresh_history_for_subspace(
        &[0.0],
        &[1.0],
        &[71.0],
        &history
    ));
    assert!(!should_refresh_history_for_subspace(
        &[0.0],
        &[1.0],
        &[69.0],
        &history
    ));
    assert!(!should_refresh_history_for_subspace(
        &[0.0],
        &[1.0],
        &[71.0],
        &history[..2]
    ));
}

#[test]
fn subspace_minimizer_solves_free_block_after_cauchy_point() {
    let cauchy = CauchyPoint {
        x: vec![-0.5, 1.0],
        active_count: 1,
    };

    let subspace = subspace_minimizer_with_hessian(
        &[0.0, 0.0],
        &[5.0, -4.0],
        &[-0.5, -10.0],
        &[10.0, 10.0],
        &cauchy,
        dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
    )
    .expect("positive definite subspace");

    assert_eq!(subspace.free_count, 1);
    assert!(!subspace.clipped);
    assert_vec_close(&subspace.x, &[-0.5, 2.25], 1e-14);
}

#[test]
fn subspace_minimizer_clips_to_remaining_bounds() {
    let cauchy = CauchyPoint {
        x: vec![-0.5, 1.0],
        active_count: 1,
    };

    let subspace = subspace_minimizer_with_hessian(
        &[0.0, 0.0],
        &[5.0, -4.0],
        &[-0.5, -10.0],
        &[10.0, 2.0],
        &cauchy,
        dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
    )
    .expect("positive definite subspace");

    assert_eq!(subspace.free_count, 1);
    assert!(subspace.clipped);
    assert_vec_close(&subspace.x, &[-0.5, 2.0], 1e-14);
}

#[test]
fn subspace_minimizer_returns_cauchy_point_when_all_variables_are_active() {
    let cauchy = CauchyPoint {
        x: vec![-0.5, 2.0],
        active_count: 2,
    };

    let subspace = subspace_minimizer_with_hessian(
        &[0.0, 0.0],
        &[5.0, -4.0],
        &[-0.5, -10.0],
        &[10.0, 2.0],
        &cauchy,
        dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
    )
    .expect("no free variables still yields a subspace point");

    assert_eq!(subspace.free_count, 0);
    assert!(!subspace.clipped);
    assert_vec_close(&subspace.x, &cauchy.x, 1e-14);
}

#[test]
fn line_search_reports_zero_trials_for_accepted_full_step() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[1.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[f64::INFINITY],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("line search succeeds")
    .expect("descent step");

    assert_eq!(step.line_search_trials, 0);
    assert_eq!(counts.function, 1);
    assert_eq!(counts.gradient, 1);
    assert_vec_close(&step.x, &[1.0], 1e-15);
    assert!((step.alpha - 1.0).abs() <= 1e-15);
    assert!((step.max_alpha - 1.0).abs() <= 1e-15);
    assert!((step.step_norm - 1.0).abs() <= 1e-15);
    assert!((step.curvature_ratio - 0.5).abs() <= 1e-15);
    assert!(step.wolfe_curvature_satisfied);
}

#[test]
fn line_search_candidate_copies_exact_unit_step_target() {
    let x = [-25.060655267031027];
    let target = [30.0];
    let direction = [target[0] - x[0]];
    let recomputed = bounded_step(&x, &direction, 1.0, &[-30.0], &[30.0]);
    assert_ne!(recomputed[0].to_bits(), target[0].to_bits());

    let candidate = line_search_candidate(
        &LineSearchRequest {
            x: &x,
            value: 0.0,
            gradient: &[-1.0],
            direction: &direction,
            unit_step_target: Some(&target),
            lower: &[-30.0],
            upper: &[30.0],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        1.0,
    );

    assert_eq!(candidate[0].to_bits(), target[0].to_bits());
}

#[test]
fn exact_bound_unit_step_target_ignores_interior_targets() {
    assert_eq!(
        exact_bound_unit_step_target(&[0.5], &[-1.0], &[1.0], BoundActivity::Exact),
        None
    );
    assert_eq!(
        exact_bound_unit_step_target(&[1.0], &[-1.0], &[1.0], BoundActivity::Tolerant),
        None
    );
    assert_eq!(
        exact_bound_unit_step_target(&[1.0], &[-1.0], &[1.0], BoundActivity::Exact),
        Some(vec![1.0])
    );
    assert_eq!(
        exact_bound_unit_step_target(&[1.0; 10], &[-1.0; 10], &[1.0; 10], BoundActivity::Exact),
        None
    );
}

#[test]
fn line_search_reports_interpolated_trial_after_rejected_full_step() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[10.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[f64::INFINITY],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: true,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("line search succeeds")
    .expect("descent step");

    assert_eq!(step.line_search_trials, 1);
    assert_eq!(counts.function, 2);
    assert_eq!(counts.gradient, 2);
    assert!((step.alpha - 0.1998).abs() <= 1e-15);
    assert!(step.value < 1e-4, "{step:?}");
    assert!((step.step_norm - step.x[0].abs()).abs() <= 1e-15);
    assert!((step.curvature_ratio - 1e-3).abs() <= 1e-15, "{step:?}");
    assert!(step.wolfe_curvature_satisfied);
}

#[test]
fn line_search_diagnoses_armijo_step_that_is_too_short_for_wolfe_curvature() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[0.1],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[f64::INFINITY],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("line search succeeds")
    .expect("descent step");

    assert_eq!(step.line_search_trials, 0);
    assert_eq!(counts.function, 1);
    assert_eq!(counts.gradient, 1);
    assert_vec_close(&step.x, &[0.1], 1e-15);
    assert!((step.alpha - 1.0).abs() <= 1e-15);
    assert!((step.curvature_ratio - 0.95).abs() <= 1e-15, "{step:?}");
    assert!(!step.wolfe_curvature_satisfied);
}

#[test]
fn more_thuente_accepts_r_max_step_warning_case() {
    let mut problem = LinearDescent;
    let mut counts = OptimCounts::default();
    let step = more_thuente_line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 0.0,
            gradient: &[-1.0],
            direction: &[1.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[10.0],
            max_step_cap: Some(1.0),
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("More-Thuente should not error")
    .expect("R accepts STPMAX warning as a new iterate");

    assert_eq!(counts.function, 1);
    assert_eq!(counts.gradient, 1);
    assert_eq!(step.alpha, 1.0);
    assert_eq!(step.max_alpha, 1.0);
    assert!(!step.wolfe_curvature_satisfied);
    assert_vec_close(&step.x, &[1.0], 1e-15);
}

#[test]
fn strong_wolfe_line_search_expands_beyond_unit_alpha_when_direction_is_short() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = strong_wolfe_line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[0.1],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[1.0],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("strong-Wolfe line search should not error")
    .expect("strong-Wolfe line search should expand to satisfy curvature");

    assert_eq!(step.line_search_trials, 1);
    assert_eq!(counts.function, 2);
    assert_eq!(counts.gradient, 2);
    assert!((step.alpha - 2.0).abs() <= 1e-15, "{step:?}");
    assert!((step.max_alpha - 10.0).abs() <= 1e-15, "{step:?}");
    assert_vec_close(&step.x, &[0.2], 1e-15);
    assert!(step.wolfe_curvature_satisfied);
}

#[test]
fn strong_wolfe_line_search_zooms_after_overlarge_unit_step() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = strong_wolfe_line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[10.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[10.0],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("strong-Wolfe line search should not error")
    .expect("strong-Wolfe line search should zoom into the acceptable interval");

    assert_eq!(step.line_search_trials, 2);
    assert_eq!(counts.function, 3);
    assert_eq!(counts.gradient, 3);
    assert!((step.alpha - 0.25).abs() <= 1e-15, "{step:?}");
    assert_vec_close(&step.x, &[2.5], 1e-15);
    assert!(step.wolfe_curvature_satisfied);
}

#[test]
fn strong_wolfe_line_search_accepts_armijo_step_at_feasible_cap() {
    let mut problem = OneDimensionalQuadratic;
    let mut counts = OptimCounts::default();
    let step = strong_wolfe_line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[1.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[0.1],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("strong-Wolfe line search should not error")
    .expect("bounded line search should accept sufficient decrease at the cap");

    assert_eq!(step.line_search_trials, 0);
    assert_eq!(counts.function, 1);
    assert_eq!(counts.gradient, 1);
    assert!((step.alpha - 0.1).abs() <= 1e-15, "{step:?}");
    assert!((step.max_alpha - 0.1).abs() <= 1e-15, "{step:?}");
    assert_vec_close(&step.x, &[0.1], 1e-15);
    assert!((step.curvature_ratio - 0.95).abs() <= 1e-15, "{step:?}");
    assert!(!step.wolfe_curvature_satisfied);
}

#[test]
fn strong_wolfe_line_search_propagates_evaluation_errors() {
    let mut problem = AlwaysNonFiniteObjective;
    let mut counts = OptimCounts::default();
    let error = strong_wolfe_line_search(
        &mut problem,
        LineSearchRequest {
            x: &[0.0],
            value: 4.0,
            gradient: &[-4.0],
            direction: &[1.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[f64::INFINITY],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect_err("line search should return the objective error");

    assert!(matches!(
        error,
        OptimError::NonFiniteObjective { value } if value.is_nan()
    ));
    assert_eq!(counts.function, 0);
    assert_eq!(counts.gradient, 0);
}

#[test]
fn strong_wolfe_mode_records_rosenbrock_probe_drift() {
    let armijo = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::ProjectedLbfgs,
        5,
        1e7,
    );
    let strong_wolfe = run_rosenbrock_backend(
        LineSearchMode::StrongWolfe,
        DirectionMode::ProjectedLbfgs,
        5,
        1e7,
    );
    let strong_wolfe_loose = run_rosenbrock_backend(
        LineSearchMode::StrongWolfe,
        DirectionMode::ProjectedLbfgs,
        5,
        1e12,
    );
    let strong_wolfe_lmm_one = run_rosenbrock_backend(
        LineSearchMode::StrongWolfe,
        DirectionMode::ProjectedLbfgs,
        1,
        1e7,
    );

    assert_eq!(armijo.convergence, 0, "{armijo:?}");
    assert_eq!(armijo.counts.function, 48, "{armijo:?}");
    assert_eq!(armijo.counts.gradient, 48, "{armijo:?}");

    assert_eq!(strong_wolfe.convergence, 0, "{strong_wolfe:?}");
    assert_eq!(strong_wolfe.counts.function, 55, "{strong_wolfe:?}");
    assert_eq!(strong_wolfe.counts.gradient, 55, "{strong_wolfe:?}");

    assert_eq!(strong_wolfe_loose.convergence, 0, "{strong_wolfe_loose:?}");
    assert_eq!(
        strong_wolfe_loose.counts.function, 50,
        "{strong_wolfe_loose:?}"
    );
    assert_eq!(
        strong_wolfe_loose.counts.gradient, 50,
        "{strong_wolfe_loose:?}"
    );

    assert_eq!(
        strong_wolfe_lmm_one.convergence, 0,
        "{strong_wolfe_lmm_one:?}"
    );
    assert_eq!(
        strong_wolfe_lmm_one.counts.function, 92,
        "{strong_wolfe_lmm_one:?}"
    );
    assert_eq!(
        strong_wolfe_lmm_one.counts.gradient, 92,
        "{strong_wolfe_lmm_one:?}"
    );
}

#[test]
fn cauchy_subspace_direction_mode_records_rosenbrock_probe_drift() {
    let cauchy_armijo = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchySubspace,
        5,
        1e7,
    );
    let cauchy_capped_first_step = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchySubspaceCappedFirstStep,
        5,
        1e7,
    );
    let cauchy_first_then_projected = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchyFirstThenProjectedCapped,
        5,
        1e7,
    );
    let cauchy_strong_wolfe = run_rosenbrock_backend(
        LineSearchMode::StrongWolfe,
        DirectionMode::CauchySubspace,
        5,
        1e7,
    );
    let cauchy_more_thuente = run_rosenbrock_backend(
        LineSearchMode::MoreThuente,
        DirectionMode::CauchySubspace,
        5,
        1e7,
    );
    let cauchy_loose = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchySubspace,
        5,
        1e12,
    );
    let cauchy_lmm_one = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchySubspace,
        1,
        1e7,
    );
    let cauchy_more_thuente_loose = run_rosenbrock_backend(
        LineSearchMode::MoreThuente,
        DirectionMode::CauchySubspace,
        5,
        1e12,
    );
    let cauchy_more_thuente_lmm_one = run_rosenbrock_backend(
        LineSearchMode::MoreThuente,
        DirectionMode::CauchySubspace,
        1,
        1e7,
    );
    let cauchy_first_then_projected_loose = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchyFirstThenProjectedCapped,
        5,
        1e12,
    );
    let cauchy_first_then_projected_lmm_one = run_rosenbrock_backend(
        LineSearchMode::BacktrackingArmijo,
        DirectionMode::CauchyFirstThenProjectedCapped,
        1,
        1e7,
    );
    assert_eq!(cauchy_armijo.convergence, 0, "{cauchy_armijo:?}");
    assert_eq!(cauchy_armijo.counts.function, 47, "{cauchy_armijo:?}");
    assert_eq!(
        cauchy_armijo.counts.function, cauchy_armijo.counts.gradient,
        "{cauchy_armijo:?}"
    );
    assert_eq!(
        cauchy_capped_first_step.convergence, 0,
        "{cauchy_capped_first_step:?}"
    );
    assert_eq!(
        cauchy_capped_first_step.counts.function, 53,
        "{cauchy_capped_first_step:?}"
    );
    assert_eq!(
        cauchy_capped_first_step.counts.function, cauchy_capped_first_step.counts.gradient,
        "{cauchy_capped_first_step:?}"
    );
    assert_eq!(
        cauchy_first_then_projected.convergence, 0,
        "{cauchy_first_then_projected:?}"
    );
    assert_eq!(
        cauchy_first_then_projected.counts.function, 53,
        "{cauchy_first_then_projected:?}"
    );
    assert_eq!(
        cauchy_first_then_projected.counts.function, cauchy_first_then_projected.counts.gradient,
        "{cauchy_first_then_projected:?}"
    );
    assert_eq!(
        cauchy_strong_wolfe.convergence, 0,
        "{cauchy_strong_wolfe:?}"
    );
    assert_eq!(
        cauchy_strong_wolfe.counts.function, 56,
        "{cauchy_strong_wolfe:?}"
    );
    assert_eq!(
        cauchy_strong_wolfe.counts.function, cauchy_strong_wolfe.counts.gradient,
        "{cauchy_strong_wolfe:?}"
    );
    assert_eq!(
        cauchy_more_thuente.convergence, 0,
        "{cauchy_more_thuente:?}"
    );
    assert_eq!(
        cauchy_more_thuente.counts.function, 48,
        "{cauchy_more_thuente:?}"
    );
    assert_eq!(
        cauchy_more_thuente.counts.function, cauchy_more_thuente.counts.gradient,
        "{cauchy_more_thuente:?}"
    );
    assert_eq!(cauchy_loose.convergence, 0, "{cauchy_loose:?}");
    assert_eq!(cauchy_loose.counts.function, 43, "{cauchy_loose:?}");
    assert_eq!(
        cauchy_loose.counts.function, cauchy_loose.counts.gradient,
        "{cauchy_loose:?}"
    );
    assert_eq!(cauchy_lmm_one.convergence, 0, "{cauchy_lmm_one:?}");
    assert_eq!(cauchy_lmm_one.counts.function, 81, "{cauchy_lmm_one:?}");
    assert_eq!(
        cauchy_lmm_one.counts.function, cauchy_lmm_one.counts.gradient,
        "{cauchy_lmm_one:?}"
    );
    assert_eq!(
        cauchy_more_thuente_loose.convergence, 0,
        "{cauchy_more_thuente_loose:?}"
    );
    assert_eq!(
        cauchy_more_thuente_loose.counts.function, 45,
        "{cauchy_more_thuente_loose:?}"
    );
    assert_eq!(
        cauchy_more_thuente_loose.counts.function, cauchy_more_thuente_loose.counts.gradient,
        "{cauchy_more_thuente_loose:?}"
    );
    assert_eq!(
        cauchy_more_thuente_lmm_one.convergence, 0,
        "{cauchy_more_thuente_lmm_one:?}"
    );
    assert_eq!(
        cauchy_more_thuente_lmm_one.counts.function, 69,
        "{cauchy_more_thuente_lmm_one:?}"
    );
    assert_eq!(
        cauchy_more_thuente_lmm_one.counts.function, cauchy_more_thuente_lmm_one.counts.gradient,
        "{cauchy_more_thuente_lmm_one:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_loose.convergence, 0,
        "{cauchy_first_then_projected_loose:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_loose.counts.function, 48,
        "{cauchy_first_then_projected_loose:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_loose.counts.function,
        cauchy_first_then_projected_loose.counts.gradient,
        "{cauchy_first_then_projected_loose:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_lmm_one.convergence, 0,
        "{cauchy_first_then_projected_lmm_one:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_lmm_one.counts.function, 69,
        "{cauchy_first_then_projected_lmm_one:?}"
    );
    assert_eq!(
        cauchy_first_then_projected_lmm_one.counts.function,
        cauchy_first_then_projected_lmm_one.counts.gradient,
        "{cauchy_first_then_projected_lmm_one:?}"
    );
}

#[test]
fn more_thuente_first_step_then_projected_records_rosenbrock_probe_drift() {
    let r_shaped_first = run_rosenbrock_backend(
        LineSearchMode::MoreThuenteFirstThenBacktracking,
        DirectionMode::CauchyFirstThenProjected,
        5,
        1e7,
    );
    let r_shaped_first_loose = run_rosenbrock_backend(
        LineSearchMode::MoreThuenteFirstThenBacktracking,
        DirectionMode::CauchyFirstThenProjected,
        5,
        1e12,
    );
    let r_shaped_first_lmm_one = run_rosenbrock_backend(
        LineSearchMode::MoreThuenteFirstThenBacktracking,
        DirectionMode::CauchyFirstThenProjected,
        1,
        1e7,
    );

    assert_eq!(r_shaped_first.convergence, 0, "{r_shaped_first:?}");
    assert_eq!(r_shaped_first.counts.function, 46, "{r_shaped_first:?}");
    assert_eq!(
        r_shaped_first.counts.function, r_shaped_first.counts.gradient,
        "{r_shaped_first:?}"
    );
    assert_eq!(
        r_shaped_first_loose.convergence, 0,
        "{r_shaped_first_loose:?}"
    );
    assert_eq!(
        r_shaped_first_loose.counts.function, 43,
        "{r_shaped_first_loose:?}"
    );
    assert_eq!(
        r_shaped_first_loose.counts.function, r_shaped_first_loose.counts.gradient,
        "{r_shaped_first_loose:?}"
    );
    assert_eq!(
        r_shaped_first_lmm_one.convergence, 0,
        "{r_shaped_first_lmm_one:?}"
    );
    assert_eq!(
        r_shaped_first_lmm_one.counts.function, 62,
        "{r_shaped_first_lmm_one:?}"
    );
    assert_eq!(
        r_shaped_first_lmm_one.counts.function, r_shaped_first_lmm_one.counts.gradient,
        "{r_shaped_first_lmm_one:?}"
    );
}

#[test]
fn more_thuente_first_step_full_probe_matches_r_trace_prefix() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let mut backend = NativeBackend;
    let mut problem = RecordingRosenbrockProblem::default();
    let result = backend
        .minimize_with_modes(
            &mut problem,
            &[-1.2, 1.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            BackendControl {
                maxit: 100,
                factr: 1e7,
                pgtol: 0.0,
                lmm: 5,
                trace: 0,
                report: 10,
                has_user_gradient: true,
            },
            BackendModes {
                line_search: LineSearchMode::MoreThuenteFirstThenBacktracking,
                direction: DirectionMode::CauchyFirstThenProjected,
            },
        )
        .expect("Rosenbrock backend optimization");

    assert_eq!(result.convergence, 0, "{result:?}");
    assert_eq!(result.counts.function, 46, "{result:?}");
    assert_eq!(result.counts.function, result.counts.gradient, "{result:?}");
    assert!(
        problem.points.len() >= 8,
        "expected at least the first accepted R-shaped step: {:?}",
        problem.points
    );

    for (actual, expected) in problem
        .points
        .iter()
        .zip(fixture.first_points.iter())
        .take(7)
    {
        assert_vec_close(actual, expected, 1e-12);
    }
    assert_vec_close(
        &problem.points[7],
        &[-0.020278688163727798, -0.9287651476463392],
        1e-12,
    );
    assert_vec_not_close(&problem.points[7], &fixture.first_points[7], 1e-3);
}

#[test]
fn cauchy_more_thuente_full_probe_matches_r_trace_prefix_and_counts() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let mut backend = NativeBackend;
    let mut problem = RecordingRosenbrockProblem::default();
    let result = backend
        .minimize_with_modes(
            &mut problem,
            &[-1.2, 1.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            BackendControl {
                maxit: 100,
                factr: 1e7,
                pgtol: 0.0,
                lmm: 5,
                trace: 0,
                report: 10,
                has_user_gradient: true,
            },
            BackendModes {
                line_search: LineSearchMode::MoreThuente,
                direction: DirectionMode::CauchySubspace,
            },
        )
        .expect("Rosenbrock backend optimization");

    assert_eq!(result.convergence, 0, "{result:?}");
    assert_eq!(result.counts.function, 48, "{result:?}");
    assert_eq!(result.counts.function, result.counts.gradient, "{result:?}");
    assert!(
        problem.points.len() >= 8,
        "expected at least the first third-iteration trial: {:?}",
        problem.points
    );
    for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-10);
    }
}

#[test]
fn fourth_iteration_more_thuente_extrapolates_to_r_trace_step() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let mut problem = RosenbrockProblem;
    let x0 = fixture.first_points[0].clone();
    let x1 = fixture.first_points[3].clone();
    let x2 = fixture.first_points[4].clone();
    let x3 = fixture.first_points[5].clone();
    let g0 = problem.gradient(&x0).expect("Rosenbrock gradient");
    let g1 = problem.gradient(&x1).expect("Rosenbrock gradient");
    let g2 = problem.gradient(&x2).expect("Rosenbrock gradient");
    let g3 = problem.gradient(&x3).expect("Rosenbrock gradient");
    let mut history = Vec::new();

    update_history(
        &mut history,
        5,
        difference(&x1, &x0),
        difference(&g1, &g0),
        &g0,
    );
    update_history(
        &mut history,
        5,
        difference(&x2, &x1),
        difference(&g2, &g1),
        &g1,
    );
    update_history(
        &mut history,
        5,
        difference(&x3, &x2),
        difference(&g3, &g2),
        &g2,
    );

    let direction = cauchy_subspace_direction(
        &x3,
        &g3,
        &lower,
        &upper,
        &history,
        BoundActivity::Tolerant,
        MAIN_PATH_MIN_STEP,
    )
    .expect("fourth iteration model should produce a step")
    .direction;
    let value = problem.value(&x3).expect("Rosenbrock value");
    let mut recording = RecordingRosenbrockProblem::default();
    let mut counts = OptimCounts::default();
    let step = line_search_with_mode(
        &mut recording,
        LineSearchRequest {
            x: &x3,
            value,
            gradient: &g3,
            direction: &direction,
            unit_step_target: None,
            lower: &lower,
            upper: &upper,
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
        LineSearchMode::MoreThuente,
        4,
    )
    .expect("line search should not error")
    .expect("line search should accept the R trace step");
    assert_eq!(history.len(), 3);
    assert_eq!(counts.function, 5);
    assert_eq!(counts.gradient, 5);
    assert_eq!(step.line_search_trials, 4);
    for (actual, expected) in recording
        .points
        .iter()
        .zip(fixture.first_points.iter().skip(6))
    {
        assert_vec_close(actual, expected, 1e-10);
    }
    assert_vec_close(&step.x, &fixture.first_points[10], 1e-10);
}

#[test]
fn capped_cauchy_subspace_first_step_matches_r_trace_shape() {
    let x = vec![-1.2, 1.0];
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let history = Vec::new();
    let modes = BackendModes {
        line_search: LineSearchMode::BacktrackingArmijo,
        direction: DirectionMode::CauchySubspaceCappedFirstStep,
    };
    let mut problem = RosenbrockProblem;
    let value = problem.value(&x).expect("Rosenbrock value");
    let gradient = problem.gradient(&x).expect("Rosenbrock gradient");
    let direction = direction_with_mode(
        &x,
        &gradient,
        &lower,
        &upper,
        &history,
        DirectionSettings {
            mode: modes.direction,
            bound_activity: BoundActivity::Tolerant,
            min_step: MIN_STEP,
        },
    )
    .direction;
    let mut counts = OptimCounts::default();

    let step = line_search(
        &mut problem,
        LineSearchRequest {
            x: &x,
            value,
            gradient: &gradient,
            direction: &direction,
            unit_step_target: None,
            lower: &lower,
            upper: &upper,
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: initial_step_cap_for_modes(modes, history.is_empty()),
            allow_quadratic_interpolation: true,
            allow_retry_warning_accept: false,
            min_step: MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
    )
    .expect("line search should not error")
    .expect("capped Cauchy step should be accepted");

    assert_eq!(step.line_search_trials, 1);
    assert_eq!(counts.function, 2);
    assert_eq!(counts.gradient, 2);
    assert!(
        (step.alpha - 0.032438841511832885).abs() <= 1e-14,
        "{step:?}"
    );
    assert!((step.x[0] + 0.9988791826266361).abs() <= 1e-14, "{step:?}");
    assert!((step.x[1] - 1.1297553660473316).abs() <= 1e-14, "{step:?}");
    assert!(
        (step.step_norm - 0.2393450191648178).abs() <= 1e-14,
        "{step:?}"
    );
}

#[test]
fn more_thuente_cauchy_first_step_matches_r_trace_prefix() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let x = vec![-1.2, 1.0];
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let history = Vec::new();
    let mut problem = RecordingRosenbrockProblem::default();
    let mut counts = OptimCounts::default();
    let Evaluation { value, gradient } =
        evaluate(&mut problem, &x, &mut counts).expect("initial Rosenbrock evaluation");
    let direction = direction_with_mode(
        &x,
        &gradient,
        &lower,
        &upper,
        &history,
        DirectionSettings {
            mode: DirectionMode::CauchySubspace,
            bound_activity: BoundActivity::Tolerant,
            min_step: MAIN_PATH_MIN_STEP,
        },
    )
    .direction;

    let step = line_search_with_mode(
        &mut problem,
        LineSearchRequest {
            x: &x,
            value,
            gradient: &gradient,
            direction: &direction,
            unit_step_target: None,
            lower: &lower,
            upper: &upper,
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            initial_step_cap: None,
            allow_quadratic_interpolation: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            quadratic_interpolation_damping: INTERPOLATION_DAMPING,
        },
        &mut counts,
        LineSearchMode::MoreThuente,
        1,
    )
    .expect("More-Thuente probe should not error")
    .expect("More-Thuente probe should accept the first R-like step");

    assert_eq!(step.line_search_trials, 2);
    assert_eq!(counts.function, 4);
    assert_eq!(counts.gradient, 4);
    assert!((step.alpha - 0.03238226005834323).abs() <= 1e-14);
    assert_vec_close(&step.x, &fixture.first_points[3], 1e-14);
    for (actual, expected) in problem
        .points
        .iter()
        .zip(fixture.first_points.iter())
        .take(4)
    {
        assert_vec_close(actual, expected, 1e-12);
    }
}

#[test]
fn rosenbrock_default_evaluation_trace_matches_r_prefix() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    assert_eq!(fixture.fixture, "rosenbrock_default_trace");
    assert_eq!(fixture.source_fixture, "rosenbrock");
    assert_eq!(fixture.trace_kind, "function_evaluation_points");

    let mut backend = NativeBackend;
    let mut problem = RecordingRosenbrockProblem::default();
    let result = backend
        .minimize_with_modes(
            &mut problem,
            &[-1.2, 1.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            BackendControl {
                maxit: 100,
                factr: 1e7,
                pgtol: 0.0,
                lmm: 5,
                trace: 0,
                report: 10,
                has_user_gradient: true,
            },
            BackendModes::default(),
        )
        .expect("Rosenbrock backend optimization");

    assert_eq!(result.counts.function, 48, "{result:?}");
    assert_eq!(result.counts.gradient, 48, "{result:?}");
    assert!(
        problem.points.len() >= fixture.first_points.len(),
        "{:?}",
        problem.points
    );

    for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-10);
    }
}

#[test]
fn strong_wolfe_scaffold_expands_past_too_short_armijo_step() {
    let result = strong_wolfe_search(0.1, 4.0, 4.0, -4.0, |alpha| Ok(one_dimensional_phi(alpha)))
        .expect("strong Wolfe search should not error")
        .expect("strong Wolfe search should expand to a curvature-satisfying step");

    assert!(result.point.alpha > 0.1, "{result:?}");
    assert!(result.point.value <= armijo_value(4.0, -4.0, result.point.alpha));
    assert!(result.point.derivative.abs() <= -WOLFE_CURVATURE * -4.0);
    assert_eq!(result.point.alpha, 0.2);
}

#[test]
fn strong_wolfe_scaffold_zooms_from_overlarge_step() {
    let result = strong_wolfe_search(10.0, 10.0, 4.0, -4.0, |alpha| {
        Ok(one_dimensional_phi(alpha))
    })
    .expect("strong Wolfe search should not error")
    .expect("strong Wolfe search should zoom into the acceptable interval");

    assert!(result.point.alpha > 0.0, "{result:?}");
    assert!(result.point.alpha < 10.0, "{result:?}");
    assert!(result.point.value <= armijo_value(4.0, -4.0, result.point.alpha));
    assert!(result.point.derivative.abs() <= -WOLFE_CURVATURE * -4.0);
    assert_eq!(result.point.alpha, 2.5);
}

fn dense_two_by_two_product(matrix: [[f64; 2]; 2]) -> impl FnMut(&[f64]) -> Vec<f64> {
    move |vector| {
        vec![
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
        ]
    }
}

fn one_dimensional_phi(alpha: f64) -> LineSearchPoint1D {
    LineSearchPoint1D {
        alpha,
        value: (alpha - 2.0).powi(2),
        derivative: 2.0 * (alpha - 2.0),
    }
}

fn run_rosenbrock_backend(
    line_search_mode: LineSearchMode,
    direction_mode: DirectionMode,
    lmm: usize,
    factr: f64,
) -> BackendResult {
    let mut backend = NativeBackend;
    let mut problem = RosenbrockProblem;
    backend
        .minimize_with_modes(
            &mut problem,
            &[-1.2, 1.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            BackendControl {
                maxit: 100,
                factr,
                pgtol: 0.0,
                lmm,
                trace: 0,
                report: 10,
                has_user_gradient: true,
            },
            BackendModes {
                line_search: line_search_mode,
                direction: direction_mode,
            },
        )
        .expect("Rosenbrock backend optimization")
}

struct RosenbrockProblem;

impl BackendProblem for RosenbrockProblem {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ])
    }
}

#[derive(Default)]
struct RecordingRosenbrockProblem {
    points: Vec<Vec<f64>>,
}

impl BackendProblem for RecordingRosenbrockProblem {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ])
    }

    fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        self.points.push(x.to_vec());
        Ok((self.value(x)?, self.gradient(x)?))
    }
}

struct OneDimensionalQuadratic;

impl BackendProblem for OneDimensionalQuadratic {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok((x[0] - 2.0).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![2.0 * (x[0] - 2.0)])
    }
}

struct LinearDescent;

impl BackendProblem for LinearDescent {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(-x[0])
    }

    fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![-1.0])
    }
}

struct AlwaysNonFiniteObjective;

impl BackendProblem for AlwaysNonFiniteObjective {
    fn value(&mut self, _x: &[f64]) -> Result<f64, OptimError> {
        Err(OptimError::NonFiniteObjective { value: f64::NAN })
    }

    fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![0.0])
    }
}
