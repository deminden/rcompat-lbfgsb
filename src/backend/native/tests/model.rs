use super::*;

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
