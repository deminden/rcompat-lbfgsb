use super::*;

pub(super) fn run_fixture(fixture: &Fixture) -> OptimResult {
    let control = control_from_fixture(&fixture.control);
    let bounds = Bounds::new(fixture.lower.clone(), fixture.upper.clone()).unwrap();

    match fixture.fixture.as_str() {
        "active_bounds" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "active_upper_bound" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 2.0).powi(2),
            control,
        )
        .unwrap(),
        "fixed_parameter" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 10.0).powi(2),
            |p| vec![2.0 * (p[0] - 10.0)],
            control,
        )
        .unwrap(),
        "fixed_free_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 10.0).powi(2) + (p[1] - 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 10.0), 2.0 * (p[1] - 2.0)],
            control,
        )
        .unwrap(),
        "all_fixed_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 10.0).powi(2) + (p[1] - 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 10.0), 2.0 * (p[1] - 2.0)],
            control,
        )
        .unwrap(),
        "finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 1.0).powi(2),
            control,
        )
        .unwrap(),
        "fnscale" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| -(p[0] - 3.0).powi(2),
            control,
        )
        .unwrap(),
        "fnscale_parscale_gradient_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| -((p[0] - 3.0).powi(2) + 2.0 * (p[1] + 1.0).powi(2)),
            |p| vec![-2.0 * (p[0] - 3.0), -4.0 * (p[1] + 1.0)],
            control,
        )
        .unwrap(),
        "fnscale_parscale_finite_difference_2d" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| -((p[0] - 3.0).powi(2) + 2.0 * (p[1] + 1.0).powi(2)),
            control,
        )
        .unwrap(),
        "negative_fnscale_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| -(p[0] - 3.0).powi(2),
            |p| vec![-2.0 * (p[0] - 3.0)],
            control,
        )
        .unwrap(),
        "initial_projected_gradient" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "initial_projected_gradient_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 2.0).powi(2) + (p[1] + 1.0).powi(2),
            |p| vec![2.0 * (p[0] + 2.0), 2.0 * (p[1] + 1.0)],
            control,
        )
        .unwrap(),
        "initial_outside_bounds" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| p[0] * p[0],
            control,
        )
        .unwrap(),
        "initial_pos_inf_projected" | "initial_neg_inf_projected" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| p[0] * p[0],
            |p| vec![2.0 * p[0]],
            control,
        )
        .unwrap(),
        "mixed_bounds_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)],
            control,
        )
        .unwrap(),
        "mixed_bounds_finite_difference_2d"
        | "lower_bounded_finite_difference_2d"
        | "upper_bounded_finite_difference_2d" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "all_unbounded_2d_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + 2.0 * (p[1] + 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 4.0 * (p[1] + 2.0)],
            control,
        )
        .unwrap(),
        "ndeps_vector" | "all_unbounded_finite_difference_2d" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "near_upper_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 0.25).powi(2),
            control,
        )
        .unwrap(),
        "near_lower_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 0.25).powi(2),
            control,
        )
        .unwrap(),
        "near_box_finite_difference_2d" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 0.25).powi(2) + 2.0 * (p[1] + 0.25).powi(2),
            control,
        )
        .unwrap(),
        "parscale" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2),
            |p| vec![2.0 * (p[0] - 4.0)],
            control,
        )
        .unwrap(),
        "parscale_bounds_gradient_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 6.0).powi(2) + 2.0 * (p[1] - 0.75).powi(2),
            |p| vec![2.0 * (p[0] - 6.0), 4.0 * (p[1] - 0.75)],
            control,
        )
        .unwrap(),
        "parscale_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2),
            control,
        )
        .unwrap(),
        "two_dim_parscale_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + 3.0 * (p[1] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "pgtol_initial" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 2.0)],
            control,
        )
        .unwrap(),
        "pgtol_near_upper_uses_bound_distance" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| -10.0 * p[0],
            |_| vec![-10.0],
            control,
        )
        .unwrap(),
        "pgtol_finite_difference_2d"
        | "pgtol_initial_finite_difference_2d"
        | "maxit_zero_finite_difference_2d"
        | "maxit_one_finite_difference_2d" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "lower_bound_finite_difference_trace_probe" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| p[0] + 2.0 * p[1],
            control,
        )
        .unwrap(),
        "line_search_refresh_sin_quad" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| {
                let coupled = 0.3 * (7.0 * p.iter().sum::<f64>()).cos();
                p.iter()
                    .map(|x| 0.02 * x.powi(2) + (15.0 * x).sin() + coupled)
                    .sum::<f64>()
            },
            control,
        )
        .unwrap(),
        "pgtol_after_step_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)],
            control,
        )
        .unwrap(),
        "factr_zero_2d_quadratic"
        | "factr_loose_2d_quadratic"
        | "maxit_one_2d_quadratic"
        | "maxit_zero_2d_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)],
            control,
        )
        .unwrap(),
        "first_constrained_linear_2d" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| -p[0] - 0.5 * p[1],
            |_| vec![-1.0, -0.5],
            control,
        )
        .unwrap(),
        "quadratic" | "maxit" | "maxit_zero" | "factr_zero_quadratic" => {
            optim_lbfgsb_with_gradient(
                fixture.initial_par.clone(),
                bounds,
                |p| (p[0] - 2.0).powi(2),
                |p| vec![2.0 * (p[0] - 2.0)],
                control,
            )
            .unwrap()
        }
        "zero_dim" => {
            optim_lbfgsb(fixture.initial_par.clone(), bounds, |_| 123.0, control).unwrap()
        }
        "zero_dim_with_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |_| 123.0,
            |_| panic!("zero-dimensional gradient should not be called"),
            control,
        )
        .unwrap(),
        "two_dim_parscale_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2) + (p[1] + 1.0).powi(2),
            |p| vec![2.0 * (p[0] - 4.0), 2.0 * (p[1] + 1.0)],
            control,
        )
        .unwrap(),
        "three_dim_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + 3.0 * (p[1] + 2.0).powi(2) + 0.5 * (p[2] - 0.25).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 6.0 * (p[1] + 2.0), p[2] - 0.25],
            control,
        )
        .unwrap(),
        "three_dim_box_active" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 5.0).powi(2) + (p[1] + 2.0).powi(2) + (p[2] - 0.25).powi(2),
            |p| vec![2.0 * (p[0] - 5.0), 2.0 * (p[1] + 2.0), 2.0 * (p[2] - 0.25)],
            control,
        )
        .unwrap(),
        "unbounded_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.25).powi(2),
            |p| vec![2.0 * (p[0] - 1.25)],
            control,
        )
        .unwrap(),
        "rosenbrock_finite_difference_box" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| 100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2),
            control,
        )
        .unwrap(),
        "rosenbrock"
        | "factr_loose_rosenbrock"
        | "lmm_one_rosenbrock"
        | "lmm_two_rosenbrock"
        | "lmm_ten_rosenbrock" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| 100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2),
            |p| {
                vec![
                    -400.0 * p[0] * (p[1] - p[0] * p[0]) - 2.0 * (1.0 - p[0]),
                    200.0 * (p[1] - p[0] * p[0]),
                ]
            },
            control,
        )
        .unwrap(),
        other => panic!("unknown fixture {other}"),
    }
}
