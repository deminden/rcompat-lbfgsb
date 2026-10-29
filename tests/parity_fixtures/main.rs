#![allow(missing_docs)]
#[path = "../support/mod.rs"]
mod support;
use rcompat_lbfgsb::{Bounds, OptimControl, OptimError, OptimResult};
use serde::Deserialize;
use support::{
    optim_reference as optim_lbfgsb, optim_reference_with_gradient as optim_lbfgsb_with_gradient,
};
mod data;
use data::*;
mod objective;
use objective::*;

#[test]
fn fixtures_that_are_currently_at_float_noise_match_r() {
    let mut failures = Vec::new();
    for fixture_json in [
        include_str!("../../fixtures/r_optim_lbfgsb/active_bounds.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/active_upper_bound.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/all_fixed_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/all_unbounded_2d_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/all_unbounded_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/factr_zero_2d_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/factr_zero_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/factr_loose_2d_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/first_constrained_linear_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/fixed_free_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/fixed_parameter.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/finite_difference.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/fnscale_parscale_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/fnscale_parscale_gradient_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/fnscale.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/initial_projected_gradient_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/initial_projected_gradient.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/initial_outside_bounds.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/initial_pos_inf_projected.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/initial_neg_inf_projected.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/lower_bounded_finite_difference_2d.json"),
        include_str!(
            "../../fixtures/r_optim_lbfgsb/lower_bound_finite_difference_trace_probe.json"
        ),
        include_str!("../../fixtures/r_optim_lbfgsb/mixed_bounds_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/mixed_bounds_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/near_box_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/near_lower_finite_difference.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/near_upper_finite_difference.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/negative_fnscale_gradient.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/ndeps_vector.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/parscale.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/parscale_bounds_gradient_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/parscale_finite_difference.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/pgtol_after_step_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/pgtol_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/pgtol_initial_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/pgtol_initial.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/pgtol_near_upper_uses_bound_distance.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/rosenbrock_finite_difference_box.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/three_dim_box_active.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/three_dim_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/two_dim_parscale_finite_difference.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/unbounded_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/upper_bounded_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/zero_dim.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/zero_dim_with_gradient.json"),
    ] {
        let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
        let result = run_fixture(&fixture);
        let issues = fixture_failures(&fixture, &result, 1e-8, 1e-10);
        if !issues.is_empty() {
            failures.push(format!("{}: {}", fixture.fixture, issues.join("; ")));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "full trigonometric replay requires the recorded Apple Silicon callback math; mandatory in pinned CI"
)]
fn line_search_refresh_fixture_matches_reference() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/line_search_refresh_sin_quad.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
}

#[test]
fn line_search_refresh_has_portable_endpoint_accuracy() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/line_search_refresh_sin_quad.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    let comparison = support::compare_scientific(&result, &expected_result(&fixture), 1e-8, 1e-10);
    assert!(comparison.failures.is_empty(), "{:?}", comparison.failures);
    eprintln!(
        "trigonometric bookkeeping observations: {:?}",
        comparison.bookkeeping_differences
    );
}

#[test]
fn rosenbrock_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-7, 1e-12);
}

#[test]
fn maxit_fixture_matches_r() {
    for fixture_json in [
        include_str!("../../fixtures/r_optim_lbfgsb/maxit.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/maxit_zero.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/maxit_one_2d_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/maxit_zero_2d_quadratic.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/maxit_zero_finite_difference_2d.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/maxit_one_finite_difference_2d.json"),
    ] {
        let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
        let result = run_fixture(&fixture);
        assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
    }
}

#[test]
fn two_dim_parscale_gradient_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/two_dim_parscale_gradient.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
}

#[test]
fn loose_factr_rosenbrock_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/factr_loose_rosenbrock.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-6, 1e-8);
}

#[test]
fn rosenbrock_finite_difference_box_trace_matches_r_prefix() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_finite_difference_box.json"
    ))
    .unwrap();
    let trace: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/rosenbrock_finite_difference_box_trace.json"
    ))
    .unwrap();
    assert_eq!(trace.fixture, "rosenbrock_finite_difference_box_trace");
    assert_eq!(trace.source_fixture, fixture.fixture);
    assert_eq!(trace.trace_kind, "function_evaluation_points");

    let control = control_from_fixture(&fixture.control);
    let bounds = Bounds::new(fixture.lower.clone(), fixture.upper.clone()).unwrap();
    let mut calls = Vec::new();
    let result = optim_lbfgsb(
        fixture.initial_par.clone(),
        bounds,
        |p| {
            calls.push(p.to_vec());
            100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2)
        },
        control,
    )
    .unwrap();

    assert_eq!(result.counts.function, trace.result.counts.function);
    assert_eq!(result.counts.gradient, trace.result.counts.gradient);
    assert_eq!(result.convergence, trace.result.convergence);
    assert_eq!(result.message, trace.result.message);
    assert!(
        calls.len() >= trace.first_points.len(),
        "expected at least {} raw objective calls, got {}",
        trace.first_points.len(),
        calls.len()
    );
    for (actual, expected) in calls.iter().zip(trace.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-10);
    }
}

#[test]
fn lower_bound_finite_difference_trace_matches_r_prefix() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/lower_bound_finite_difference_trace_probe.json"
    ))
    .unwrap();
    let trace: TraceFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/lower_bound_finite_difference_trace.json"
    ))
    .unwrap();
    assert_eq!(trace.fixture, "lower_bound_finite_difference_trace");
    assert_eq!(trace.source_fixture, fixture.fixture);
    assert_eq!(trace.trace_kind, "function_evaluation_points");

    let control = control_from_fixture(&fixture.control);
    let bounds = Bounds::new(fixture.lower.clone(), fixture.upper.clone()).unwrap();
    let mut calls = Vec::new();
    let result = optim_lbfgsb(
        fixture.initial_par.clone(),
        bounds,
        |p| {
            calls.push(p.to_vec());
            p[0] + 2.0 * p[1]
        },
        control,
    )
    .unwrap();

    assert_eq!(result.counts.function, trace.result.counts.function);
    assert_eq!(result.counts.gradient, trace.result.counts.gradient);
    assert_eq!(result.convergence, trace.result.convergence);
    assert_eq!(result.message, trace.result.message);
    assert!(
        calls.len() >= trace.first_points.len(),
        "expected at least {} raw objective calls, got {}",
        trace.first_points.len(),
        calls.len()
    );
    for (actual, expected) in calls.iter().zip(trace.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-12);
    }
}

#[test]
fn lmm_rosenbrock_fixtures_match_r() {
    for fixture_json in [
        include_str!("../../fixtures/r_optim_lbfgsb/lmm_one_rosenbrock.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/lmm_two_rosenbrock.json"),
        include_str!("../../fixtures/r_optim_lbfgsb/lmm_ten_rosenbrock.json"),
    ] {
        let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
        let result = run_fixture(&fixture);
        assert_fixture_close(&fixture, &result, 1e-6, 1e-8);
    }
}

#[test]
fn fixed_parameter_without_gradient_matches_r_error_class() {
    let fixture: ErrorFixture = serde_json::from_str(include_str!(
        "../../fixtures/r_optim_lbfgsb/fixed_no_gradient_error.json"
    ))
    .unwrap();
    assert_eq!(fixture.fixture, "fixed_no_gradient_error");
    assert_eq!(
        fixture.error.message,
        "non-finite finite-difference value [1]"
    );

    let error = optim_lbfgsb(
        fixture.initial_par,
        Bounds::new(fixture.lower, fixture.upper).unwrap(),
        |p| (p[0] - 10.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap_err();

    assert!(matches!(error, OptimError::InvalidBounds { .. }));
}
