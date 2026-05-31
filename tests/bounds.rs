#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, optim_lbfgsb_with_gradient, Bounds, OptimControl, OptimError};

#[test]
fn active_lower_bound_is_respected() {
    let result = optim_lbfgsb(
        vec![3.0],
        Bounds::new(vec![0.0], vec![10.0]).unwrap(),
        |p| (p[0] + 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 0.0).abs() < 1e-8);
}

#[test]
fn fixed_parameter_stays_fixed() {
    let result = optim_lbfgsb_with_gradient(
        vec![1.5],
        Bounds::new(vec![1.5], vec![1.5]).unwrap(),
        |p| (p[0] - 10.0).powi(2),
        |p| vec![2.0 * (p[0] - 10.0)],
        OptimControl::default_for_dimension(1),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert_eq!(result.par, vec![1.5]);
}

#[test]
fn fixed_parameter_without_gradient_is_reported() {
    let error = optim_lbfgsb(
        vec![1.5],
        Bounds::new(vec![1.5], vec![1.5]).unwrap(),
        |p| (p[0] - 10.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap_err();

    assert!(matches!(error, OptimError::InvalidBounds { .. }));
}

#[test]
fn invalid_bounds_are_reported() {
    let error = Bounds::new(vec![2.0], vec![1.0]).unwrap_err();
    assert!(matches!(error, OptimError::InvalidBounds { .. }));
}

#[test]
fn initial_parameter_outside_bounds_is_projected_like_r() {
    let result = optim_lbfgsb(
        vec![2.0],
        Bounds::new(vec![0.0], vec![1.0]).unwrap(),
        |p| p[0] * p[0],
        OptimControl::default_for_dimension(1),
    )
    .unwrap();

    assert!(
        result.par[0].abs() < 1e-12,
        "projected parameter should be numerically zero: {:?}",
        result.par
    );
    assert!(result.is_success(), "{}", result.message);
}

#[test]
fn non_projectable_infinite_initial_parameter_is_reported() {
    let error = optim_lbfgsb_with_gradient(
        vec![f64::INFINITY],
        Bounds::unbounded(1),
        |p| p[0] * p[0],
        |p| vec![2.0 * p[0]],
        OptimControl::default_for_dimension(1),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OptimError::NonFiniteInitialParameter { .. }
    ));
}
