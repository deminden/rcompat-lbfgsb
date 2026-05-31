#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, optim_lbfgsb_with_gradient, Bounds, OptimControl};

#[test]
fn negative_fnscale_maximizes_user_objective() {
    let mut control = OptimControl::default_for_dimension(1);
    control.fnscale = -1.0;

    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |p| -(p[0] - 3.0).powi(2),
        control,
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 3.0).abs() < 1e-5);
    assert!(result.value.abs() < 1e-10);
}

#[test]
fn parscale_keeps_result_in_user_scale() {
    let mut control = OptimControl::default_for_dimension(1);
    control.parscale = vec![2.0];

    let result = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |p| (p[0] - 4.0).powi(2),
        |p| vec![2.0 * (p[0] - 4.0)],
        control,
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 4.0).abs() < 1e-8);
    assert!(result.value.abs() < 1e-12);
}

#[test]
fn invalid_fnscale_is_reported() {
    let mut control = OptimControl::default_for_dimension(1);
    control.fnscale = 0.0;

    let error =
        optim_lbfgsb(vec![0.0], Bounds::unbounded(1), |p| p[0] * p[0], control).unwrap_err();

    assert!(error.to_string().contains("fnscale"));
}
