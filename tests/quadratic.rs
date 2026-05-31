#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, optim_lbfgsb_with_gradient, Bounds, OptimControl};

#[test]
fn no_gradient_quadratic_converges() {
    let result = optim_lbfgsb(
        vec![5.0, -3.0],
        Bounds::new(vec![-10.0, -10.0], vec![10.0, 10.0]).unwrap(),
        |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
        OptimControl::default_for_dimension(2),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 1.0).abs() < 1e-4);
    assert!((result.par[1] + 2.0).abs() < 1e-4);
}

#[test]
fn supplied_gradient_quadratic_converges() {
    let result = optim_lbfgsb_with_gradient(
        vec![5.0, -3.0],
        Bounds::new(vec![-10.0, -10.0], vec![10.0, 10.0]).unwrap(),
        |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
        |p| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)],
        OptimControl::default_for_dimension(2),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 1.0).abs() < 1e-8);
    assert!((result.par[1] + 2.0).abs() < 1e-8);
    assert!(result.gradient_count() > 0);
}

#[test]
fn factr_zero_keeps_r_style_projected_gradient_stop() {
    let mut control = OptimControl::default_for_dimension(1);
    control.factr = 0.0;
    control.pgtol = 0.0;

    let result = optim_lbfgsb_with_gradient(
        vec![10.0],
        Bounds::new(vec![f64::NEG_INFINITY], vec![f64::INFINITY]).unwrap(),
        |p| (p[0] - 2.0).powi(2),
        |p| vec![2.0 * (p[0] - 2.0)],
        control,
    )
    .unwrap();

    assert_eq!(result.par, vec![2.0]);
    assert_eq!(result.value, 0.0);
    assert_eq!(result.function_count(), 3);
    assert_eq!(result.gradient_count(), 3);
    assert_eq!(
        result.message,
        "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
    );
}
