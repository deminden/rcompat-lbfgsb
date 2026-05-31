#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};

#[test]
fn no_gradient_path_uses_finite_difference_counts() {
    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |p| (p[0] - 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!(result.gradient_count() > 0);
    assert_eq!(result.function_count(), result.gradient_count());
}

#[test]
fn ndeps_controls_scaled_finite_difference_path() {
    let mut control = OptimControl::default_for_dimension(1);
    control.ndeps = vec![1e-4];

    let result = optim_lbfgsb(
        vec![6.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |p| (p[0] + 1.0).powi(2),
        control,
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] + 1.0).abs() < 1e-4);
}
