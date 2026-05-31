#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};

#[test]
fn public_api_optimizes_simple_quadratic() {
    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |p| (p[0] - 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap();

    assert!(result.is_success(), "{}", result.message);
    assert!((result.par[0] - 2.0).abs() < 1e-5);
    assert!(result.value.abs() < 1e-10);
    assert!(result.function_count() > 0);
}
