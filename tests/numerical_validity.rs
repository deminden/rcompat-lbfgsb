//! Public numerical validity and termination regressions.
use rcompat_lbfgsb::{optim_lbfgsb, optim_lbfgsb_with_gradient, Bounds, OptimControl};

#[test]
fn scaled_objective_overflow_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 1e-320;
    assert!(
        optim_lbfgsb_with_gradient(vec![0.0], Bounds::unbounded(1), |_| 1.0, |_| vec![0.0], c)
            .is_err()
    );
}

#[test]
fn scaled_parameter_overflow_never_reaches_callback() {
    let mut c = OptimControl::default_for_dimension(1);
    c.parscale[0] = 1e-320;
    let mut called = false;
    let result = optim_lbfgsb_with_gradient(
        vec![1.0],
        Bounds::new(vec![-2.0], vec![2.0]).unwrap(),
        |_| {
            called = true;
            1.0
        },
        |_| vec![0.0],
        c,
    );
    assert!(result.is_err());
    assert!(!called);
}

#[test]
fn scaled_gradient_overflow_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.parscale[0] = 10.0;
    assert!(optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-1.0], vec![1.0]).unwrap(),
        |x| 1e308 * x[0],
        |_| vec![1e308],
        c
    )
    .is_err());
}

#[test]
fn overflowing_finite_difference_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.ndeps[0] = 1e308;
    assert!(optim_lbfgsb(vec![0.0], Bounds::unbounded(1), |x| x[0], c).is_err());
}

#[test]
fn near_bound_stagnation_cannot_claim_stationarity() {
    let mut c = OptimControl::default_for_dimension(1);
    c.factr = 0.0;
    c.pgtol = 1e-15;
    let result = optim_lbfgsb_with_gradient(
        vec![5e-13],
        Bounds::new(vec![0.0], vec![1.0]).unwrap(),
        |x| 1e12 * x[0],
        |_| vec![1e12],
        c,
    )
    .unwrap();
    assert!(!result.is_success() || result.par[0] <= 1e-15, "{result:?}");
}

#[test]
fn small_gradient_above_positive_tolerance_cannot_claim_stationarity() {
    let mut c = OptimControl::default_for_dimension(1);
    c.factr = 0.0;
    c.pgtol = 1e-20;
    let result = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-1.0], vec![1.0]).unwrap(),
        |x| 1e-15 * x[0],
        |_| vec![1e-15],
        c,
    )
    .unwrap();
    assert!(!result.is_success() || result.par[0] == -1.0, "{result:?}");
}

#[test]
fn unbounded_linear_objective_has_no_stationary_point() {
    let mut c = OptimControl::default_for_dimension(1);
    c.factr = 0.0;
    let result = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::unbounded(1),
        |x| 1e-13 * x[0],
        |_| vec![1e-13],
        c,
    )
    .unwrap();
    assert!(!result.is_success(), "{result:?}");
}

#[test]
fn bounds_membership_rejects_nonfinite_parameters() {
    let b = Bounds::unbounded(1);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(b.contains(&[value]).is_err());
    }
}

#[test]
fn finite_bound_overflow_is_rejected_before_callback() {
    let mut c = OptimControl::default_for_dimension(1);
    c.parscale[0] = 1e-320;
    assert!(optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-2.0], vec![2.0]).unwrap(),
        |_| panic!("invalid scaled bounds reached callback"),
        |_| vec![0.0],
        c
    )
    .is_err());
}

#[test]
fn unrepresentable_stencil_is_rejected() {
    assert!(optim_lbfgsb(
        vec![1e20],
        Bounds::unbounded(1),
        |x| x[0],
        OptimControl::default_for_dimension(1)
    )
    .is_err());
}

#[test]
fn derivative_overflow_with_finite_stencil_is_rejected() {
    let mut c = OptimControl::default_for_dimension(1);
    c.ndeps[0] = 1.0;
    assert!(optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-1.0], vec![1.0]).unwrap(),
        |x| 1e308 * x[0],
        c
    )
    .is_err());
}

#[test]
fn trace_preserves_callback_sequence_and_result() {
    fn run(trace: usize) -> (rcompat_lbfgsb::OptimResult, Vec<Vec<f64>>) {
        let mut calls = Vec::new();
        let mut c = OptimControl::default_for_dimension(4);
        c.trace = trace;
        c.report = 1;
        let result = optim_lbfgsb(
            vec![2.0; 4],
            Bounds::new(vec![-10.0; 4], vec![10.0; 4]).unwrap(),
            |x| {
                calls.push(x.to_vec());
                x.iter().map(|v| (v - 1.0).powi(2)).sum()
            },
            c,
        )
        .unwrap();
        (result, calls)
    }
    assert_eq!(run(0), run(2));
}
