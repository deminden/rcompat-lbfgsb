//! Public numerical validity and termination regressions.
use rcompat_lbfgsb::{Bounds, OptimControl, OptimError, optim_lbfgsb, optim_lbfgsb_with_gradient};

fn assert_numerical_failure(
    error: OptimError,
    operation: &'static str,
    index: Option<usize>,
    value: f64,
) {
    assert_eq!(
        error,
        OptimError::NumericalFailure {
            operation,
            index,
            value
        }
    );
}

#[test]
fn scaled_objective_overflow_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 1e-320;
    let error = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::unbounded(1),
        |_| 1.0,
        |_| panic!("objective scaling overflow reached gradient callback"),
        c,
    )
    .unwrap_err();
    assert_numerical_failure(error, "objective scaling", None, f64::INFINITY);
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
        |_| panic!("parameter scaling overflow reached gradient callback"),
        c,
    );
    assert_numerical_failure(
        result.unwrap_err(),
        "parameter scaling",
        Some(0),
        f64::INFINITY,
    );
    assert!(!called);
}

#[test]
fn scaled_gradient_overflow_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.parscale[0] = 10.0;
    let error = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-1.0], vec![1.0]).unwrap(),
        |x| 1e308 * x[0],
        |_| vec![1e308],
        c,
    )
    .unwrap_err();
    assert_numerical_failure(error, "gradient scaling", Some(0), f64::INFINITY);
}

#[test]
fn overflowing_finite_difference_is_an_error() {
    let mut c = OptimControl::default_for_dimension(1);
    c.ndeps[0] = 1e308;
    let error = optim_lbfgsb(vec![0.0], Bounds::unbounded(1), |x| x[0], c).unwrap_err();
    assert_numerical_failure(
        error,
        "finite-difference denominator",
        Some(0),
        f64::INFINITY,
    );
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
        let error = b.contains(&[value]).unwrap_err();
        match error {
            OptimError::NonFiniteInitialParameter {
                index,
                value: actual,
            } => {
                assert_eq!(index, 0);
                assert_eq!(actual.to_bits(), value.to_bits());
            }
            other => panic!("expected non-finite parameter error for {value}, got {other:?}"),
        }
    }
}

#[test]
fn finite_bound_overflow_is_rejected_before_callback() {
    let mut c = OptimControl::default_for_dimension(1);
    c.parscale[0] = 1e-320;
    let error = optim_lbfgsb_with_gradient(
        vec![0.0],
        Bounds::new(vec![-2.0], vec![2.0]).unwrap(),
        |_| panic!("invalid scaled bounds reached objective callback"),
        |_| panic!("invalid scaled bounds reached gradient callback"),
        c,
    )
    .unwrap_err();
    assert_numerical_failure(error, "bound scaling", Some(0), f64::NEG_INFINITY);
}

#[test]
fn unrepresentable_stencil_is_rejected() {
    let error = optim_lbfgsb(
        vec![1e20],
        Bounds::unbounded(1),
        |x| x[0],
        OptimControl::default_for_dimension(1),
    )
    .unwrap_err();
    assert!(
        matches!(error, OptimError::InvalidControl { field: "ndeps", ref reason }
        if reason.contains("cannot perturb coordinate 0")),
        "{error:?}"
    );
}

#[test]
fn derivative_overflow_with_finite_stencil_is_rejected() {
    let mut c = OptimControl::default_for_dimension(1);
    c.ndeps[0] = 1.0;
    let error = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-1.0], vec![1.0]).unwrap(),
        |x| 1e308 * x[0],
        c,
    )
    .unwrap_err();
    assert_numerical_failure(error, "finite-difference gradient", Some(0), f64::INFINITY);
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
