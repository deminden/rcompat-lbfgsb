//! Explicit arithmetic selection is local, reentrant and feature independent.
use rcompat_lbfgsb::*;

fn objective(x: &[f64]) -> f64 {
    100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
}
fn gradient(x: &[f64]) -> Vec<f64> {
    vec![
        -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
        200.0 * (x[1] - x[0] * x[0]),
    ]
}
fn bounds() -> Bounds {
    Bounds::new(vec![-5.0; 2], vec![5.0; 2]).unwrap()
}
fn run(profile: ArithmeticProfile, numeric: bool) -> OptimResult {
    let c = OptimControl::default_for_dimension(2);
    if numeric {
        optim_lbfgsb_with_profile(vec![-1.2, 1.0], bounds(), objective, c, profile)
    } else {
        optim_lbfgsb_with_gradient_and_profile(
            vec![-1.2, 1.0],
            bounds(),
            objective,
            gradient,
            c,
            profile,
        )
    }
    .unwrap()
}
fn bits(result: &OptimResult) -> (Vec<u64>, u64, OptimCounts, i32, String) {
    (
        result.par.iter().map(|v| v.to_bits()).collect(),
        result.value.to_bits(),
        result.counts,
        result.convergence,
        result.message.clone(),
    )
}

#[test]
fn legacy_entry_points_keep_their_arithmetic_contract() {
    let c = OptimControl::default_for_dimension(2);
    let plain = optim_lbfgsb(vec![-1.2, 1.0], bounds(), objective, c.clone()).unwrap();
    let analytic =
        optim_lbfgsb_with_gradient(vec![-1.2, 1.0], bounds(), objective, gradient, c).unwrap();
    assert_eq!(bits(&plain), bits(&run(ArithmeticProfile::legacy(), true)));
    assert_eq!(
        bits(&analytic),
        bits(&run(ArithmeticProfile::legacy(), false))
    );
}

#[test]
fn explicit_profiles_are_deterministic_when_alternated_and_run_concurrently() {
    for numeric in [false, true] {
        let x86 = bits(&run(ArithmeticProfile::R461X86, numeric));
        let arm = bits(&run(ArithmeticProfile::R461MacArm, numeric));
        assert_ne!(x86, arm, "probe must expose the rounding distinction");
        for _ in 0..3 {
            assert_eq!(bits(&run(ArithmeticProfile::R461MacArm, numeric)), arm);
            assert_eq!(bits(&run(ArithmeticProfile::R461X86, numeric)), x86);
        }
        let a = std::thread::spawn(move || bits(&run(ArithmeticProfile::R461MacArm, numeric)));
        let b = std::thread::spawn(move || bits(&run(ArithmeticProfile::R461X86, numeric)));
        assert_eq!(a.join().unwrap(), arm);
        assert_eq!(b.join().unwrap(), x86);
    }
}

#[test]
fn nested_callback_with_another_profile_does_not_change_outer_run() {
    for outer in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
        let inner = if outer == ArithmeticProfile::R461X86 {
            ArithmeticProfile::R461MacArm
        } else {
            ArithmeticProfile::R461X86
        };
        let expected = bits(&run(outer, false));
        let mut first = true;
        let result = optim_lbfgsb_with_gradient_and_profile(
            vec![-1.2, 1.0],
            bounds(),
            |x| {
                if first {
                    first = false;
                    let nested = run(inner, true);
                    assert!(nested.value.is_finite());
                }
                objective(x)
            },
            gradient,
            OptimControl::default_for_dimension(2),
            outer,
        )
        .unwrap();
        assert_eq!(bits(&result), expected);
    }
}
