//! Equivalent dyadic changes of user units must preserve the internal run.
#![allow(missing_docs)]
mod support;
use rcompat_lbfgsb::*;

fn replay(
    n: usize,
    route: usize,
    numeric: bool,
    profile: ArithmeticProfile,
    pattern: usize,
    units: f64,
) -> (OptimResult, Vec<Vec<f64>>) {
    let scales: Vec<_> = (0..n)
        .map(|i| match pattern {
            0 => 1.0,
            1 => {
                if i % 2 == 0 {
                    0.5
                } else {
                    2.0
                }
            }
            _ => {
                if i % 2 == 0 {
                    8.0
                } else {
                    0.25
                }
            }
        })
        .collect();
    let mut lower = vec![f64::NEG_INFINITY; n];
    let mut upper = vec![f64::INFINITY; n];
    if route > 0 {
        for i in 0..n {
            lower[i] = -3.0 * scales[i];
            if route == 2 || i % 2 == 0 {
                upper[i] = 1.0 * scales[i];
            }
        }
    }
    let bounds = Bounds::new(lower, upper).unwrap();
    let initial: Vec<_> = (0..n)
        .map(|i| (if i % 2 == 0 { 0.0 } else { -0.5 }) * scales[i])
        .collect();
    let mut control = OptimControl::default_for_dimension(n);
    control.parscale = scales.clone();
    control.fnscale = units;
    control.pgtol = 1e-7;
    let mut calls = Vec::new();
    let f = |x: &[f64]| {
        let q: Vec<_> = x.iter().zip(&scales).map(|(x, s)| x / s).collect();
        calls.push(q.clone());
        let mut sum = 0.0;
        let mut value = 0.0;
        for (i, x) in q.iter().enumerate() {
            let d = x - if i % 2 == 0 { 2.0 } else { -1.0 };
            sum += d;
            value += 0.5 * (1 + i % 3) as f64 * d * d;
        }
        (value + 0.125 * sum * sum) * units
    };
    let g = |x: &[f64]| {
        let q: Vec<_> = x.iter().zip(&scales).map(|(x, s)| x / s).collect();
        let mut sum = 0.0;
        for (i, x) in q.iter().enumerate() {
            sum += x - if i % 2 == 0 { 2.0 } else { -1.0 };
        }
        q.iter()
            .enumerate()
            .map(|(i, x)| {
                let canonical =
                    (1 + i % 3) as f64 * (x - if i % 2 == 0 { 2.0 } else { -1.0 }) + 0.25 * sum;
                canonical * units / scales[i]
            })
            .collect()
    };
    let mut result = if numeric {
        optim_lbfgsb_with_profile(initial, bounds, f, control, profile)
    } else {
        optim_lbfgsb_with_gradient_and_profile(initial, bounds, f, g, control, profile)
    }
    .unwrap();
    for (x, s) in result.par.iter_mut().zip(&scales) {
        *x /= s;
    }
    result.value /= units;
    (result, calls)
}

#[test]
fn parameter_objective_and_maximization_transformations_preserve_every_run_bit() {
    let mut checked = 0;
    for n in [1, 2, 9, 10, 11, 13, 128] {
        for route in 0..3 {
            for profile in [
                ArithmeticProfile::X86OpenBlas,
                ArithmeticProfile::AppleSiliconOpenBlas,
            ] {
                for numeric in [false, true] {
                    if numeric && n > 13 {
                        continue;
                    }
                    let (expected, expected_calls) = replay(n, route, numeric, profile, 0, 1.0);
                    for pattern in 0..3 {
                        for units in [2.0_f64.powi(-20), 1.0, 2.0_f64.powi(20)] {
                            for sign in [-1.0, 1.0] {
                                let (actual, calls) =
                                    replay(n, route, numeric, profile, pattern, sign * units);
                                let issues =
                                    support::compare(&actual, &expected, support::Contract::Exact);
                                assert!(
                                    issues.is_empty(),
                                    "n={n} route={route} numeric={numeric} profile={profile:?} pattern={pattern} units={units} sign={sign}: {}",
                                    issues.join("; ")
                                );
                                assert_eq!(calls.len(), expected_calls.len());
                                for (a, b) in calls.iter().zip(&expected_calls) {
                                    assert_eq!(a.len(), b.len());
                                    assert!(
                                        a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
                                    );
                                }
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 1404);
}
