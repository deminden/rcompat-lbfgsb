//! Black-box R 4.6.2 oracle for mixed/unbounded routing and first-step behavior.
#![allow(missing_docs)]
mod support;
use rcompat_lbfgsb::*;
use serde_json::Value;
#[test]
fn mixed_and_unbounded_paths_match_recorded_r_results() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/mixed_bounds/cases.json")).unwrap();
    assert!(fixture["r_version"].as_str().unwrap().contains("4.6.2"));
    assert_eq!(fixture["platform"], "aarch64-apple-darwin27.2.0");
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 37);
    assert_eq!(fixture["generator_sha256"].as_str().unwrap().len(), 64);
    let mut differences = support::MaxAbsoluteDifferences::default();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let initial = numbers(&case["initial"]);
        let n = initial.len();
        let target = numbers(&case["target"]);
        let weights = numbers(&case["weights"]);
        let scale = case["scale"].as_f64().unwrap();
        let linear = case["linear"].as_bool().unwrap();
        let offset = case["offset"].as_f64().unwrap();
        let bounds = Bounds::new(numbers(&case["lower"]), numbers(&case["upper"])).unwrap();
        let mut calls = Vec::new();
        let f = |x: &[f64]| {
            calls.push(x.to_vec());
            let mut value = 0.0;
            for i in 0..n {
                let d = x[i] - target[i];
                value += if linear { -x[i] } else { weights[i] * d * d };
            }
            value * scale + offset
        };
        let g = |x: &[f64]| {
            (0..n)
                .map(|i| {
                    if linear {
                        -scale
                    } else {
                        (2.0 * weights[i]) * (x[i] - target[i]) * scale
                    }
                })
                .collect()
        };
        let mut c = OptimControl::default_for_dimension(n);
        c.maxit = case["control"]["maxit"].as_u64().unwrap() as usize;
        c.factr = case["control"]["factr"].as_f64().unwrap();
        let actual = if case["gradient"].as_bool().unwrap() {
            optim_lbfgsb_with_gradient_and_profile(
                initial,
                bounds,
                f,
                g,
                c,
                ArithmeticProfile::AppleSiliconOpenBlas,
            )
        } else {
            optim_lbfgsb_with_profile(
                initial,
                bounds,
                f,
                c,
                ArithmeticProfile::AppleSiliconOpenBlas,
            )
        }
        .unwrap();
        let expected = OptimResult {
            par: numbers(&case["result"]["par"]),
            value: case["result"]["value"].as_f64().unwrap(),
            counts: OptimCounts {
                function: case["result"]["counts"]["function"].as_u64().unwrap() as usize,
                gradient: case["result"]["counts"]["gradient"].as_u64().unwrap() as usize,
            },
            convergence: case["result"]["convergence"].as_i64().unwrap() as i32,
            message: case["result"]["message"].as_str().unwrap().to_owned(),
        };
        let exact = case["contract"] == "exact";
        let contract = if exact {
            support::Contract::Exact
        } else {
            support::Contract::Numerical {
                parameters: 1e-8,
                value: 1e-10,
            }
        };
        differences.record(&actual, &expected);
        let mut issues = support::compare(&actual, &expected, contract);
        // Trace comparison covers the normalized first trial and finite-difference order.
        let prefix = case["first_calls"].as_array().unwrap();
        if calls.len() < prefix.len() {
            issues.push("raw callback prefix truncated".to_owned());
        }
        for (index, (a, b)) in calls
            .iter()
            .zip(prefix)
            .take(if case["gradient"].as_bool().unwrap() {
                2
            } else {
                2 * n + 2
            })
            .enumerate()
        {
            let expected_call = numbers(b);
            if a.len() != n
                || expected_call.len() != n
                || a.iter().zip(expected_call).any(|(x, y)| {
                    if exact {
                        x.to_bits() != y.to_bits()
                    } else {
                        (x - y).abs() > 1e-12
                    }
                })
            {
                issues.push(format!("raw callback {index} mismatch"));
                break;
            }
        }
        if !issues.is_empty() {
            failures.push(format!("{}: {}", case["id"], issues.join("; ")));
        }
    }
    differences.report("mixed_bounds");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

use support::fixtures::numbers;
