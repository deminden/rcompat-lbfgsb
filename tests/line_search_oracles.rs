//! Black-box R failures, best-point restoration, and narrow-bracket warnings.
#![allow(missing_docs)]
mod support;
use rcompat_lbfgsb::*;
use serde_json::Value;

#[test]
fn search_outcomes_and_every_trial_match_r() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/line_search/cases.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 6);
    assert_eq!(fixture["platform"], "aarch64-apple-darwin27.2.0");
    let mut differences = support::MaxAbsoluteDifferences::default();
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let scenario = case["scenario"].as_str().unwrap();
        let bounds = if case["finite"].as_bool().unwrap() {
            Bounds::new(vec![-5.0], vec![5.0]).unwrap()
        } else {
            Bounds::unbounded(1)
        };
        let mut control = OptimControl::default_for_dimension(1);
        control.factr = 0.0;
        control.maxit = case["control"]["maxit"].as_u64().unwrap() as usize;
        let mut calls = Vec::new();
        let mut events = Vec::new();
        let actual = optim_lbfgsb_with_gradient_observed(
            numbers(&case["initial"]),
            bounds,
            |x| {
                calls.push(x.to_vec());
                if scenario == "absolute_warning" {
                    (x[0] - 0.12345).abs()
                } else {
                    x[0] * x[0]
                }
            },
            |x| {
                vec![match scenario {
                    "wrong_gradient" => -2.0 * x[0],
                    "absolute_warning" => {
                        if x[0] < 0.12345 {
                            -1.0
                        } else {
                            1.0
                        }
                    }
                    "history_exhaustion" => {
                        if x[0] > 1.5 {
                            2.0 * x[0]
                        } else {
                            -x[0]
                        }
                    }
                    _ => panic!("unknown scenario"),
                }]
            },
            control,
            ArithmeticProfile::AppleSiliconOpenBlas,
            |e| events.push(e),
        )
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
        differences.record(&actual, &expected);
        let mut issues = support::compare(&actual, &expected, support::Contract::Exact);
        let expected_calls = case["calls"].as_array().unwrap();
        if calls.len() != expected_calls.len() {
            issues.push("raw callback count differs".to_owned());
        }
        for (i, (a, b)) in calls.iter().zip(expected_calls).enumerate() {
            let b = numbers(b);
            if a.len() != b.len() || a.iter().zip(&b).any(|(a, b)| a.to_bits() != b.to_bits()) {
                issues.push(format!("raw callback {i} bits differ: {a:?} vs {b:?}"));
                break;
            }
        }
        let resets = events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    OptimEvent::MemoryReset {
                        reason: RecoveryReason::LineSearch,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(resets, usize::from(scenario == "history_exhaustion"));
        if scenario == "absolute_warning" {
            assert!(events.iter().any(|e| matches!(
                e,
                OptimEvent::Accepted {
                    acceptance: StepAcceptance::BracketWarning,
                    wolfe_curvature: false,
                    ..
                }
            )));
            // R revisits the best point after a rejected trial. The last raw
            // callback must restore that earlier point, not return the last trial.
            assert_eq!(calls.last(), calls.get(calls.len() - 3));
            assert_ne!(calls.last(), calls.get(calls.len() - 2));
        } else {
            let failed: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    OptimEvent::LineSearchFailed {
                        reason,
                        evaluations,
                        ..
                    } => Some((*reason, *evaluations)),
                    _ => None,
                })
                .collect();
            assert_eq!(
                failed,
                vec![(LineSearchFailure::TrialLimit, 20); 1 + resets]
            );
            let accepted: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    OptimEvent::Accepted { counts, .. } => Some(counts.function - 1),
                    _ => None,
                })
                .collect();
            assert_eq!(accepted.len(), resets);
            let retained = accepted.last().copied().unwrap_or(0);
            assert_eq!(actual.par, calls[retained]);
            assert!(calls[retained + 1..].iter().any(|x| x != &actual.par));
        }
        if !issues.is_empty() {
            failures.push(format!("{}: {}", case["id"], issues.join("; ")));
        }
    }
    differences.report("line_search");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

use support::fixtures::numbers;
