//! Complete-result replay of the self-contained replacement stress corpus.
#![allow(missing_docs)]
mod objective;
#[path = "../support/mod.rs"]
mod support;
use objective::Problem;
mod witness;
use witness::active_set_trace_issues;
mod accuracy;
mod corruption;
use rcompat_lbfgsb::*;
use serde_json::Value;
use std::collections::HashSet;

fn fixture() -> Value {
    let path = std::env::var_os("LBFGSB_STRESS_ORACLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/synthetic_v1/cases.json")
        });
    serde_json::from_str(&std::fs::read_to_string(path).expect("stress oracle is required"))
        .unwrap()
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "full R replay requires the recorded macOS ARM callback math; mandatory in the ARM CI job"
)]
fn stress_corpus_matches_r() {
    replay_corpus(false);
}

#[test]
fn polynomial_stress_corpus_matches_r_on_portable_hosts() {
    replay_corpus(true);
}

fn replay_corpus(polynomial_only: bool) {
    let fixture = fixture();
    assert_eq!(fixture["identity"], "synthetic_v1");
    assert_eq!(
        fixture["arithmetic_profile"],
        ArithmeticProfile::AppleSiliconOpenBlas.id()
    );
    let cases = fixture["cases"].as_array().unwrap();
    let witness: Value =
        serde_json::from_str(include_str!("../../fixtures/active_set/case_trace.json")).unwrap();
    assert_eq!(
        witness["arithmetic_profile"],
        ArithmeticProfile::AppleSiliconOpenBlas.id()
    );
    assert!(cases.iter().any(|c| c["id"] == witness["id"]));
    assert_eq!(cases.len(), 512, "incomplete stress corpus");
    let mut identities = HashSet::new();
    for c in cases {
        assert!(
            identities.insert(c["id"].as_str().unwrap()),
            "duplicate stress case"
        );
    }
    let selected = std::env::var("LBFGSB_STRESS_CASE").ok();
    let mut executed = 0;
    let mut failures = Vec::new();
    let mut differences = support::MaxAbsoluteDifferences::default();
    let mut contract_matches = 0;
    for c in cases {
        let id = c["id"].as_str().unwrap();
        if polynomial_only && c["family"] == "negative_binomial" {
            continue;
        }
        if selected.as_ref().is_some_and(|s| s != id) {
            continue;
        }
        executed += 1;
        let p = Problem::new(c);
        let initial = numbers(&c["initial"]);
        assert_eq!(initial.len(), p.n);
        let bounds = Bounds::new(numbers(&c["lower"]), numbers(&c["upper"])).unwrap();
        let mut control = OptimControl::default_for_dimension(p.n);
        control.maxit = c["control"]["maxit"].as_u64().unwrap() as usize;
        control.fnscale = p.sign;
        control.parscale = numbers(&c["control"]["parscale"]);
        control.ndeps = numbers(&c["control"]["ndeps"]);
        control.factr = c["control"]["factr"].as_f64().unwrap();
        control.pgtol = c["control"]["pgtol"].as_f64().unwrap();
        control.lmm = c["control"]["lmm"].as_u64().unwrap() as usize;
        let trace_path = std::env::var_os("LBFGSB_STRESS_TRACE");
        let active_set_witness = c["id"] == witness["id"];
        let capture_calls = trace_path.is_some() || active_set_witness;
        assert!(
            trace_path.is_none() || selected.is_some(),
            "a callback trace requires one selected case"
        );
        let mut value_calls = Vec::new();
        let mut gradient_calls = Vec::new();
        let f = |x: &[f64]| {
            let value = p.value(x);
            if capture_calls {
                value_calls.push(serde_json::json!({"x":x,"value":value}));
            }
            value
        };
        let g = |x: &[f64]| {
            let gradient = p.gradient(x);
            if capture_calls {
                gradient_calls.push(serde_json::json!({"x":x,"gradient":gradient}));
            }
            gradient
        };
        let mut trace_events = Vec::new();
        let result = if trace_path.is_some() && c["gradient"].as_bool().unwrap() {
            optim_lbfgsb_with_gradient_observed(
                initial,
                bounds,
                f,
                g,
                control,
                ArithmeticProfile::AppleSiliconOpenBlas,
                |event| trace_events.push(format!("{event:?}")),
            )
        } else if trace_path.is_some() {
            optim_lbfgsb_observed(
                initial,
                bounds,
                f,
                control,
                ArithmeticProfile::AppleSiliconOpenBlas,
                |event| trace_events.push(format!("{event:?}")),
            )
        } else if c["gradient"].as_bool().unwrap() {
            optim_lbfgsb_with_gradient_and_profile(
                initial,
                bounds,
                f,
                g,
                control,
                ArithmeticProfile::AppleSiliconOpenBlas,
            )
        } else {
            optim_lbfgsb_with_profile(
                initial,
                bounds,
                f,
                control,
                ArithmeticProfile::AppleSiliconOpenBlas,
            )
        };
        if let Some(path) = trace_path {
            std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"id":id,"value_calls":value_calls,"gradient_calls":gradient_calls,"events":trace_events})).unwrap()).unwrap();
        }
        let expected = support::recorded_result(&c["result"]);
        match result {
            Ok(actual) => {
                differences.record(&actual, &expected);
                let mut issues = support::compare(
                    &actual,
                    &expected,
                    support::Contract::Numerical {
                        parameters: 1e-8,
                        value: 1e-8,
                    },
                );
                if active_set_witness {
                    issues.extend(active_set_trace_issues(&actual, &value_calls, &witness));
                }
                if issues.is_empty() {
                    contract_matches += 1;
                } else {
                    failures.push(format!("{id}: {}", issues.join("; ")));
                }
            }
            Err(e) => failures.push(format!("{id}: {e}")),
        }
    }
    assert_eq!(
        executed,
        if selected.is_some() {
            1
        } else if polynomial_only {
            384
        } else {
            512
        },
        "stress selection must execute every requested case"
    );
    differences.report(if polynomial_only {
        "polynomial_stress"
    } else {
        "stress"
    });
    eprintln!("stress full-result matches: {contract_matches}/{executed}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

use support::fixtures::numbers;
