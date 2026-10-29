//! Scientific accuracy against the independently captured native R 4.6.2 stack.
#![allow(missing_docs)]
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[path = "../examples/support/deseq2_objective.rs"]
mod objective;
#[path = "support/previous_deseq2.rs"]
mod previous;
mod support;
use rcompat_lbfgsb::*;
use serde_json::Value;
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn check(inputs: Value, reference: Value, total: usize, value_tolerance: f64) {
    assert!(reference["r_version"].as_str().unwrap().contains("4.6.2"));
    assert_eq!(reference["deseq2_version"], "1.52.0");
    assert_eq!(reference["threads"], "1");
    for key in ["CFLAGS", "CXXFLAGS", "FFLAGS"] {
        assert!(reference["build_config"][key]
            .as_str()
            .unwrap()
            .contains("-mcpu=native"));
    }
    let mut old_errors = Vec::new();
    let mut new_errors = Vec::new();
    let mut observed = 0;
    let mut failures = Vec::new();
    for contrast in inputs["contrasts"].as_array().unwrap() {
        let design: Vec<_> = contrast["design"]
            .as_array()
            .unwrap()
            .iter()
            .map(nums)
            .collect();
        let sf = nums(&contrast["size_factors"]);
        for case in contrast["cases"].as_array().unwrap() {
            let fit = reference["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["fixture"] == case["fixture"])
                .unwrap();
            if fit["optimizer_start"].is_null() {
                continue;
            }
            let start = nums(&fit["optimizer_start"]);
            let n = start.len();
            let counts = nums(&case["counts"]);
            let dispersion = case["dispersion"].as_f64().unwrap();
            let bounds = Bounds::new(nums(&case["lower"]), nums(&case["upper"])).unwrap();
            let expected = support::recorded_result(&fit["optimizer_result"]);
            let old = optim_lbfgsb_with_profile(
                start.clone(),
                bounds.clone(),
                |x| previous::value(x, &counts, &design, &sf, dispersion),
                OptimControl::default_for_dimension(n),
                ArithmeticProfile::R461X86,
            )
            .unwrap();
            let old_comparison = support::compare_scientific(&old, &expected, f64::MAX, f64::MAX);
            assert!(
                old_comparison.failures.is_empty(),
                "old callback validity: {}",
                old_comparison.failures.join("; ")
            );
            let old_error = old_comparison.max_parameter_error;
            let f =
                objective::Deseq2Objective::new(design.clone(), sf.clone(), &counts, dispersion);
            let new = optim_lbfgsb_with_profile(
                start,
                bounds,
                |x| f.value(x),
                OptimControl::default_for_dimension(n),
                ArithmeticProfile::R461MacArm,
            )
            .unwrap();
            let comparison =
                support::compare_scientific(&new, &expected, old_error + 1e-6, value_tolerance);
            if !comparison.failures.is_empty() {
                failures.push(format!(
                    "{}: {}",
                    case["fixture"],
                    comparison.failures.join("; ")
                ));
            }
            observed += usize::from(!comparison.bookkeeping_differences.is_empty());
            old_errors.push(old_error);
            new_errors.push(comparison.max_parameter_error);
        }
    }
    assert_eq!(new_errors.len(), total);
    old_errors.sort_by(f64::total_cmp);
    new_errors.sort_by(f64::total_cmp);
    let old_max = old_errors[total - 1];
    let new_max = new_errors[total - 1];
    eprintln!("R 4.6.2 native: {total} fits; worst {old_max:e} -> {new_max:e} ({:.1}x); {observed} observed count/message differences",old_max/new_max);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        new_max <= old_max / 10.0,
        "worst-case precision gate: old={old_max:e}, new={new_max:e}"
    );
    assert!(
        new_errors[total / 2] <= old_errors[total / 2],
        "median accuracy regressed"
    );
}
#[test]
fn native_r462_historical_fits_preserve_precision_gains() {
    check(
        serde_json::from_str(include_str!(
            "../fixtures/deseq_hard_real_subset/optim_cases.json"
        ))
        .unwrap(),
        serde_json::from_str(include_str!(
            "../fixtures/native_r462/hard_real_reference.json"
        ))
        .unwrap(),
        46,
        1e-6,
    );
}
#[test]
fn native_r462_airway_fits_preserve_precision_gains() {
    check(
        serde_json::from_str(include_str!("../fixtures/deseq_airway/cases.json")).unwrap(),
        serde_json::from_str(include_str!(
            "../fixtures/native_r462/airway_reference.json"
        ))
        .unwrap(),
        28,
        1e-9,
    );
}
