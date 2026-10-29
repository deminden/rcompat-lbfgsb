//! Scientific accuracy against the independently captured native R 4.6.2 stack.
#![allow(missing_docs)]
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[path = "../examples/support/deseq2_objective.rs"]
mod objective;
mod support;
use rcompat_lbfgsb::*;
use serde_json::Value;
fn check(
    cohort: &str,
    inputs: Value,
    reference: Value,
    total: usize,
    value_tolerance: f64,
    coefficient_limit: f64,
) {
    assert!(reference["r_version"].as_str().unwrap().contains("4.6.2"));
    assert_eq!(reference["deseq2_version"], "1.52.0");
    assert_eq!(reference["threads"], "1");
    for key in ["CFLAGS", "CXXFLAGS", "FFLAGS"] {
        assert!(
            reference["build_config"][key]
                .as_str()
                .unwrap()
                .contains("-mcpu=native")
        );
    }
    let mut differences = support::MaxAbsoluteDifferences::default();
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
            let f =
                objective::Deseq2Objective::new(design.clone(), sf.clone(), &counts, dispersion);
            let new = optim_lbfgsb_with_profile(
                start,
                bounds,
                |x| f.value(x),
                OptimControl::default_for_dimension(n),
                ArithmeticProfile::AppleSiliconOpenBlas,
            )
            .unwrap();
            differences.record(&new, &expected);
            let comparison =
                support::compare_scientific(&new, &expected, coefficient_limit, value_tolerance);
            if !comparison.failures.is_empty() {
                failures.push(format!(
                    "{}: {}",
                    case["fixture"],
                    comparison.failures.join("; ")
                ));
            }
            observed += usize::from(!comparison.bookkeeping_differences.is_empty());
            new_errors.push(comparison.max_parameter_error);
        }
    }
    differences.report(cohort);
    assert_eq!(new_errors.len(), total);
    new_errors.sort_by(f64::total_cmp);
    let new_max = new_errors[total - 1];
    assert!(
        new_max <= coefficient_limit,
        "absolute coefficient accuracy regressed: {new_max}"
    );
    eprintln!(
        "R 4.6.2 native: {total} fits; worst coefficient error {new_max:e}; {observed} observed count/message differences"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn native_r462_hard_fits_preserve_precision() {
    check(
        "deseq2_hard",
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
        0.0014,
    );
}
#[test]
fn native_r462_airway_fits_preserve_precision() {
    check(
        "deseq2_airway",
        serde_json::from_str(include_str!("../fixtures/deseq_airway/cases.json")).unwrap(),
        serde_json::from_str(include_str!(
            "../fixtures/native_r462/airway_reference.json"
        ))
        .unwrap(),
        28,
        1e-9,
        3.8e-6,
    );
}

use support::fixtures::numbers as nums;
