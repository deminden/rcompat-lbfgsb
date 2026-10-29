#![allow(missing_docs)]
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

#[path = "../examples/support/deseq2_objective.rs"]
mod objective;

mod support;
use objective::Deseq2Objective;
use rcompat_lbfgsb::{optim_lbfgsb_with_profile, ArithmeticProfile, Bounds, OptimControl};
use serde_json::Value;

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn check(fixture: Value, reference: Value, expected_cases: usize, value_tolerance: f64) {
    let baseline: Value =
        serde_json::from_str(include_str!("../fixtures/deseq2_precision_baseline.json")).unwrap();
    assert_eq!(reference["deseq2_version"], "1.52.0");
    assert_eq!(reference["platform"], "aarch64-apple-darwin20.0.0");
    assert!(reference["r_version"].as_str().unwrap().contains("4.6.1"));
    let mut errors = Vec::new();
    let mut old_errors = Vec::new();
    let mut max_increase = 0.0_f64;
    let mut improved = 0;
    let mut observed_bookkeeping_differences = 0;
    for contrast in fixture["contrasts"].as_array().unwrap() {
        let design: Vec<_> = contrast["design"]
            .as_array()
            .unwrap()
            .iter()
            .map(nums)
            .collect();
        let sf = nums(&contrast["size_factors"]);
        for case in contrast["cases"].as_array().unwrap() {
            if !case["actual_optim_routed"].as_bool().unwrap() {
                continue;
            }
            let fit = reference["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|fit| fit["fixture"] == case["fixture"])
                .unwrap();
            let old_error = baseline["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|fit| fit["fixture"] == case["fixture"])
                .unwrap()["max_coefficient_error"]
                .as_f64()
                .unwrap();
            let start = nums(&fit["optimizer_start"]);
            let dimension = start.len();
            let objective = Deseq2Objective::new(
                design.clone(),
                sf.clone(),
                &nums(&case["counts"]),
                case["dispersion"].as_f64().unwrap(),
            );
            let result = optim_lbfgsb_with_profile(
                start,
                Bounds::new(nums(&case["lower"]), nums(&case["upper"])).unwrap(),
                |beta| objective.value(beta),
                OptimControl::default_for_dimension(dimension),
                ArithmeticProfile::R461MacArm,
            )
            .unwrap();
            let expected = support::recorded_result(&fit["optimizer_result"]);
            assert_eq!(expected.par, nums(&fit["beta"]));
            let comparison =
                support::compare_scientific(&result, &expected, old_error + 1e-6, value_tolerance);
            assert!(
                comparison.failures.is_empty(),
                "{}: {}",
                case["fixture"],
                comparison.failures.join("; ")
            );
            observed_bookkeeping_differences +=
                usize::from(!comparison.bookkeeping_differences.is_empty());
            let error = comparison.max_parameter_error;
            if error <= old_error {
                improved += 1;
            }
            max_increase = max_increase.max(error - old_error);
            errors.push(error);
            old_errors.push(old_error);
        }
    }
    assert_eq!(errors.len(), expected_cases);
    errors.sort_by(f64::total_cmp);
    old_errors.sort_by(f64::total_cmp);
    let old_max = *old_errors.last().unwrap();
    let new_max = *errors.last().unwrap();
    assert!(
        new_max <= old_max / 10.0,
        "worst-case precision gate: old={old_max:e}, new={new_max:e}"
    );
    assert!(errors[expected_cases / 2] <= old_errors[expected_cases / 2]);
    eprintln!("{observed_bookkeeping_differences}/{expected_cases} fits differ in observed counts/message; scientific accuracy is checked separately");
    eprintln!("{expected_cases} fits: worst {old_max:e} -> {new_max:e} ({:.1}x); {improved} improved/equal; largest increase={max_increase:e}", old_max / new_max);
}

#[test]
fn old_real_deseq2_fits_improve_tenfold_without_material_regressions() {
    check(
        serde_json::from_str(include_str!(
            "../fixtures/deseq_hard_real_subset/optim_cases.json"
        ))
        .unwrap(),
        serde_json::from_str(include_str!(
            "../fixtures/deseq_hard_real_subset/deseq2_r461_nbinomWaldTest.json"
        ))
        .unwrap(),
        46,
        1e-6,
    );
}

#[test]
fn published_airway_fits_improve_tenfold_without_material_regressions() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/deseq_airway/cases.json")).unwrap();
    let reference = fixture["reference"].clone();
    check(fixture, reference, 28, 1e-9);
}
