#![cfg(feature = "arm-fma")]
#![allow(missing_docs)]

#[path = "../examples/support/deseq2_objective.rs"]
mod objective;

use objective::Deseq2Objective;
use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};
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
            let result = optim_lbfgsb(
                start,
                Bounds::new(nums(&case["lower"]), nums(&case["upper"])).unwrap(),
                |beta| objective.value(beta),
                OptimControl::default_for_dimension(dimension),
            )
            .unwrap();
            let error = result
                .par
                .iter()
                .zip(nums(&fit["beta"]))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                error <= old_error + 1e-6,
                "{} regressed beyond the agreed 1e-6 allowance: old={old_error:e}, new={error:e}",
                case["fixture"]
            );
            assert_eq!(
                result.convergence as i64,
                fit["optimizer_result"]["convergence"].as_i64().unwrap(),
                "{} changed convergence status",
                case["fixture"]
            );
            assert!(
                (result.value - fit["optimizer_result"]["value"].as_f64().unwrap()).abs()
                    <= value_tolerance,
                "{} objective scale/endpoint drift",
                case["fixture"]
            );
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
