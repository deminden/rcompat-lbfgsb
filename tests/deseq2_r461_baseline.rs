#![allow(missing_docs)]
#![cfg(all(target_arch = "aarch64", target_os = "macos"))]

mod support;
use rcompat_lbfgsb::{Bounds, OptimControl};
use serde_json::Value;
use support::optim_x86 as optim_lbfgsb;

#[test]
fn full_nb_likelihood_with_deseq2_prior_replays_the_real_route() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/deseq_hard_real_subset/optim_cases.json"
    ))
    .unwrap();
    let baseline: Value = serde_json::from_str(include_str!(
        "../fixtures/deseq_hard_real_subset/deseq2_r461_nbinomWaldTest.json"
    ))
    .unwrap();
    let mut routed_errors = Vec::new();
    let mut reduced_errors = Vec::new();
    let mut error_cases = Vec::new();
    let mut max_initial_delta: f64 = 0.0;
    assert_eq!(baseline["deseq2_version"], "1.52.0");
    assert!(baseline["r_version"].as_str().unwrap().contains("4.6.1"));
    assert_eq!(baseline["results"].as_array().unwrap().len(), 48);
    for contrast in fixture["contrasts"].as_array().unwrap() {
        let x = contrast["design"].as_array().unwrap();
        let sf = nums(&contrast["size_factors"]);
        let k = contrast["coefficients"].as_u64().unwrap() as usize;
        // DESeq2 calls stats::optim with its default control list. The source
        // objective-only fixture has an independently recorded replay setup;
        // use the captured wrapper route here instead.
        let mut control = OptimControl::default_for_dimension(k);
        control.maxit = 100;
        for case in contrast["cases"].as_array().unwrap() {
            if !case["actual_optim_routed"].as_bool().unwrap() {
                continue;
            }
            let expected = baseline["results"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| {
                    entry["contrast"] == contrast["contrast"] && entry["gene"] == case["gene"]
                })
                .unwrap();
            assert_eq!(expected["optimizer_has_gradient"], false);
            let optimizer_start = nums(&expected["optimizer_start"]);
            max_initial_delta = max_initial_delta.max(
                optimizer_start
                    .iter()
                    .zip(nums(&case["initial_par"]))
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
            );
            let y = nums(&case["counts"]);
            let r = 1.0 / case["dispersion"].as_f64().unwrap();
            let objective_offset = 0.5 * k as f64 * (2.0 * std::f64::consts::PI * 1e6).ln();
            let observed_offset = expected["objective_offset_start"].as_f64().unwrap();
            assert!(
                (observed_offset - objective_offset).abs() < 3e-5,
                "DESeq2 objective scale mismatch: observed={observed_offset}, formula={objective_offset}"
            );
            let captured_endpoint = nums(&expected["optimizer_result"]["par"]);
            let endpoint_prior = captured_endpoint.iter().map(|b| b * b).sum::<f64>() / 2e6;
            let observed_endpoint_offset = expected["objective_offset_endpoint"].as_f64().unwrap();
            assert!(
                (observed_endpoint_offset - objective_offset - endpoint_prior).abs() < 1e-9,
                "DESeq2 endpoint offset does not match its Gaussian coefficient prior: observed={observed_endpoint_offset}, expected={}",
                objective_offset + endpoint_prior
            );
            let constant: f64 = y
                .iter()
                .map(|&yi| log_gamma(yi + r) - log_gamma(r) - log_gamma(yi + 1.0))
                .sum();
            let result = optim_lbfgsb(
                optimizer_start.clone(),
                Bounds::new(nums(&case["lower"]), nums(&case["upper"])).unwrap(),
                |beta| {
                    let mut value = objective_offset - constant;
                    for (i, &yi) in y.iter().enumerate() {
                        let mut linear = 0.0;
                        for j in 0..beta.len() {
                            linear += x[i][j].as_f64().unwrap() * beta[j];
                        }
                        let mu = (std::f64::consts::LN_2 * linear + sf[i].ln()).exp();
                        value += (yi + r) * (r + mu).ln();
                        value -= r * r.ln();
                        value -= yi * mu.ln();
                    }
                    // DESeq2's coefficient prior has sd=1000. Its normalizing
                    // constant is objective_offset; retain the matching
                    // quadratic term as well, since near-flat fits amplify it.
                    value += beta.iter().map(|b| b * b).sum::<f64>() / (2.0 * 1e6);
                    value
                },
                control.clone(),
            )
            .unwrap();
            let expected_result = support::recorded_result(&expected["optimizer_result"]);
            let comparison =
                support::compare_scientific(&result, &expected_result, f64::MAX, f64::MAX);
            assert!(
                comparison.failures.is_empty(),
                "{}: {}",
                case["fixture"],
                comparison.failures.join("; ")
            );
            let error = comparison.max_parameter_error;
            routed_errors.push(error);
            error_cases.push((error, case["fixture"].as_str().unwrap().to_owned()));
            let reduced = optim_lbfgsb(
                optimizer_start,
                Bounds::new(nums(&case["lower"]), nums(&case["upper"])).unwrap(),
                |beta| reduced_nb_nll(beta, &y, x, &sf, r),
                control.clone(),
            )
            .unwrap();
            let reduced_comparison =
                support::compare_scientific(&reduced, &expected_result, f64::MAX, f64::MAX);
            assert!(
                reduced_comparison.failures.is_empty(),
                "{} reduced objective: {}",
                case["fixture"],
                reduced_comparison.failures.join("; ")
            );
            reduced_errors.push(reduced_comparison.max_parameter_error);
        }
    }
    assert_eq!(routed_errors.len(), 46);
    let routed_max = routed_errors.iter().copied().fold(0.0, f64::max);
    let reduced_max = reduced_errors.iter().copied().fold(0.0, f64::max);
    let full_close = routed_errors.iter().filter(|&&err| err <= 5e-3).count();
    assert_eq!(full_close, 45);
    routed_errors.sort_by(f64::total_cmp);
    reduced_errors.sort_by(f64::total_cmp);
    error_cases.sort_by(|a, b| b.0.total_cmp(&a.0));
    let full_median = routed_errors[routed_errors.len() / 2];
    let reduced_median = reduced_errors[reduced_errors.len() / 2];
    assert!(
        reduced_median / full_median >= 10.0,
        "full likelihood should improve median endpoint parity by at least 10x: reduced={reduced_median}, full={full_median}"
    );
    assert!(
        reduced_max / routed_max >= 10.0,
        "full DESeq2 objective should improve worst-case endpoint parity by at least 10x: reduced={reduced_max}, full={routed_max}"
    );
    eprintln!(
        "real DESeq2 1.52 baseline: routed={}; saved-vs-wrapper start max delta={:.6}; full NB likelihood + coefficient prior max/median={:.6}/{:.9}; likelihood-only max/median={:.6}/{:.9}; worst-case improvement={:.1}x",
        routed_errors.len(),
        max_initial_delta,
        routed_max,
        full_median,
        reduced_max,
        reduced_median,
        reduced_max / routed_max
    );
    eprintln!(
        "largest full-likelihood deviations: {:?}",
        &error_cases[..5]
    );
}

fn reduced_nb_nll(beta: &[f64], y: &[f64], x: &[Value], sf: &[f64], r: f64) -> f64 {
    let mut value = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        let mut linear = 0.0;
        for j in 0..beta.len() {
            linear += x[i][j].as_f64().unwrap() * beta[j];
        }
        let mu = (std::f64::consts::LN_2 * linear + sf[i].ln()).exp();
        value += (yi + r) * (r + mu).ln();
        value -= r * r.ln();
        value -= yi * mu.ln();
    }
    value
}

fn nums(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[allow(clippy::excessive_precision)] // Preserve the frozen callback's coefficients.
fn log_gamma(z: f64) -> f64 {
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}
