#![allow(missing_docs)]

use serde_json::Value;
use std::collections::HashSet;

#[test]
fn published_airway_fallback_fixture_preserves_the_public_fit_contract() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/deseq_airway/cases.json")).unwrap();
    assert_eq!(fixture["source"]["accession"], "GSE52778");
    assert_eq!(fixture["source"]["package_version"], "1.32.0");
    let reference: Value = serde_json::from_str(include_str!(
        "../fixtures/native_r462/airway_reference.json"
    ))
    .unwrap();
    assert_eq!(reference["deseq2_version"], "1.52.0");
    assert!(reference["r_version"].as_str().unwrap().contains("4.6.2"));
    assert_eq!(reference["maxit"], 1);
    let fits = reference["results"].as_array().unwrap();
    assert_eq!(fits.len(), 28);

    let mut matched = 0;
    for contrast in fixture["contrasts"].as_array().unwrap() {
        let k = contrast["coefficients"].as_u64().unwrap() as usize;
        assert!(k == 2 || k == 5);
        assert_eq!(contrast["design"].as_array().unwrap().len(), 8);
        assert_eq!(contrast["size_factors"].as_array().unwrap().len(), 8);
        for row in contrast["design"].as_array().unwrap() {
            assert_eq!(row.as_array().unwrap().len(), k);
        }
        let mut profiles = HashSet::new();
        for case in contrast["cases"].as_array().unwrap() {
            let counts = case["counts"].as_array().unwrap();
            assert_eq!(counts.len(), 8);
            let profile: Vec<u64> = counts.iter().map(|y| y.as_u64().unwrap()).collect();
            assert!(profile.iter().sum::<u64>() >= 10);
            assert!(profiles.insert(profile), "duplicate numeric count profile");
            assert!(case["dispersion"].as_f64().unwrap() > 0.0);
            assert_eq!(case["actual_optim_routed"], true);
            let fit = fits
                .iter()
                .find(|fit| fit["fixture"] == case["fixture"])
                .unwrap();
            assert_eq!(fit["contrast"], contrast["contrast"]);
            assert_eq!(fit["optimizer_method"], "L-BFGS-B");
            assert_eq!(fit["optimizer_has_gradient"], false);
            assert!(fit["optimizer_control"].as_array().unwrap().is_empty());
            assert_eq!(fit["optimizer_start"], case["initial_par"]);
            assert_eq!(fit["beta"], fit["optimizer_result"]["par"]);
            assert_eq!(fit["beta"].as_array().unwrap().len(), k);
            for beta in fit["beta"].as_array().unwrap() {
                assert!((-30.0..=30.0).contains(&beta.as_f64().unwrap()));
            }
            assert_eq!(fit["optimizer_result"]["convergence"], 0);
            matched += 1;
        }
    }
    assert_eq!(matched, fits.len());
}
