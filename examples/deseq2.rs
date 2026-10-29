//! Replay published DESeq2 fits with the independent full NB + prior callback.
#[path = "support/deseq2_objective.rs"]
mod objective;

use objective::Deseq2Objective;
use rcompat_lbfgsb::{ArithmeticProfile, Bounds, OptimControl, optim_lbfgsb_with_profile};
use serde_json::Value;

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let (input_path, reference_path) = match arguments.as_slice() {
        [] => (
            root.join("fixtures/deseq_airway/cases.json"),
            root.join("fixtures/native_r462/airway_reference.json"),
        ),
        [inputs, reference] => (inputs.into(), reference.into()),
        _ => return Err("usage: deseq2 [INPUT_JSON REFERENCE_JSON]".into()),
    };
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(input_path)?)?;
    let reference: Value = serde_json::from_str(&std::fs::read_to_string(reference_path)?)?;
    let fits = reference["results"]
        .as_array()
        .ok_or("expected DESeq2 package reference results")?;
    for contrast in fixture["contrasts"].as_array().unwrap() {
        let design: Vec<_> = contrast["design"]
            .as_array()
            .unwrap()
            .iter()
            .map(nums)
            .collect();
        let sf = nums(&contrast["size_factors"]);
        for case in contrast["cases"].as_array().unwrap() {
            let fit = fits
                .iter()
                .find(|fit| fit["fixture"] == case["fixture"])
                .ok_or_else(|| format!("missing package reference for {}", case["fixture"]))?;
            if fit["optimizer_start"].is_null() {
                continue;
            }
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
                Bounds::new(nums(&case["lower"]), nums(&case["upper"]))?,
                |beta| objective.value(beta),
                OptimControl::default_for_dimension(dimension),
                ArithmeticProfile::AppleSiliconOpenBlas,
            )?;
            let expected = nums(&fit["beta"]);
            if result.par.len() != expected.len() {
                return Err(format!("coefficient count mismatch for {}", case["fixture"]).into());
            }
            let error = result
                .par
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            println!(
                "{} max coefficient error={error:.6e} convergence={}",
                case["fixture"].as_str().unwrap(),
                result.convergence
            );
        }
    }
    Ok(())
}
