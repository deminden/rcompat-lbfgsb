//! Replay published DESeq2 fits with the independent full NB + prior callback.
#[path = "support/deseq2_objective.rs"]
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/deseq_airway/cases.json")
        });
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let fits = fixture["reference"]["results"]
        .as_array()
        .ok_or("expected a fixture with embedded DESeq2 reference results")?;
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
                Bounds::new(nums(&case["lower"]), nums(&case["upper"]))?,
                |beta| objective.value(beta),
                OptimControl::default_for_dimension(dimension),
            )?;
            let error = result
                .par
                .iter()
                .zip(nums(&fit["beta"]))
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
