//! Shared recorded controls/results and strict numeric decoding.
use rcompat_lbfgsb::{OptimControl, OptimCounts, OptimResult};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct RecordedControl {
    pub maxit: usize,
    pub fnscale: f64,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub parscale: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub ndeps: Vec<f64>,
    pub factr: f64,
    pub pgtol: f64,
    pub lmm: usize,
}
impl RecordedControl {
    pub fn to_control(&self, trace: usize, report: usize) -> OptimControl {
        OptimControl {
            maxit: self.maxit,
            fnscale: self.fnscale,
            parscale: self.parscale.clone(),
            ndeps: self.ndeps.clone(),
            factr: self.factr,
            pgtol: self.pgtol,
            lmm: self.lmm,
            trace,
            report,
        }
    }
}
#[derive(Debug, Deserialize)]
pub struct RecordedCounts {
    pub function: usize,
    pub gradient: usize,
}
#[derive(Debug, Deserialize)]
pub struct RecordedResult {
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub par: Vec<f64>,
    pub value: f64,
    pub counts: RecordedCounts,
    pub convergence: i32,
    pub message: String,
}
impl RecordedResult {
    pub fn to_result(&self) -> OptimResult {
        OptimResult {
            par: self.par.clone(),
            value: self.value,
            counts: OptimCounts {
                function: self.counts.function,
                gradient: self.counts.gradient,
            },
            convergence: self.convergence,
            message: self.message.clone(),
        }
    }
}
#[derive(Deserialize)]
#[serde(untagged)]
enum JsonF64 {
    Number(f64),
    String(String),
}
fn number(value: JsonF64) -> Result<f64, String> {
    match value {
        JsonF64::Number(value) => Ok(value),
        JsonF64::String(value) if value == "Inf" => Ok(f64::INFINITY),
        JsonF64::String(value) if value == "-Inf" => Ok(f64::NEG_INFINITY),
        JsonF64::String(value) => Err(format!("unsupported numeric string {value:?}")),
    }
}
pub fn deserialize_f64_vec<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<f64>, D::Error> {
    Vec::<JsonF64>::deserialize(deserializer)?
        .into_iter()
        .map(|value| number(value).map_err(serde::de::Error::custom))
        .collect()
}
pub fn numbers(value: &Value) -> Vec<f64> {
    let values: Vec<JsonF64> = serde_json::from_value(value.clone()).expect("numeric array");
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            number(value).unwrap_or_else(|error| panic!("element {index}: {error}"))
        })
        .collect()
}
