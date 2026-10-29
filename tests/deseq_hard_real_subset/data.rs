use super::*;
#[derive(Debug, Deserialize)]
pub(super) struct HardFixture {
    pub(super) fixture: String,
    pub(super) contrasts: Vec<ContrastFixture>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ContrastFixture {
    pub(super) contrast: String,
    pub(super) samples: usize,
    pub(super) coefficients: usize,
    pub(super) coefficient_names: Vec<String>,
    pub(super) design: Vec<Vec<f64>>,
    pub(super) size_factors: Vec<f64>,
    pub(super) cases: Vec<HardCase>,
}

#[derive(Debug, Deserialize)]
pub(super) struct HardCase {
    pub(super) fixture: String,
    pub(super) dispersion: f64,
    pub(super) counts: Vec<f64>,
    pub(super) initial_par: Vec<f64>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) control: HardControl,
    pub(super) gradient_supplied: bool,
    pub(super) result: HardResult,
}

pub(super) fn hard_fixture() -> HardFixture {
    serde_json::from_str(include_str!(
        "../../fixtures/deseq_hard_real_subset/optim_cases.json"
    ))
    .unwrap()
}

pub(super) fn control_from_case(control: &HardControl) -> OptimControl {
    control.to_control(0, 10)
}
