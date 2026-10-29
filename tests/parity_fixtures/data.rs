use super::support::fixtures::deserialize_f64_vec;
use super::support::fixtures::{
    RecordedControl as FixtureControl, RecordedCounts as FixtureCounts,
    RecordedResult as FixtureResult,
};
use super::*;

#[derive(Debug, Deserialize)]
pub(super) struct Fixture {
    pub(super) fixture: String,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) initial_par: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) lower: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) upper: Vec<f64>,
    pub(super) control: FixtureControl,
    pub(super) result: FixtureResult,
}

#[derive(Debug, Deserialize)]
pub(super) struct TraceFixture {
    pub(super) fixture: String,
    pub(super) source_fixture: String,
    pub(super) trace_kind: String,
    pub(super) first_points: Vec<Vec<f64>>,
    pub(super) result: TraceResult,
}

#[derive(Debug, Deserialize)]
pub(super) struct TraceResult {
    pub(super) counts: FixtureCounts,
    pub(super) convergence: i32,
    pub(super) message: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct ErrorFixture {
    pub(super) fixture: String,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) initial_par: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) lower: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    pub(super) upper: Vec<f64>,
    pub(super) error: FixtureError,
}

#[derive(Debug, Deserialize)]
pub(super) struct FixtureError {
    pub(super) message: String,
}

pub(super) fn control_from_fixture(control: &FixtureControl) -> OptimControl {
    control.to_control(0, 10)
}

pub(super) fn assert_fixture_close(
    fixture: &Fixture,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
) {
    let failures = fixture_failures(fixture, result, par_tolerance, value_tolerance);
    assert!(
        failures.is_empty(),
        "{}: {}",
        fixture.fixture,
        failures.join("; ")
    );
}

pub(super) fn fixture_failures(
    fixture: &Fixture,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
) -> Vec<String> {
    let mut differences = support::MaxAbsoluteDifferences::default();
    differences.record(result, &expected_result(fixture));
    differences.report(&format!("standard/{}", fixture.fixture));
    support::compare(
        result,
        &expected_result(fixture),
        support::Contract::Numerical {
            parameters: par_tolerance,
            value: value_tolerance,
        },
    )
}

pub(super) fn expected_result(fixture: &Fixture) -> OptimResult {
    fixture.result.to_result()
}

pub(super) fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }
}
