#![allow(missing_docs)]

use rcompat_lbfgsb::{
    optim_lbfgsb, optim_lbfgsb_with_gradient, Bounds, OptimControl, OptimError, OptimResult,
};
use serde::{Deserialize, Deserializer};

#[derive(Debug, Deserialize)]
struct Fixture {
    fixture: String,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    initial_par: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    lower: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    upper: Vec<f64>,
    control: FixtureControl,
    result: FixtureResult,
}

#[derive(Debug, Deserialize)]
struct FixtureControl {
    maxit: usize,
    fnscale: f64,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    parscale: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    ndeps: Vec<f64>,
    factr: f64,
    pgtol: f64,
    lmm: usize,
}

#[derive(Debug, Deserialize)]
struct FixtureResult {
    #[serde(deserialize_with = "deserialize_f64_vec")]
    par: Vec<f64>,
    value: f64,
    counts: FixtureCounts,
    convergence: i32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct FixtureCounts {
    function: usize,
    gradient: usize,
}

#[derive(Debug, Deserialize)]
struct ErrorFixture {
    fixture: String,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    initial_par: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    lower: Vec<f64>,
    #[serde(deserialize_with = "deserialize_f64_vec")]
    upper: Vec<f64>,
    error: FixtureError,
}

#[derive(Debug, Deserialize)]
struct FixtureError {
    message: String,
}

#[test]
fn fixtures_that_are_currently_at_float_noise_match_r() {
    for fixture_json in [
        include_str!("../fixtures/r_optim_lbfgsb/active_bounds.json"),
        include_str!("../fixtures/r_optim_lbfgsb/active_upper_bound.json"),
        include_str!("../fixtures/r_optim_lbfgsb/factr_zero_quadratic.json"),
        include_str!("../fixtures/r_optim_lbfgsb/fixed_parameter.json"),
        include_str!("../fixtures/r_optim_lbfgsb/finite_difference.json"),
        include_str!("../fixtures/r_optim_lbfgsb/fnscale.json"),
        include_str!("../fixtures/r_optim_lbfgsb/initial_projected_gradient.json"),
        include_str!("../fixtures/r_optim_lbfgsb/initial_outside_bounds.json"),
        include_str!("../fixtures/r_optim_lbfgsb/initial_pos_inf_projected.json"),
        include_str!("../fixtures/r_optim_lbfgsb/initial_neg_inf_projected.json"),
        include_str!("../fixtures/r_optim_lbfgsb/mixed_bounds_quadratic.json"),
        include_str!("../fixtures/r_optim_lbfgsb/near_lower_finite_difference.json"),
        include_str!("../fixtures/r_optim_lbfgsb/near_upper_finite_difference.json"),
        include_str!("../fixtures/r_optim_lbfgsb/negative_fnscale_gradient.json"),
        include_str!("../fixtures/r_optim_lbfgsb/ndeps_vector.json"),
        include_str!("../fixtures/r_optim_lbfgsb/parscale.json"),
        include_str!("../fixtures/r_optim_lbfgsb/parscale_finite_difference.json"),
        include_str!("../fixtures/r_optim_lbfgsb/pgtol_initial.json"),
        include_str!("../fixtures/r_optim_lbfgsb/quadratic.json"),
        include_str!("../fixtures/r_optim_lbfgsb/unbounded_quadratic.json"),
        include_str!("../fixtures/r_optim_lbfgsb/zero_dim.json"),
        include_str!("../fixtures/r_optim_lbfgsb/zero_dim_with_gradient.json"),
    ] {
        let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
        let result = run_fixture(&fixture);
        assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
    }
}

#[test]
fn rosenbrock_fixture_matches_r() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("../fixtures/r_optim_lbfgsb/rosenbrock.json")).unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-7, 1e-12);
}

#[test]
fn maxit_fixture_matches_r() {
    for fixture_json in [
        include_str!("../fixtures/r_optim_lbfgsb/maxit.json"),
        include_str!("../fixtures/r_optim_lbfgsb/maxit_zero.json"),
    ] {
        let fixture: Fixture = serde_json::from_str(fixture_json).unwrap();
        let result = run_fixture(&fixture);
        assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
    }
}

#[test]
fn two_dim_parscale_gradient_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/r_optim_lbfgsb/two_dim_parscale_gradient.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-8, 1e-10);
}

#[test]
fn loose_factr_rosenbrock_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/r_optim_lbfgsb/factr_loose_rosenbrock.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-6, 1e-8);
}

#[test]
fn lmm_one_rosenbrock_fixture_matches_r() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../fixtures/r_optim_lbfgsb/lmm_one_rosenbrock.json"
    ))
    .unwrap();
    let result = run_fixture(&fixture);
    assert_fixture_close(&fixture, &result, 1e-6, 1e-8);
}

#[test]
fn fixed_parameter_without_gradient_matches_r_error_class() {
    let fixture: ErrorFixture = serde_json::from_str(include_str!(
        "../fixtures/r_optim_lbfgsb/fixed_no_gradient_error.json"
    ))
    .unwrap();
    assert_eq!(fixture.fixture, "fixed_no_gradient_error");
    assert_eq!(
        fixture.error.message,
        "non-finite finite-difference value [1]"
    );

    let error = optim_lbfgsb(
        fixture.initial_par,
        Bounds::new(fixture.lower, fixture.upper).unwrap(),
        |p| (p[0] - 10.0).powi(2),
        OptimControl::default_for_dimension(1),
    )
    .unwrap_err();

    assert!(matches!(error, OptimError::InvalidBounds { .. }));
}

fn run_fixture(fixture: &Fixture) -> OptimResult {
    let control = control_from_fixture(&fixture.control);
    let bounds = Bounds::new(fixture.lower.clone(), fixture.upper.clone()).unwrap();

    match fixture.fixture.as_str() {
        "active_bounds" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "active_upper_bound" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 2.0).powi(2),
            control,
        )
        .unwrap(),
        "fixed_parameter" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 10.0).powi(2),
            |p| vec![2.0 * (p[0] - 10.0)],
            control,
        )
        .unwrap(),
        "finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 1.0).powi(2),
            control,
        )
        .unwrap(),
        "fnscale" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| -(p[0] - 3.0).powi(2),
            control,
        )
        .unwrap(),
        "negative_fnscale_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| -(p[0] - 3.0).powi(2),
            |p| vec![-2.0 * (p[0] - 3.0)],
            control,
        )
        .unwrap(),
        "initial_projected_gradient" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "initial_outside_bounds" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| p[0] * p[0],
            control,
        )
        .unwrap(),
        "initial_pos_inf_projected" | "initial_neg_inf_projected" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| p[0] * p[0],
            |p| vec![2.0 * p[0]],
            control,
        )
        .unwrap(),
        "mixed_bounds_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 1.0), 2.0 * (p[1] + 2.0)],
            control,
        )
        .unwrap(),
        "ndeps_vector" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.0).powi(2) + (p[1] + 2.0).powi(2),
            control,
        )
        .unwrap(),
        "near_upper_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 0.25).powi(2),
            control,
        )
        .unwrap(),
        "near_lower_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] + 0.25).powi(2),
            control,
        )
        .unwrap(),
        "parscale" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2),
            |p| vec![2.0 * (p[0] - 4.0)],
            control,
        )
        .unwrap(),
        "parscale_finite_difference" => optim_lbfgsb(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2),
            control,
        )
        .unwrap(),
        "pgtol_initial" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 2.0).powi(2),
            |p| vec![2.0 * (p[0] - 2.0)],
            control,
        )
        .unwrap(),
        "quadratic" | "maxit" | "maxit_zero" | "factr_zero_quadratic" => {
            optim_lbfgsb_with_gradient(
                fixture.initial_par.clone(),
                bounds,
                |p| (p[0] - 2.0).powi(2),
                |p| vec![2.0 * (p[0] - 2.0)],
                control,
            )
            .unwrap()
        }
        "zero_dim" => {
            optim_lbfgsb(fixture.initial_par.clone(), bounds, |_| 123.0, control).unwrap()
        }
        "zero_dim_with_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |_| 123.0,
            |_| panic!("zero-dimensional gradient should not be called"),
            control,
        )
        .unwrap(),
        "two_dim_parscale_gradient" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 4.0).powi(2) + (p[1] + 1.0).powi(2),
            |p| vec![2.0 * (p[0] - 4.0), 2.0 * (p[1] + 1.0)],
            control,
        )
        .unwrap(),
        "unbounded_quadratic" => optim_lbfgsb_with_gradient(
            fixture.initial_par.clone(),
            bounds,
            |p| (p[0] - 1.25).powi(2),
            |p| vec![2.0 * (p[0] - 1.25)],
            control,
        )
        .unwrap(),
        "rosenbrock" | "factr_loose_rosenbrock" | "lmm_one_rosenbrock" => {
            optim_lbfgsb_with_gradient(
                fixture.initial_par.clone(),
                bounds,
                |p| 100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2),
                |p| {
                    vec![
                        -400.0 * p[0] * (p[1] - p[0] * p[0]) - 2.0 * (1.0 - p[0]),
                        200.0 * (p[1] - p[0] * p[0]),
                    ]
                },
                control,
            )
            .unwrap()
        }
        other => panic!("unknown fixture {other}"),
    }
}

fn control_from_fixture(control: &FixtureControl) -> OptimControl {
    OptimControl {
        maxit: control.maxit,
        fnscale: control.fnscale,
        parscale: control.parscale.clone(),
        ndeps: control.ndeps.clone(),
        factr: control.factr,
        pgtol: control.pgtol,
        lmm: control.lmm,
        trace: 0,
        report: 10,
    }
}

fn assert_fixture_close(
    fixture: &Fixture,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
) {
    assert_eq!(
        result.convergence, fixture.result.convergence,
        "{} convergence mismatch: {}",
        fixture.fixture, result.message
    );
    assert_eq!(
        result.message, fixture.result.message,
        "{} message mismatch",
        fixture.fixture
    );
    assert_eq!(
        result.counts.function, fixture.result.counts.function,
        "{} function-count mismatch",
        fixture.fixture
    );
    assert_eq!(
        result.counts.gradient, fixture.result.counts.gradient,
        "{} gradient-count mismatch",
        fixture.fixture
    );
    for (actual, expected) in result.par.iter().zip(fixture.result.par.iter()) {
        assert!(
            (actual - expected).abs() <= par_tolerance,
            "{} par mismatch: actual={actual:?}, expected={expected:?}, result={result:?}",
            fixture.fixture
        );
    }
    assert!(
        (result.value - fixture.result.value).abs() <= value_tolerance,
        "{} value mismatch: actual={:?}, expected={:?}, result={result:?}",
        fixture.fixture,
        result.value,
        fixture.result.value
    );
}

#[derive(Deserialize)]
#[serde(untagged)]
enum JsonF64 {
    Number(f64),
    String(String),
}

fn deserialize_f64_vec<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<JsonF64>::deserialize(deserializer)?;
    values
        .into_iter()
        .map(|value| match value {
            JsonF64::Number(value) => Ok(value),
            JsonF64::String(value) if value == "Inf" => Ok(f64::INFINITY),
            JsonF64::String(value) if value == "-Inf" => Ok(f64::NEG_INFINITY),
            JsonF64::String(value) => Err(serde::de::Error::custom(format!(
                "unsupported numeric string {value:?}"
            ))),
        })
        .collect()
}
