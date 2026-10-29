use super::*;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct TraceFixture {
    pub(super) fixture: String,
    pub(super) source_fixture: String,
    pub(super) trace_kind: String,
    pub(super) first_points: Vec<Vec<f64>>,
}

pub(super) fn assert_mirrored_steps(right: Option<f64>, left: Option<f64>) {
    let right = right.expect("right-hand step");
    let left = left.expect("left-hand mirrored step");
    assert!(
        (right + left).abs() <= 1e-14,
        "right={right:?}, left={left:?}"
    );
}

pub(super) fn dense_two_by_two_product(matrix: [[f64; 2]; 2]) -> impl FnMut(&[f64]) -> Vec<f64> {
    move |vector| {
        vec![
            matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
            matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
        ]
    }
}

pub(super) struct RosenbrockProblem;

impl BackendProblem for RosenbrockProblem {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ])
    }
}

#[derive(Default)]
pub(super) struct RecordingRosenbrockProblem {
    pub(super) points: Vec<Vec<f64>>,
}

impl BackendProblem for RecordingRosenbrockProblem {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ])
    }

    fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        self.points.push(x.to_vec());
        Ok((self.value(x)?, self.gradient(x)?))
    }
}

pub(super) struct OneDimensionalQuadratic;

impl BackendProblem for OneDimensionalQuadratic {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok((x[0] - 2.0).powi(2))
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![2.0 * (x[0] - 2.0)])
    }
}

pub(super) struct LinearDescent;

impl BackendProblem for LinearDescent {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        Ok(-x[0])
    }

    fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![-1.0])
    }
}

pub(super) struct AlwaysNonFiniteObjective;

impl BackendProblem for AlwaysNonFiniteObjective {
    fn value(&mut self, _x: &[f64]) -> Result<f64, OptimError> {
        Err(OptimError::NonFiniteObjective { value: f64::NAN })
    }

    fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
        Ok(vec![0.0])
    }
}
