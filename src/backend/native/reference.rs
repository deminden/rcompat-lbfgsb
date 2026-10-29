//! Independent comparison models used only by backend tests.
use super::*;

mod memory;
pub(super) use memory::*;
mod cauchy;
pub(super) use cauchy::*;
mod subspace;
pub(super) use subspace::*;

pub(super) const CURVATURE_EPS: f64 = 1e-12;
pub(super) const HISTORY_CURVATURE_EPS: f64 = f64::EPSILON;
pub(super) const MIN_STEP: f64 = 1e-14;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundActivity {
    Tolerant,
    Exact,
}

#[derive(Debug, Clone)]
pub(super) struct Correction {
    pub(super) s: Vec<f64>,
    pub(super) y: Vec<f64>,
    pub(super) rho: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct CauchyPoint {
    pub(super) x: Vec<f64>,
    pub(super) active_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct SubspacePoint {
    pub(super) x: Vec<f64>,
    pub(super) free_count: usize,
    pub(super) clipped: bool,
}

#[derive(Debug, Clone)]
pub(super) struct PreparedHessianUpdate {
    pub(super) correction_index: usize,
    pub(super) bs: Vec<f64>,
    pub(super) sbs: f64,
}

pub(super) fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }
}
