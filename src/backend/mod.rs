mod native;

use crate::error::OptimError;
use crate::result::OptimCounts;

pub(crate) use native::NativeBackend;

pub(crate) trait BackendProblem {
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError>;

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError>;

    fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        let value = self.value(x)?;
        let gradient = self.gradient(x)?;
        Ok((value, gradient))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BackendControl {
    pub(crate) maxit: usize,
    pub(crate) factr: f64,
    pub(crate) pgtol: f64,
    pub(crate) lmm: usize,
    pub(crate) trace: usize,
    pub(crate) report: usize,
    pub(crate) has_user_gradient: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct BackendResult {
    pub(crate) x: Vec<f64>,
    pub(crate) value: f64,
    pub(crate) counts: OptimCounts,
    pub(crate) convergence: i32,
    pub(crate) message: String,
}

pub(crate) trait LbfgsbBackend {
    fn minimize<P>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
    ) -> Result<BackendResult, OptimError>
    where
        P: BackendProblem;
}
