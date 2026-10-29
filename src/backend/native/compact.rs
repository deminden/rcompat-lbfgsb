//! Retained compact state and direction selection.
use super::*;

mod algebra;
use algebra::*;
mod cauchy;
#[cfg(test)]
mod exact_tests;
#[cfg(test)]
mod factorization_tests;
mod history;
#[cfg(test)]
mod model_tests;
mod subspace;

/// Stateful compact L-BFGS-B model for all production bound and gradient routes.
///
/// Retains the cyclic S/Y columns and the cached
/// SY/SS/WT/WN/WN1 workspaces across outer iterations.  Its operation order is
/// the one documented by the permissive language-neutral subroutine specs.
#[derive(Debug)]
pub(super) struct ExactCompactState {
    profile: ArithmeticProfile,
    n: usize,
    memory: usize,
    ws: Vec<Vec<f64>>,
    wy: Vec<Vec<f64>>,
    sy: Vec<Vec<f64>>,
    ss: Vec<Vec<f64>>,
    wt: Vec<Vec<f64>>,
    wn: Vec<Vec<f64>>,
    wn1: Vec<Vec<f64>>,
    head: usize,
    itail: usize,
    col: usize,
    iupdat: usize,
    theta: f64,
    updatd: bool,
    iwhere: Vec<i32>,
    index: Vec<usize>,
    indx2: Vec<usize>,
    nfree_previous: usize,
    iteration: usize,
}

#[derive(Debug)]
struct ExactCauchyPoint {
    x: Vec<f64>,
    c: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct FreePartition {
    nfree: usize,
    nenter: usize,
    ileave: usize,
    work: bool,
}

impl ExactCompactState {
    pub(super) fn new(
        profile: ArithmeticProfile,
        dimension: usize,
        memory: usize,
        lower: &[f64],
        upper: &[f64],
    ) -> Self {
        let memory = memory.max(1);
        let mut iwhere = vec![0; dimension];
        for index in 0..dimension {
            iwhere[index] = match coherent_bound_type(lower[index], upper[index]) {
                0 => -1,
                2 if upper[index] - lower[index] <= 0.0 => 3,
                _ => 0,
            };
        }
        Self {
            profile,
            n: dimension,
            memory,
            ws: vec![vec![0.0; dimension]; memory],
            wy: vec![vec![0.0; dimension]; memory],
            sy: vec![vec![0.0; memory]; memory],
            ss: vec![vec![0.0; memory]; memory],
            wt: vec![vec![0.0; memory]; memory],
            wn: vec![vec![0.0; 2 * memory]; 2 * memory],
            wn1: vec![vec![0.0; 2 * memory]; 2 * memory],
            head: 0,
            itail: 0,
            col: 0,
            iupdat: 0,
            theta: 1.0,
            updatd: false,
            iwhere,
            index: (0..dimension).collect(),
            indx2: vec![0; dimension],
            nfree_previous: 0,
            iteration: 0,
        }
    }

    pub(super) fn clear_history(&mut self) {
        self.head = 0;
        self.itail = 0;
        self.col = 0;
        self.iupdat = 0;
        self.theta = 1.0;
        self.updatd = false;
        for matrix in [
            &mut self.sy,
            &mut self.ss,
            &mut self.wt,
            &mut self.wn,
            &mut self.wn1,
        ] {
            for row in matrix {
                row.fill(0.0);
            }
        }
    }

    pub(super) fn snapshot(&self) -> crate::ModelSnapshot {
        crate::ModelSnapshot {
            active: self.n - self.nfree_previous,
            free: self.nfree_previous,
            history: self.col,
        }
    }

    pub(super) fn has_history(&self) -> bool {
        self.col > 0
    }

    pub(super) fn direction(
        &mut self,
        x: &[f64],
        gradient: &[f64],
        lower: &[f64],
        upper: &[f64],
        projected_gradient_norm: f64,
    ) -> Result<DirectionChoice, RecoveryReason> {
        debug_assert_eq!(x.len(), self.n);
        // With no constraints and a retained model, solve directly in the
        // full subspace. Computing and cancelling an unnecessary Cauchy step
        // changes rounding and can amplify drift on ill-conditioned models.
        let cauchy = if self.col > 0 && lower.iter().chain(upper).all(|bound| bound.is_infinite()) {
            ExactCauchyPoint {
                x: x.to_vec(),
                c: vec![0.0; 2 * self.col],
            }
        } else {
            self.cauchy(x, gradient, lower, upper, projected_gradient_norm)
                .ok_or(RecoveryReason::CauchyModel)?
        };
        let partition = self.partition_free_variables();
        let reduced = self
            .reduced_gradient(x, gradient, &cauchy.x, &cauchy.c, partition.nfree)
            .ok_or(RecoveryReason::ReducedGradient)?;

        let mut target = cauchy.x;
        if partition.nfree > 0 && self.col > 0 {
            if partition.work && !self.formk(partition) {
                self.clear_history();
                return Err(RecoveryReason::SubspaceFactorization);
            }
            target = self
                .subspace_minimizer(lower, upper, target, reduced, partition.nfree)
                .ok_or(RecoveryReason::SubspaceSolve)?;
        }

        self.nfree_previous = partition.nfree;
        self.iteration += 1;
        let direction = difference(&target, x);
        Ok(DirectionChoice {
            direction,
            unit_step_target: Some(target),
        })
    }
}
