use crate::{ArithmeticProfile, OptimCounts};

/// Typed optimizer termination, rendered into the existing R-style result fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Projected gradient meets the requested tolerance.
    ProjectedGradient,
    /// Signed relative reduction meets factr times machine epsilon.
    RelativeReduction,
    /// The iteration budget was exhausted.
    IterationLimit,
    /// No line-search iterate could be accepted, or the target is unrepresentable.
    AbnormalLineSearch,
    /// Neither the model nor projected gradient provides a descent direction.
    NoDescentDirection,
    /// The parameter vector is empty.
    NothingToDo,
}
impl TerminationReason {
    /// R-style public convergence code.
    pub const fn convergence(self) -> i32 {
        match self {
            Self::IterationLimit => 1,
            Self::AbnormalLineSearch | Self::NoDescentDirection => 52,
            _ => 0,
        }
    }
    /// R-style public termination message.
    pub const fn message(self) -> &'static str {
        match self {
            Self::ProjectedGradient => "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
            Self::RelativeReduction => "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH",
            Self::IterationLimit => "NEW_X",
            Self::AbnormalLineSearch => "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH",
            Self::NoDescentDirection => "ERROR: NO FEASIBLE DESCENT DIRECTION",
            Self::NothingToDo => "NOTHING TO DO",
        }
    }
}

/// Numerical operation responsible for a model reset or fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryReason {
    /// Cauchy model construction failed.
    CauchyModel,
    /// Reduced-gradient construction failed.
    ReducedGradient,
    /// The free-subspace factorization failed.
    SubspaceFactorization,
    /// The free-subspace solve failed.
    SubspaceSolve,
    /// A line search failed with correction history available.
    LineSearch,
    /// The correction-history factorization failed.
    HistoryFactorization,
}

/// Outcome of the compact correction-history update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryUpdate {
    /// A correction was stored, possibly replacing the oldest column.
    Stored,
    /// Curvature was too small for a correction.
    SkippedCurvature,
    /// Factorization failed and history was cleared.
    ResetFactorization,
}

/// Condition that accepted a line-search trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepAcceptance {
    /// Sufficient decrease and strong-Wolfe curvature both held.
    StrongWolfe,
    /// A descending trial reached the feasible/algorithmic step cap.
    StepLimit,
    /// A sufficiently decreasing trial was accepted at a narrow bracket warning.
    BracketWarning,
    /// Armijo sufficient decrease in a historical comparison probe.
    Armijo,
}

/// Reason a line-search attempt ended without an accepted iterate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchFailure {
    /// The initial multiplier was nonpositive or non-finite.
    InvalidInitialStep,
    /// The initial directional derivative did not describe descent.
    NonDescentDirection,
    /// An updated multiplier was nonpositive or non-finite.
    InvalidUpdatedStep,
    /// The search consumed its trial budget without acceptance.
    TrialLimit,
    /// A compact target rounded back to the current point.
    UnrepresentableTarget,
}

/// Metrics from the compact model that actually produced a proposed target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSnapshot {
    /// Coordinates active at the model target.
    pub active: usize,
    /// Coordinates free in the model subspace.
    pub free: usize,
    /// Stored corrections used by the model.
    pub history: usize,
}

/// Structured observations from one synchronous optimization run.
///
/// Floating-point diagnostics use internal scaled coordinates/objective units.
/// Counts are completed optimizer-level evaluations, excluding finite-difference
/// callback calls. Events do not contain borrowed workspaces or parameter copies.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum OptimEvent {
    /// The initial value and gradient were evaluated successfully.
    Started {
        /// Selected reference arithmetic.
        profile: ArithmeticProfile,
        /// Internal objective value.
        value: f64,
        /// Projected gradient infinity norm.
        projected_gradient: f64,
        /// Completed optimizer evaluations.
        counts: OptimCounts,
    },
    /// A final search direction was selected.
    Direction {
        /// One-based attempted iteration.
        iteration: usize,
        /// Actual compact model metrics, absent for a projected fallback.
        model: Option<ModelSnapshot>,
        /// Infinity norm of the chosen direction.
        direction_norm: f64,
    },
    /// A search attempt failed, before deciding whether to retry with empty history.
    LineSearchFailed {
        /// One-based attempted iteration.
        iteration: usize,
        /// Specific reason for this failed attempt.
        reason: LineSearchFailure,
        /// Completed evaluations in this attempt, excluding earlier attempts.
        evaluations: usize,
        /// Completed optimizer evaluations, including the failed attempt.
        counts: OptimCounts,
    },
    /// A model or line-search failure cleared correction history.
    MemoryReset {
        /// One-based attempted iteration.
        iteration: usize,
        /// Operation responsible for recovery.
        reason: RecoveryReason,
    },
    /// Model construction still failed after a reset; a projected fallback is used.
    ModelFallback {
        /// One-based attempted iteration.
        iteration: usize,
        /// Operation that failed after retry.
        reason: RecoveryReason,
    },
    /// A line-search iterate was accepted, before updating correction history.
    Accepted {
        /// One-based attempted iteration.
        iteration: usize,
        /// Internal accepted objective value.
        value: f64,
        /// Projected gradient infinity norm at the accepted point.
        projected_gradient: f64,
        /// Accepted line-search multiplier.
        alpha: f64,
        /// Feasible/algorithmic search cap used for this search.
        max_alpha: f64,
        /// Euclidean displacement norm.
        step_norm: f64,
        /// Number of evaluations performed by the accepted search attempt.
        search_evaluations: usize,
        /// Exact branch that accepted the trial.
        acceptance: StepAcceptance,
        /// Directional derivative at the beginning of the search.
        initial_derivative: f64,
        /// Directional derivative at the accepted trial.
        final_derivative: f64,
        /// Whether the curvature condition held (warning accepts may relax it).
        wolfe_curvature: bool,
        /// Completed optimizer evaluations, including earlier failed attempts.
        counts: OptimCounts,
    },
    /// The accepted step's correction history was processed.
    History {
        /// One-based accepted iteration.
        iteration: usize,
        /// Stored, skipped, or reset outcome.
        outcome: HistoryUpdate,
        /// Number of corrections after the update.
        history: usize,
    },
    /// The wrapper successfully constructed a public result.
    Finished {
        /// Number of completed accepted iterations.
        iterations: usize,
        /// Typed reason for the public convergence/message fields.
        reason: TerminationReason,
        /// Final optimizer-level counts.
        counts: OptimCounts,
    },
    /// A final conversion failed; the API returns the detailed OptimError.
    OutputConversionFailed {
        /// Completed optimizer-level evaluations.
        counts: OptimCounts,
    },
    /// An evaluation failed; the API returns the detailed OptimError.
    EvaluationFailed {
        /// Evaluations completed before the failure.
        counts: OptimCounts,
    },
}

/// Receive read-only diagnostic events. Panics propagate like callback panics.
/// Observers run synchronously and may perform independent nested optimizations.
pub trait OptimObserver {
    /// Whether this observer needs events; false removes diagnostic work at compile time.
    const ENABLED: bool = true;
    /// Consume an event from the current run.
    fn observe(&mut self, event: OptimEvent);
}
impl<F: FnMut(OptimEvent)> OptimObserver for F {
    fn observe(&mut self, event: OptimEvent) {
        self(event)
    }
}
pub(crate) struct NoObserver;
impl OptimObserver for NoObserver {
    const ENABLED: bool = false;
    fn observe(&mut self, _event: OptimEvent) {}
}
