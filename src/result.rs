/// R-like function and gradient evaluation counts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OptimCounts {
    /// Number of optimizer-level function evaluations reported in R style.
    pub function: usize,
    /// Number of optimizer-level gradient evaluations reported in R style.
    pub gradient: usize,
}

/// R-like optimization result.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimResult {
    /// Final parameters in user scale.
    pub par: Vec<f64>,
    /// Final objective value in user scale.
    pub value: f64,
    /// Function and gradient counts.
    pub counts: OptimCounts,
    /// R-like convergence code. `0` indicates success.
    pub convergence: i32,
    /// R-like convergence or failure message.
    pub message: String,
}

impl OptimResult {
    /// Returns `true` when `convergence == 0`.
    pub fn is_success(&self) -> bool {
        self.convergence == 0
    }

    /// Returns the R-like convergence code.
    pub fn convergence_status(&self) -> i32 {
        self.convergence
    }

    /// Returns the objective call count.
    pub fn function_count(&self) -> usize {
        self.counts.function
    }

    /// Returns the gradient call count.
    pub fn gradient_count(&self) -> usize {
        self.counts.gradient
    }
}
