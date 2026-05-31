use crate::error::OptimError;

/// R-like controls for L-BFGS-B optimization.
///
/// The field names intentionally mirror R's `optim` controls. Use
/// [`OptimControl::default_for_dimension`] to create defaults because
/// `parscale` and `ndeps` are dimension-dependent.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimControl {
    /// Maximum number of backend iterations.
    pub maxit: usize,
    /// Function scaling. Internal objective values are divided by this value.
    pub fnscale: f64,
    /// Parameter scaling vector.
    pub parscale: Vec<f64>,
    /// Finite-difference step sizes in scaled parameter space.
    pub ndeps: Vec<f64>,
    /// Relative reduction factor used by L-BFGS-B-style convergence tests.
    pub factr: f64,
    /// Projected gradient tolerance.
    pub pgtol: f64,
    /// Limited-memory history size.
    pub lmm: usize,
    /// Trace verbosity level. Exact R trace output is not implemented yet, but
    /// backend diagnostics include line-search and model-step details.
    pub trace: usize,
    /// Trace reporting interval.
    pub report: usize,
}

impl OptimControl {
    /// Returns R-like default controls for a parameter vector of length `n`.
    pub fn default_for_dimension(n: usize) -> Self {
        Self {
            maxit: 100,
            fnscale: 1.0,
            parscale: vec![1.0; n],
            ndeps: vec![1e-3; n],
            factr: 1e7,
            pgtol: 0.0,
            lmm: 5,
            trace: 0,
            report: 10,
        }
    }

    /// Validates controls against the optimization dimension.
    pub fn validate_for_dimension(&self, n: usize) -> Result<(), OptimError> {
        validate_len("parscale", n, self.parscale.len())?;
        validate_len("ndeps", n, self.ndeps.len())?;

        if !self.fnscale.is_finite() || self.fnscale == 0.0 {
            return Err(OptimError::InvalidControl {
                field: "fnscale",
                reason: "must be finite and nonzero".to_string(),
            });
        }
        if !self.factr.is_finite() || self.factr < 0.0 {
            return Err(OptimError::InvalidControl {
                field: "factr",
                reason: "must be finite and nonnegative".to_string(),
            });
        }
        if !self.pgtol.is_finite() || self.pgtol < 0.0 {
            return Err(OptimError::InvalidControl {
                field: "pgtol",
                reason: "must be finite and nonnegative".to_string(),
            });
        }
        if self.lmm == 0 {
            return Err(OptimError::InvalidControl {
                field: "lmm",
                reason: "must be greater than zero".to_string(),
            });
        }
        for (index, value) in self.parscale.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(OptimError::InvalidControl {
                    field: "parscale",
                    reason: format!("entry {index} must be finite and positive"),
                });
            }
        }
        for (index, value) in self.ndeps.iter().copied().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(OptimError::InvalidControl {
                    field: "ndeps",
                    reason: format!("entry {index} must be finite and positive"),
                });
            }
        }
        Ok(())
    }
}

impl Default for OptimControl {
    fn default() -> Self {
        Self::default_for_dimension(0)
    }
}

fn validate_len(name: &'static str, expected: usize, actual: usize) -> Result<(), OptimError> {
    if actual == expected {
        Ok(())
    } else {
        Err(OptimError::DimensionMismatch {
            name,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_dimensioned() {
        let control = OptimControl::default_for_dimension(3);
        assert_eq!(control.maxit, 100);
        assert_eq!(control.parscale, vec![1.0; 3]);
        assert_eq!(control.ndeps, vec![1e-3; 3]);
        assert!(control.validate_for_dimension(3).is_ok());
    }

    #[test]
    fn zero_dimension_defaults_are_valid() {
        let control = OptimControl::default_for_dimension(0);
        assert!(control.parscale.is_empty());
        assert!(control.ndeps.is_empty());
        assert!(control.validate_for_dimension(0).is_ok());
    }

    #[test]
    fn rejects_invalid_scaling() {
        let mut control = OptimControl::default_for_dimension(1);
        control.parscale[0] = 0.0;
        assert!(matches!(
            control.validate_for_dimension(1),
            Err(OptimError::InvalidControl {
                field: "parscale",
                ..
            })
        ));
    }
}
