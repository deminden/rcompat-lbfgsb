use crate::error::OptimError;

/// Lower and upper bounds for each parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    /// Lower bounds. Use `f64::NEG_INFINITY` for an unbounded lower side.
    pub lower: Vec<f64>,
    /// Upper bounds. Use `f64::INFINITY` for an unbounded upper side.
    pub upper: Vec<f64>,
}

impl Bounds {
    /// Creates bounds and validates shape and ordering.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self, OptimError> {
        let bounds = Self { lower, upper };
        bounds.validate()?;
        Ok(bounds)
    }

    /// Creates fully unbounded bounds for `n` parameters.
    pub fn unbounded(n: usize) -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY; n],
            upper: vec![f64::INFINITY; n],
        }
    }

    /// Returns the number of bounded coordinates.
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// Returns `true` when there are no coordinates.
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Validates that lower and upper vectors have the same length and values.
    pub fn validate(&self) -> Result<(), OptimError> {
        if self.lower.len() != self.upper.len() {
            return Err(OptimError::DimensionMismatch {
                name: "bounds",
                expected: self.lower.len(),
                actual: self.upper.len(),
            });
        }
        for (index, (&lower, &upper)) in self.lower.iter().zip(&self.upper).enumerate() {
            if lower.is_nan() || upper.is_nan() {
                return Err(OptimError::InvalidBounds {
                    index: Some(index),
                    lower: Some(lower),
                    upper: Some(upper),
                    reason: "NaN bounds are not allowed".to_string(),
                });
            }
            if lower > upper {
                return Err(OptimError::InvalidBounds {
                    index: Some(index),
                    lower: Some(lower),
                    upper: Some(upper),
                    reason: "lower must be less than or equal to upper".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Validates that bounds are well formed and match `n`.
    pub fn validate_for_dimension(&self, n: usize) -> Result<(), OptimError> {
        self.validate()?;
        if self.len() == n {
            Ok(())
        } else {
            Err(OptimError::DimensionMismatch {
                name: "bounds",
                expected: n,
                actual: self.len(),
            })
        }
    }

    /// Returns whether `par` is inside the bounds.
    pub fn contains(&self, par: &[f64]) -> Result<(), OptimError> {
        self.validate_for_dimension(par.len())?;
        for (index, ((&value, &lower), &upper)) in par
            .iter()
            .zip(self.lower.iter())
            .zip(self.upper.iter())
            .enumerate()
        {
            if value < lower || value > upper {
                return Err(OptimError::InitialParameterOutOfBounds {
                    index,
                    value,
                    lower,
                    upper,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_infinite_bounds() {
        let bounds = Bounds::unbounded(2);
        assert!(bounds.validate_for_dimension(2).is_ok());
    }

    #[test]
    fn rejects_nan_bounds() {
        let err = Bounds::new(vec![f64::NAN], vec![1.0]).unwrap_err();
        assert!(matches!(err, OptimError::InvalidBounds { .. }));
    }
}
