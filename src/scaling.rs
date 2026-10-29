use crate::bounds::Bounds;
use crate::error::{OptimError, checked_finite};

#[derive(Debug, Clone)]
pub(crate) struct Scaling {
    fnscale: f64,
    parscale: Vec<f64>,
}

impl Scaling {
    pub(crate) fn new(fnscale: f64, parscale: Vec<f64>) -> Self {
        Self { fnscale, parscale }
    }

    pub(crate) fn user_to_internal_par(&self, par: &[f64]) -> Result<Vec<f64>, OptimError> {
        par.iter()
            .zip(self.parscale.iter())
            .enumerate()
            .map(|(index, (&value, &scale))| {
                checked_finite(value / scale, "parameter scaling", Some(index))
            })
            .collect()
    }

    pub(crate) fn internal_to_user_par(&self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        let mut target = Vec::with_capacity(x.len());
        self.internal_to_user_par_into(x, &mut target)?;
        Ok(target)
    }

    pub(crate) fn internal_to_user_par_into(
        &self,
        x: &[f64],
        target: &mut Vec<f64>,
    ) -> Result<(), OptimError> {
        if x.len() != self.parscale.len() {
            return Err(OptimError::DimensionMismatch {
                name: "internal parameters",
                expected: self.parscale.len(),
                actual: x.len(),
            });
        }
        target.clear();
        target.reserve(x.len());
        for (index, (&value, &scale)) in x.iter().zip(&self.parscale).enumerate() {
            target.push(checked_finite(
                value * scale,
                "parameter unscaling",
                Some(index),
            )?);
        }
        Ok(())
    }

    pub(crate) fn user_value_to_internal(&self, value: f64) -> Result<f64, OptimError> {
        checked_finite(value / self.fnscale, "objective scaling", None)
    }

    pub(crate) fn internal_value_to_user(&self, value: f64) -> Result<f64, OptimError> {
        checked_finite(value * self.fnscale, "objective unscaling", None)
    }

    #[cfg(test)]
    pub(crate) fn user_gradient_to_internal(
        &self,
        gradient: &[f64],
    ) -> Result<Vec<f64>, OptimError> {
        self.user_gradient_to_internal_owned(gradient.to_vec())
    }

    pub(crate) fn user_gradient_to_internal_owned(
        &self,
        mut gradient: Vec<f64>,
    ) -> Result<Vec<f64>, OptimError> {
        if gradient.len() != self.parscale.len() {
            return Err(OptimError::InvalidGradient {
                index: None,
                value: None,
                reason: format!(
                    "expected length {}, got {}",
                    self.parscale.len(),
                    gradient.len()
                ),
            });
        }
        for (index, (value, &scale)) in gradient.iter_mut().zip(&self.parscale).enumerate() {
            if !value.is_finite() {
                return Err(OptimError::InvalidGradient {
                    index: Some(index),
                    value: Some(*value),
                    reason: "gradient entries must be finite".to_owned(),
                });
            }
            *value = checked_finite(
                *value * scale / self.fnscale,
                "gradient scaling",
                Some(index),
            )?;
        }
        Ok(gradient)
    }

    pub(crate) fn scale_bounds(&self, bounds: &Bounds) -> Result<Bounds, OptimError> {
        let scale_side = |values: &[f64]| -> Result<Vec<f64>, OptimError> {
            values
                .iter()
                .zip(&self.parscale)
                .enumerate()
                .map(|(index, (&value, &scale))| {
                    let scaled = value / scale;
                    if value.is_finite() {
                        checked_finite(scaled, "bound scaling", Some(index))
                    } else {
                        Ok(scaled)
                    }
                })
                .collect()
        };
        let scaled = Bounds {
            lower: scale_side(&bounds.lower)?,
            upper: scale_side(&bounds.upper)?,
        };
        for index in 0..bounds.len() {
            if bounds.lower[index] < bounds.upper[index]
                && scaled.lower[index] == scaled.upper[index]
            {
                return Err(OptimError::InvalidBounds {
                    index: Some(index),
                    lower: Some(bounds.lower[index]),
                    upper: Some(bounds.upper[index]),
                    reason: "distinct bounds collapse under parameter scaling".to_owned(),
                });
            }
        }
        Ok(scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_parameters_both_ways() {
        let scaling = Scaling::new(1.0, vec![2.0, 4.0]);
        let internal = scaling.user_to_internal_par(&[8.0, -8.0]).unwrap();
        assert_eq!(internal, vec![4.0, -2.0]);
        assert_eq!(
            scaling.internal_to_user_par(&internal).unwrap(),
            vec![8.0, -8.0]
        );
    }

    #[test]
    fn scales_value_by_fnscale() {
        let scaling = Scaling::new(-2.0, vec![1.0]);
        assert_eq!(scaling.user_value_to_internal(6.0).unwrap(), -3.0);
        assert_eq!(scaling.internal_value_to_user(-3.0).unwrap(), 6.0);
    }

    #[test]
    fn scales_gradient_for_internal_coordinates() {
        let scaling = Scaling::new(2.0, vec![4.0]);
        let gradient = scaling.user_gradient_to_internal(&[3.0]).unwrap();
        assert_eq!(gradient, vec![6.0]);
    }

    #[test]
    fn scales_bounds() {
        let scaling = Scaling::new(1.0, vec![2.0]);
        let bounds = Bounds::new(vec![-4.0], vec![8.0]).unwrap();
        let scaled = scaling.scale_bounds(&bounds).unwrap();
        assert_eq!(scaled.lower, vec![-2.0]);
        assert_eq!(scaled.upper, vec![4.0]);
    }
}

#[cfg(test)]
mod numerical_tests {
    use super::*;

    #[test]
    fn final_coordinate_and_value_conversions_reject_overflow() {
        let scaling = Scaling::new(2.0, vec![2.0]);
        assert!(matches!(
            scaling.internal_to_user_par(&[f64::MAX]),
            Err(OptimError::NumericalFailure {
                operation: "parameter unscaling",
                index: Some(0),
                ..
            })
        ));
        assert!(matches!(
            scaling.internal_value_to_user(f64::MAX),
            Err(OptimError::NumericalFailure {
                operation: "objective unscaling",
                index: None,
                ..
            })
        ));
    }

    #[test]
    fn scaling_rejects_distinct_bounds_that_round_to_the_same_coordinate() {
        let scaling = Scaling::new(1.0, vec![f64::MAX]);
        let bounds = Bounds::new(vec![f64::from_bits(1)], vec![f64::from_bits(2)]).unwrap();
        assert!(matches!(
            scaling.scale_bounds(&bounds),
            Err(OptimError::InvalidBounds { .. })
        ));
    }
}
