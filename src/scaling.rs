use crate::bounds::Bounds;
use crate::error::OptimError;

#[derive(Debug, Clone)]
pub(crate) struct Scaling {
    fnscale: f64,
    parscale: Vec<f64>,
}

impl Scaling {
    pub(crate) fn new(fnscale: f64, parscale: Vec<f64>) -> Self {
        Self { fnscale, parscale }
    }

    pub(crate) fn user_to_internal_par(&self, par: &[f64]) -> Vec<f64> {
        par.iter()
            .zip(self.parscale.iter())
            .map(|(&value, &scale)| value / scale)
            .collect()
    }

    pub(crate) fn internal_to_user_par(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(self.parscale.iter())
            .map(|(&value, &scale)| value * scale)
            .collect()
    }

    pub(crate) fn user_value_to_internal(&self, value: f64) -> f64 {
        value / self.fnscale
    }

    pub(crate) fn internal_value_to_user(&self, value: f64) -> f64 {
        value * self.fnscale
    }

    pub(crate) fn user_gradient_to_internal(
        &self,
        gradient: &[f64],
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
        gradient
            .iter()
            .zip(self.parscale.iter())
            .enumerate()
            .map(|(index, (&value, &scale))| {
                if value.is_finite() {
                    Ok(value * scale / self.fnscale)
                } else {
                    Err(OptimError::InvalidGradient {
                        index: Some(index),
                        value: Some(value),
                        reason: "gradient entries must be finite".to_string(),
                    })
                }
            })
            .collect()
    }

    pub(crate) fn scale_bounds(&self, bounds: &Bounds) -> Bounds {
        Bounds {
            lower: bounds
                .lower
                .iter()
                .zip(self.parscale.iter())
                .map(|(&value, &scale)| value / scale)
                .collect(),
            upper: bounds
                .upper
                .iter()
                .zip(self.parscale.iter())
                .map(|(&value, &scale)| value / scale)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_parameters_both_ways() {
        let scaling = Scaling::new(1.0, vec![2.0, 4.0]);
        let internal = scaling.user_to_internal_par(&[8.0, -8.0]);
        assert_eq!(internal, vec![4.0, -2.0]);
        assert_eq!(scaling.internal_to_user_par(&internal), vec![8.0, -8.0]);
    }

    #[test]
    fn scales_value_by_fnscale() {
        let scaling = Scaling::new(-2.0, vec![1.0]);
        assert_eq!(scaling.user_value_to_internal(6.0), -3.0);
        assert_eq!(scaling.internal_value_to_user(-3.0), 6.0);
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
        let scaled = scaling.scale_bounds(&bounds);
        assert_eq!(scaled.lower, vec![-2.0]);
        assert_eq!(scaled.upper, vec![4.0]);
    }
}
