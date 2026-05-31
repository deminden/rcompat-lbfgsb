use crate::bounds::Bounds;
use crate::error::OptimError;
use crate::finite_diff::finite_difference_gradient;
use crate::scaling::Scaling;

pub(crate) struct ScaledProblem<F, G> {
    objective: F,
    gradient: Option<G>,
    scaling: Scaling,
    lower: Vec<f64>,
    upper: Vec<f64>,
    ndeps: Vec<f64>,
}

impl<F, G> ScaledProblem<F, G>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    pub(crate) fn new(
        objective: F,
        gradient: Option<G>,
        scaling: Scaling,
        scaled_bounds: Bounds,
        ndeps: Vec<f64>,
    ) -> Self {
        Self {
            objective,
            gradient,
            scaling,
            lower: scaled_bounds.lower,
            upper: scaled_bounds.upper,
            ndeps,
        }
    }

    pub(crate) fn user_parameters(&self, x: &[f64]) -> Vec<f64> {
        self.scaling.internal_to_user_par(x)
    }

    pub(crate) fn user_value_from_internal(&self, value: f64) -> f64 {
        self.scaling.internal_value_to_user(value)
    }

    fn internal_value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        let user_par = self.scaling.internal_to_user_par(x);
        let user_value = (self.objective)(&user_par);
        if user_value.is_finite() {
            Ok(self.scaling.user_value_to_internal(user_value))
        } else {
            Err(OptimError::NonFiniteObjective { value: user_value })
        }
    }

    fn internal_gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        if let Some(gradient) = self.gradient.as_mut() {
            let user_par = self.scaling.internal_to_user_par(x);
            let user_gradient = gradient(&user_par);
            return self.scaling.user_gradient_to_internal(&user_gradient);
        }

        let lower = self.lower.clone();
        let upper = self.upper.clone();
        let ndeps = self.ndeps.clone();
        finite_difference_gradient(x, &lower, &upper, &ndeps, |candidate| {
            self.internal_value(candidate)
        })
    }
}

impl<F, G> crate::backend::BackendProblem for ScaledProblem<F, G>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        self.internal_value(x)
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        self.internal_gradient(x)
    }
}
