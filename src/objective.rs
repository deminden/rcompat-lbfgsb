use crate::bounds::Bounds;
use crate::diagnostics::{NoObserver, OptimEvent, OptimObserver};
use crate::error::OptimError;
use crate::finite_diff::finite_difference_gradient;
use crate::scaling::Scaling;

pub(crate) struct ScaledProblem<F, G, O = NoObserver> {
    objective: F,
    observer: O,
    gradient: Option<G>,
    scaling: Scaling,
    lower: Vec<f64>,
    upper: Vec<f64>,
    ndeps: Vec<f64>,
    user_par: Vec<f64>,
}

impl<F, G, O> ScaledProblem<F, G, O>
where
    O: OptimObserver,
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    pub(crate) fn new(
        objective: F,
        gradient: Option<G>,
        scaling: Scaling,
        scaled_bounds: Bounds,
        ndeps: Vec<f64>,
        observer: O,
    ) -> Self {
        let user_par = Vec::with_capacity(scaled_bounds.len());
        Self {
            objective,
            observer,
            gradient,
            scaling,
            lower: scaled_bounds.lower,
            upper: scaled_bounds.upper,
            ndeps,
            user_par,
        }
    }

    pub(crate) fn emit(&mut self, event: OptimEvent) {
        if O::ENABLED {
            self.observer.observe(event);
        }
    }

    pub(crate) fn user_parameters(&self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        self.scaling.internal_to_user_par(x)
    }

    pub(crate) fn user_value_from_internal(&self, value: f64) -> Result<f64, OptimError> {
        self.scaling.internal_value_to_user(value)
    }

    fn internal_value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        scaled_value(&mut self.objective, &self.scaling, &mut self.user_par, x)
    }

    fn internal_gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        if let Some(gradient) = self.gradient.as_mut() {
            self.scaling
                .internal_to_user_par_into(x, &mut self.user_par)?;
            let user_gradient = gradient(&self.user_par);
            return self.scaling.user_gradient_to_internal_owned(user_gradient);
        }

        let objective = &mut self.objective;
        let scaling = &self.scaling;
        let user_par = &mut self.user_par;
        finite_difference_gradient(x, &self.lower, &self.upper, &self.ndeps, |candidate| {
            scaled_value(objective, scaling, user_par, candidate)
        })
    }
}

impl<F, G, O> crate::backend::BackendProblem for ScaledProblem<F, G, O>
where
    O: OptimObserver,
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    const OBSERVED: bool = O::ENABLED;
    fn observe(&mut self, event: OptimEvent) {
        self.emit(event);
    }

    fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
        self.internal_value(x)
    }

    fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
        self.internal_gradient(x)
    }
}

fn scaled_value<F: FnMut(&[f64]) -> f64>(
    objective: &mut F,
    scaling: &Scaling,
    user_par: &mut Vec<f64>,
    x: &[f64],
) -> Result<f64, OptimError> {
    scaling.internal_to_user_par_into(x, user_par)?;
    let user_value = objective(user_par);
    if !user_value.is_finite() {
        return Err(OptimError::NonFiniteObjective { value: user_value });
    }
    scaling.user_value_to_internal(user_value)
}
