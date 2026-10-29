use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) enum Mode {
    Joint,
    Fallible,
    Buffer,
}

pub(super) struct Joint<'a> {
    pub(super) model: &'a Model,
    pub(super) mode: Mode,
    pub(super) scaling: scaling::Scaling,
    pub(super) parameters: Vec<f64>,
    pub(super) gradient_buffer: Vec<f64>,
}

impl BackendProblem for Joint<'_> {
    fn value(&mut self, _: &[f64]) -> Result<f64, OptimError> {
        unreachable!("joint prototype must use the combined evaluation hook")
    }
    fn gradient(&mut self, _: &[f64]) -> Result<Vec<f64>, OptimError> {
        unreachable!("joint prototype must use the combined evaluation hook")
    }
    fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        self.scaling
            .internal_to_user_par_into(x, &mut self.parameters)?;
        let (value, gradient) = match self.mode {
            Mode::Joint => self.model.joint(&self.parameters),
            Mode::Fallible => self.model.fallible(&self.parameters)?,
            Mode::Buffer => {
                let value = self
                    .model
                    .joint_into(&self.parameters, &mut self.gradient_buffer);
                if !value.is_finite() {
                    return Err(OptimError::NonFiniteObjective { value });
                }
                let value = self.scaling.user_value_to_internal(value)?;
                let gradient = self
                    .scaling
                    .user_gradient_to_internal_owned(self.gradient_buffer.clone())?;
                return Ok((value, gradient));
            }
        };
        if !value.is_finite() {
            return Err(OptimError::NonFiniteObjective { value });
        }
        Ok((
            self.scaling.user_value_to_internal(value)?,
            self.scaling.user_gradient_to_internal_owned(gradient)?,
        ))
    }
}
