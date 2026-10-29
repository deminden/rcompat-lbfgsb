use super::*;
pub(super) struct Model {
    pub(super) weights: Vec<f64>,
    pub(super) work: usize,
    pub(super) exponentials: Cell<usize>,
}

impl Model {
    pub(super) fn value(&self, x: &[f64]) -> f64 {
        let mut value = 0.0;
        for _ in 0..self.work {
            value = 0.0;
            for (&x, &weight) in std::hint::black_box(x).iter().zip(&self.weights) {
                let residual = x.exp() - 1.0;
                value += 0.5 * weight * residual * residual;
            }
            std::hint::black_box(value);
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        value
    }

    pub(super) fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len()];
        for _ in 0..self.work {
            for ((slot, &x), &weight) in gradient
                .iter_mut()
                .zip(std::hint::black_box(x))
                .zip(&self.weights)
            {
                let mu = x.exp();
                *slot = weight * (mu - 1.0) * mu;
            }
            std::hint::black_box(&gradient);
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        gradient
    }

    pub(super) fn joint_into(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        let mut value = 0.0;
        for _ in 0..self.work {
            value = 0.0;
            for ((slot, &x), &weight) in gradient
                .iter_mut()
                .zip(std::hint::black_box(x))
                .zip(&self.weights)
            {
                let mu = x.exp();
                let residual = mu - 1.0;
                value += 0.5 * weight * residual * residual;
                *slot = weight * residual * mu;
            }
            std::hint::black_box((&value, &gradient));
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        value
    }

    pub(super) fn joint(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let mut gradient = vec![0.0; x.len()];
        let value = self.joint_into(x, &mut gradient);
        (value, gradient)
    }

    pub(super) fn fallible(&self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        let (value, gradient) = self.joint(x);
        if value.is_finite() {
            Ok((value, gradient))
        } else {
            Err(OptimError::NonFiniteObjective { value })
        }
    }
}
