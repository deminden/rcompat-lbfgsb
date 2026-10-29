use super::*;
pub(super) struct Problem {
    pub(super) n: usize,
    pub(super) family: String,
    pub(super) weights: Vec<f64>,
    pub(super) target: Vec<f64>,
    pub(super) sf: Vec<f64>,
    pub(super) counts: Vec<f64>,
    pub(super) dispersion: f64,
    pub(super) sign: f64,
}
impl Problem {
    pub(super) fn new(c: &Value) -> Self {
        let n = c["n"].as_u64().unwrap() as usize;
        let out = Self {
            n,
            family: c["family"].as_str().unwrap().to_owned(),
            weights: numbers(&c["weights"]),
            target: numbers(&c["target"]),
            sf: numbers(&c["size_factors"]),
            counts: numbers(&c["counts"]),
            dispersion: c["dispersion"].as_f64().unwrap(),
            sign: c["control"]["fnscale"].as_f64().unwrap(),
        };
        assert_eq!(out.weights.len(), n);
        assert_eq!(out.target.len(), n);
        assert_eq!(out.sf.len(), 2 * n);
        assert_eq!(out.counts.len(), 2 * n);
        out
    }
    pub(super) fn value(&self, x: &[f64]) -> f64 {
        let mut total = 0.0;
        match self.family.as_str() {
            "correlated_quadratic" | "ill_conditioned_quadratic" => {
                let mut combined = 0.0;
                for (i, xi) in x.iter().enumerate() {
                    let d = xi - self.target[i];
                    combined += d;
                    total += 0.5 * self.weights[i] * d * d;
                }
                total += 0.125 * combined * combined;
            }
            "double_well" => {
                for xi in x {
                    let q = xi * xi - 1.0;
                    total += q * q;
                }
                if self.n > 1 {
                    for i in 0..self.n {
                        let j = (i + 1) % self.n;
                        let d = x[i] - x[j];
                        total += 0.125 * d * d;
                    }
                }
            }
            "negative_binomial" => {
                let size = 1.0 / self.dispersion;
                for row in 0..2 * self.n {
                    let i = row % self.n;
                    let j = (i + 1) % self.n;
                    let mut eta = x[i];
                    if row >= self.n && j != i {
                        eta += x[j];
                    }
                    let mu = self.sf[row] * eta.exp();
                    let term = (self.counts[row] + size) * (size + mu).ln()
                        - size * size.ln()
                        - self.counts[row] * mu.ln();
                    total += term;
                }
            }
            _ => panic!("unknown stress family {}", self.family),
        }
        self.sign * total
    }
    pub(super) fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.n];
        match self.family.as_str() {
            "correlated_quadratic" | "ill_conditioned_quadratic" => {
                let mut combined = 0.0;
                for (i, xi) in x.iter().enumerate() {
                    combined += xi - self.target[i];
                }
                for i in 0..self.n {
                    g[i] = self.weights[i] * (x[i] - self.target[i]) + 0.25 * combined;
                }
            }
            "double_well" => {
                for i in 0..self.n {
                    g[i] = 4.0 * x[i] * (x[i] * x[i] - 1.0);
                }
                if self.n > 1 {
                    for i in 0..self.n {
                        let j = (i + 1) % self.n;
                        let d = x[i] - x[j];
                        g[i] += 0.25 * d;
                        g[j] -= 0.25 * d;
                    }
                }
            }
            "negative_binomial" => {
                let size = 1.0 / self.dispersion;
                for row in 0..2 * self.n {
                    let i = row % self.n;
                    let j = (i + 1) % self.n;
                    let mut eta = x[i];
                    if row >= self.n && j != i {
                        eta += x[j];
                    }
                    let mu = self.sf[row] * eta.exp();
                    let residual = (size / (size + mu)) * (mu - self.counts[row]);
                    g[i] += residual;
                    if row >= self.n && j != i {
                        g[j] += residual;
                    }
                }
            }
            _ => panic!("unknown stress family"),
        }
        for v in &mut g {
            *v *= self.sign;
        }
        g
    }
}
