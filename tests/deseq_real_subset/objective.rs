pub(super) fn nb_nll_without_constants(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
) -> f64 {
    let size = 1.0 / dispersion;
    let mut total = 0.0;
    for ((&count, row), &sf) in counts.iter().zip(design).zip(size_factors) {
        let mu = (row_dot(row, beta) + sf.ln()).exp();
        let term = (count + size) * (size + mu).ln() - size * size.ln() - count * mu.ln();
        total += term;
    }
    total
}

pub(super) fn nb_nll_gradient(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
) -> Vec<f64> {
    let size = 1.0 / dispersion;
    let mut gradient = vec![0.0; beta.len()];
    for ((&count, row), &sf) in counts.iter().zip(design).zip(size_factors) {
        let mu = (row_dot(row, beta) + sf.ln()).exp();
        let residual = (count + size) * mu / (size + mu) - count;
        for (target, &x) in gradient.iter_mut().zip(row) {
            *target += x * residual;
        }
    }
    gradient
}

fn row_dot(row: &[f64], beta: &[f64]) -> f64 {
    let mut total = 0.0;
    for (&x, &b) in row.iter().zip(beta) {
        total += x * b;
    }
    total
}
