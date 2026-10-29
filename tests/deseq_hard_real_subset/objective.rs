pub(super) const DESEQ2_BETA_SCALE: f64 = std::f64::consts::LN_2;
pub(super) fn nb_nll_without_constants(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
) -> f64 {
    let size = 1.0 / dispersion;
    let mut total = 0.0;
    for ((&count, row), &size_factor) in counts.iter().zip(design.iter()).zip(size_factors.iter()) {
        let eta = DESEQ2_BETA_SCALE * row_dot(row, beta) + size_factor.ln();
        let mu = eta.exp();
        total += (count + size) * (size + mu).ln();
        total -= size * size.ln();
        total -= count * mu.ln();
    }
    total
}

pub(super) fn row_dot(row: &[f64], beta: &[f64]) -> f64 {
    row.iter()
        .zip(beta.iter())
        .map(|(&x, &parameter)| x * parameter)
        .sum()
}
