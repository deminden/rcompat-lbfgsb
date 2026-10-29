//! Frozen previous Rust callback, retained only as a scientific accuracy comparator.
#![allow(dead_code, missing_docs)]

pub fn value(beta: &[f64], y: &[f64], design: &[Vec<f64>], sf: &[f64], dispersion: f64) -> f64 {
    let r = 1.0 / dispersion;
    let offset = 0.5 * beta.len() as f64 * (2.0 * std::f64::consts::PI * 1e6).ln();
    let constant: f64 = y
        .iter()
        .map(|yi| log_gamma(yi + r) - log_gamma(r) - log_gamma(yi + 1.0))
        .sum();
    let mut value = offset - constant;
    for (i, yi) in y.iter().enumerate() {
        let mut linear = 0.0;
        for (j, beta) in beta.iter().enumerate() {
            linear += design[i][j] * beta;
        }
        let mu = (std::f64::consts::LN_2 * linear + sf[i].ln()).exp();
        value += (yi + r) * (r + mu).ln();
        value -= r * r.ln();
        value -= yi * mu.ln();
    }
    value + beta.iter().map(|b| b * b).sum::<f64>() / (2.0 * 1e6)
}

#[allow(clippy::excessive_precision)] // Preserve the frozen callback's coefficients.
pub fn log_gamma(z: f64) -> f64 {
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * z).sin().ln()
            - log_gamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}
