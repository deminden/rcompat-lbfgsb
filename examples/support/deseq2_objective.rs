//! Independent full negative-binomial GLM objective with the DESeq2 sd=1000
//! coefficient prior. Intended for the recorded R 4.6.2 / DESeq2 1.52 fits.
//! Formulas: Loader's saddlepoint expansion and the log-gamma power series.
//! No R implementation text or runtime dependency is used.
#![allow(missing_docs, clippy::excessive_precision)]

pub struct Deseq2Objective {
    design: Vec<Vec<f64>>,
    size_factors: Vec<f64>,
    terms: Vec<Term>,
}

impl Deseq2Objective {
    pub fn new(
        design: Vec<Vec<f64>>,
        size_factors: Vec<f64>,
        counts: &[f64],
        dispersion: f64,
    ) -> Self {
        assert_eq!(design.len(), counts.len());
        assert_eq!(size_factors.len(), counts.len());
        let r = 1.0 / dispersion;
        assert!(r.is_finite() && r > 0.0);
        Self {
            design,
            size_factors,
            terms: counts.iter().map(|&y| Term::new(y, r)).collect(),
        }
    }

    pub fn value(&self, beta: &[f64]) -> f64 {
        let nll: f64 = self
            .design
            .iter()
            .zip(&self.size_factors)
            .zip(&self.terms)
            .map(|((row, &sf), term)| {
                assert_eq!(row.len(), beta.len());
                let eta: f64 = row.iter().zip(beta).map(|(x, b)| x * b).sum();
                -term.log_density(2.0_f64.powf(eta) * sf)
            })
            .sum();
        // Preserve the actual link, per-coefficient prior, and reduction order:
        // algebraically equivalent forms can change flat-tail stopping paths.
        let prior: f64 = beta
            .iter()
            .map(|b| (0.9189385332046728 + 0.5 * (b / 1000.0).powi(2)) + 1000.0_f64.ln())
            .sum();
        nll + prior
    }
}

struct Term {
    y: f64,
    r: f64,
    n: f64,
    correction: f64,
    normalizer: f64,
    log_ratio: f64,
}
impl Term {
    fn new(y: f64, r: f64) -> Self {
        assert!(y.is_finite() && y >= 0.0 && y.fract() == 0.0);
        let n = y + r;
        let (correction, normalizer) = if y == 0.0 {
            (0.0, 0.0)
        } else {
            (
                stirling_error(n) - stirling_error(y) - stirling_error(r),
                0.5 * (2.0 * std::f64::consts::PI * y * (r / n)).ln(),
            )
        };
        Self {
            y,
            r,
            n,
            correction,
            normalizer,
            log_ratio: (r / n).ln(),
        }
    }
    fn log_density(&self, mu: f64) -> f64 {
        let Self {
            y,
            r,
            n,
            correction,
            normalizer,
            log_ratio,
        } = *self;
        if y == 0.0 {
            return if r < mu {
                r * (r / (r + mu)).ln()
            } else {
                r * (-mu / (r + mu)).ln_1p()
            };
        }
        correction - deviance(y, n * (mu / (r + mu))) - deviance(r, n * (r / (r + mu))) - normalizer
            + log_ratio
    }
}

fn deviance(x: f64, mean: f64) -> f64 {
    let t = (x - mean) / (x + mean);
    if t.abs() < 0.5 {
        let mut power = 2.0 * x * t;
        let mut value = (x - mean) * t;
        for k in 1..100 {
            power *= t * t;
            let next = value + power / (2 * k + 1) as f64;
            if next == value {
                break;
            }
            value = next;
        }
        value
    } else {
        x.mul_add((x / mean).ln(), mean - x)
    }
}

fn stirling_error(z: f64) -> f64 {
    // Correctly rounded mathematical values of the Stirling remainder at 1..15.
    const INTEGER: [f64; 15] = [
        0.08106146679532726,
        0.0413406959554093,
        0.02767792568499834,
        0.020790672103765093,
        0.016644691189821193,
        0.013876128823070748,
        0.01189670994589177,
        0.010411265261972096,
        0.009255462182712733,
        0.00833056343336287,
        0.007573675487951841,
        0.00694284010720953,
        0.006408994188004207,
        0.0059513701127588475,
        0.005554733551962801,
    ];
    if (1.0..=15.0).contains(&z) && z.fract() == 0.0 {
        return INTEGER[z as usize - 1];
    }
    if z >= 16.0 {
        let q = 1.0 / (z * z);
        return (1.0 / 12.0
            - q * (1.0 / 360.0
                - q * (1.0 / 1260.0
                    - q * (1.0 / 1680.0 - q * (1.0 / 1188.0 - q * 691.0 / 360360.0)))))
            / z;
    }
    if z >= 8.0 {
        // Gamma recurrence avoids cancellation in log-gamma minus its leading
        // Stirling terms. The increment is sum(t^(2k)/(2k+1)), t=1/(2z+1).
        let shift = (16.0 - z).ceil() as usize;
        let mut value = stirling_error(z + shift as f64);
        for j in (0..shift).rev() {
            let t = 1.0 / (2.0 * (z + j as f64) + 1.0);
            let q = t * t;
            let mut power = q;
            let mut increment = 0.0;
            for k in 1..100 {
                let next = increment + power / (2 * k + 1) as f64;
                if next == increment {
                    break;
                }
                increment = next;
                power *= q;
            }
            value += increment;
        }
        return value;
    }
    (-(z + 0.5)).mul_add(z.ln(), log_gamma_one_plus(z)) + z - 0.9189385332046728
}

#[allow(clippy::approx_constant)] // Keep compatible with the pinned Rust toolchain.
fn log_gamma_one_plus(x: f64) -> f64 {
    let n = x.round() as usize;
    let z = x - n as f64;
    let mut value = 0.0;
    let mut power = z * z;
    // Integrate the zeta series for psi(1+z), then use Gamma recurrence.
    // |z|<=.5; beyond order40, replacing zeta(k) by1 contributes <1e-25.
    for k in 2..=60 {
        let zeta = if k <= 40 { ZETA[k - 2] } else { 1.0 };
        value += if k % 2 == 0 { 1.0 } else { -1.0 } * zeta * power / k as f64;
        power *= z;
    }
    value -= 0.5772156649015329 * z;
    for j in 1..=n {
        value += (z + j as f64).ln();
    }
    value
}

const ZETA: [f64; 39] = [
    1.6449340668482264,
    1.2020569031595942,
    1.0823232337111381,
    1.03692775514337,
    1.0173430619844492,
    1.008349277381923,
    1.0040773561979444,
    1.0020083928260821,
    1.000994575127818,
    1.0004941886041194,
    1.000246086553308,
    1.0001227133475785,
    1.0000612481350588,
    1.000030588236307,
    1.0000152822594086,
    1.0000076371976379,
    1.000003817293265,
    1.0000019082127165,
    1.0000009539620338,
    1.0000004769329869,
    1.0000002384505027,
    1.000000119219926,
    1.000000059608189,
    1.0000000298035034,
    1.0000000149015549,
    1.0000000074507118,
    1.000000003725334,
    1.0000000018626598,
    1.0000000009313275,
    1.0000000004656628,
    1.000000000232831,
    1.0000000001164155,
    1.0000000000582077,
    1.0000000000291038,
    1.000000000014552,
    1.000000000007276,
    1.000000000003638,
    1.000000000001819,
    1.0000000000009095,
];
