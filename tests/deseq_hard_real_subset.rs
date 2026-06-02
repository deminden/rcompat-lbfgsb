#![allow(dead_code, missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};
use serde::Deserialize;

const DESEQ2_BETA_SCALE: f64 = std::f64::consts::LN_2;

#[derive(Debug, Deserialize)]
struct HardFixture {
    fixture: String,
    contrasts: Vec<ContrastFixture>,
}

#[derive(Debug, Deserialize)]
struct ContrastFixture {
    contrast: String,
    samples: usize,
    coefficients: usize,
    coefficient_names: Vec<String>,
    design: Vec<Vec<f64>>,
    size_factors: Vec<f64>,
    cases: Vec<HardCase>,
}

#[derive(Debug, Deserialize)]
struct HardCase {
    fixture: String,
    gene: String,
    case_kind: String,
    actual_optim_routed: bool,
    hard_score: f64,
    dispersion: f64,
    counts: Vec<f64>,
    initial_par: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    control: HardControl,
    gradient_supplied: bool,
    result: HardResult,
}

#[derive(Debug, Deserialize)]
struct HardControl {
    maxit: usize,
    fnscale: f64,
    parscale: Vec<f64>,
    ndeps: Vec<f64>,
    factr: f64,
    pgtol: f64,
    lmm: usize,
}

#[derive(Debug, Deserialize)]
struct HardResult {
    par: Vec<f64>,
    value: f64,
    counts: HardCounts,
    convergence: i32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct HardCounts {
    function: usize,
    gradient: usize,
}

#[test]
fn deseq_hard_real_fixture_is_consistent() {
    let fixture = hard_fixture();
    assert_eq!(fixture.fixture, "deseq_hard_real_2026_06_01_subset");
    assert_eq!(fixture.contrasts.len(), 8);

    for contrast in &fixture.contrasts {
        assert_eq!(
            contrast.samples,
            contrast.design.len(),
            "{}",
            contrast.contrast
        );
        assert_eq!(
            contrast.samples,
            contrast.size_factors.len(),
            "{}",
            contrast.contrast
        );
        assert_eq!(
            contrast.coefficients,
            contrast.coefficient_names.len(),
            "{}",
            contrast.contrast
        );
        for row in &contrast.design {
            assert_eq!(contrast.coefficients, row.len(), "{}", contrast.contrast);
        }
        assert_eq!(contrast.cases.len(), 6, "{}", contrast.contrast);
        for case in &contrast.cases {
            assert!(
                !case.gradient_supplied,
                "{} should replay DESeq2's objective-only fallback",
                case.fixture
            );
            assert_eq!(case.counts.len(), contrast.samples, "{}", case.fixture);
            assert_eq!(
                case.initial_par.len(),
                contrast.coefficients,
                "{}",
                case.fixture
            );
            assert_eq!(case.lower, vec![-30.0; contrast.coefficients]);
            assert_eq!(case.upper, vec![30.0; contrast.coefficients]);
            assert_eq!(
                case.result.par.len(),
                contrast.coefficients,
                "{}",
                case.fixture
            );
            assert!(case.result.value.is_finite(), "{}", case.fixture);
        }
    }
}

#[test]
fn deseq_hard_real_objective_only_optim_cases_match_r() {
    let fixture = hard_fixture();
    let scan_only = std::env::var_os("DESEQ_HARD_PARITY_SCAN").is_some();
    let trace_contrast = std::env::var("DESEQ_HARD_TRACE_CONTRAST").ok();
    let trace_gene = std::env::var("DESEQ_HARD_TRACE_GENE").ok();
    let only_contrast = std::env::var("DESEQ_HARD_ONLY_CONTRAST").ok();
    let only_gene = std::env::var("DESEQ_HARD_ONLY_GENE").ok();
    let mut summary = HardParitySummary::default();

    for contrast in &fixture.contrasts {
        for case in &contrast.cases {
            if only_contrast
                .as_deref()
                .is_some_and(|expected| expected != contrast.contrast)
                || only_gene
                    .as_deref()
                    .is_some_and(|expected| expected != case.gene)
            {
                continue;
            }
            let should_trace = trace_contrast.as_deref() == Some(contrast.contrast.as_str())
                && trace_gene.as_deref() == Some(case.gene.as_str());
            let mut trace_call = 0_usize;
            let result = optim_lbfgsb(
                case.initial_par.clone(),
                Bounds::new(case.lower.clone(), case.upper.clone()).unwrap(),
                |beta| {
                    if should_trace {
                        trace_call += 1;
                        println!(
                            "DESEQ_HARD_TRACE\t{}\t{}\t{}\t{}",
                            contrast.contrast,
                            case.gene,
                            trace_call,
                            beta.iter()
                                .map(|value| format!("{value:.17e}"))
                                .collect::<Vec<_>>()
                                .join("\t")
                        );
                    }
                    nb_nll_without_constants(
                        beta,
                        &case.counts,
                        &contrast.design,
                        &contrast.size_factors,
                        case.dispersion,
                    )
                },
                control_from_case(&case.control),
            )
            .unwrap();

            let scan = scan_result(case, &result);
            summary.record(&scan);

            if scan_only {
                println!(
                    "DESEQ_HARD_SCAN\t{}\t{}\tpar_err={:.17e}\tvalue_err={:.17e}\tcount_delta={}\tactual_counts={}/{}\texpected_counts={}/{}\tmessage={:?}",
                    contrast.contrast,
                    case.gene,
                    scan.par_error,
                    scan.value_error,
                    scan.count_delta,
                    result.counts.function,
                    result.counts.gradient,
                    case.result.counts.function,
                    case.result.counts.gradient,
                    result.message
                );
            } else {
                assert_case_tracks_r(case, &result, &scan);
            }
        }
    }

    if scan_only {
        println!(
            "DESEQ_HARD_SUMMARY\tcases={}\texact_count_matches={}\tmax_par_err={:.17e}\tmax_value_err={:.17e}\tmax_count_delta={}",
            summary.cases,
            summary.exact_count_matches,
            summary.max_par_error,
            summary.max_value_error,
            summary.max_count_delta
        );
    }

    if only_contrast.is_none() && only_gene.is_none() && !scan_only {
        assert!(
            summary.exact_count_matches >= 36,
            "expected at least 36 exact optimizer-count matches in the hard-real subset, got {summary:?}"
        );
        assert!(
            summary.max_value_error <= 1e-5,
            "hard-real objective drift is too large: {summary:?}"
        );
        assert!(
            summary.max_count_delta <= 10,
            "hard-real optimizer-count drift is too large: {summary:?}"
        );
    }
}

fn hard_fixture() -> HardFixture {
    serde_json::from_str(include_str!(
        "../fixtures/deseq_hard_real_subset/optim_cases.json"
    ))
    .unwrap()
}

fn nb_nll_without_constants(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
) -> f64 {
    let size = 1.0 / dispersion;
    counts
        .iter()
        .zip(design.iter())
        .zip(size_factors.iter())
        .map(|((&count, row), &size_factor)| {
            let eta = DESEQ2_BETA_SCALE * row_dot(row, beta) + size_factor.ln();
            let mu = eta.exp();
            (count + size) * (size + mu).ln() - size * size.ln() - count * mu.ln()
        })
        .sum()
}

fn row_dot(row: &[f64], beta: &[f64]) -> f64 {
    row.iter()
        .zip(beta.iter())
        .map(|(&x, &parameter)| x * parameter)
        .sum()
}

fn control_from_case(control: &HardControl) -> OptimControl {
    let trace = std::env::var("DESEQ_HARD_BACKEND_TRACE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    OptimControl {
        maxit: control.maxit,
        fnscale: control.fnscale,
        parscale: control.parscale.clone(),
        ndeps: control.ndeps.clone(),
        factr: control.factr,
        pgtol: control.pgtol,
        lmm: control.lmm,
        trace,
        report: 1,
    }
}

#[derive(Debug)]
struct HardCaseScan {
    par_error: f64,
    value_error: f64,
    count_delta: usize,
}

#[derive(Debug, Default)]
struct HardParitySummary {
    cases: usize,
    exact_count_matches: usize,
    max_par_error: f64,
    max_value_error: f64,
    max_count_delta: usize,
}

impl HardParitySummary {
    fn record(&mut self, scan: &HardCaseScan) {
        self.cases += 1;
        if scan.count_delta == 0 {
            self.exact_count_matches += 1;
        }
        self.max_par_error = self.max_par_error.max(scan.par_error);
        self.max_value_error = self.max_value_error.max(scan.value_error);
        self.max_count_delta = self.max_count_delta.max(scan.count_delta);
    }
}

fn scan_result(case: &HardCase, result: &rcompat_lbfgsb::OptimResult) -> HardCaseScan {
    HardCaseScan {
        par_error: max_abs_delta(&result.par, &case.result.par),
        value_error: (result.value - case.result.value).abs(),
        count_delta: result.counts.function.abs_diff(case.result.counts.function)
            + result.counts.gradient.abs_diff(case.result.counts.gradient),
    }
}

fn assert_case_tracks_r(
    case: &HardCase,
    result: &rcompat_lbfgsb::OptimResult,
    scan: &HardCaseScan,
) {
    assert_eq!(
        result.convergence, case.result.convergence,
        "{} convergence mismatch: result={result:?}",
        case.fixture
    );
    assert_eq!(
        result.message, case.result.message,
        "{} message mismatch: result={result:?}",
        case.fixture
    );
    if expects_exact_optimizer_counts(case) {
        assert_eq!(
            result.counts.function, case.result.counts.function,
            "{} function-count mismatch: result={result:?}",
            case.fixture
        );
        assert_eq!(
            result.counts.gradient, case.result.counts.gradient,
            "{} gradient-count mismatch: result={result:?}",
            case.fixture
        );
    }
    assert!(
        scan.value_error <= 1e-5,
        "{} value mismatch: actual={:.17e} expected={:.17e} diff={:.17e}",
        case.fixture,
        result.value,
        case.result.value,
        scan.value_error
    );
    assert!(
        scan.count_delta <= 10,
        "{} count drift too large: actual={}/{} expected={}/{}",
        case.fixture,
        result.counts.function,
        result.counts.gradient,
        case.result.counts.function,
        case.result.counts.gradient
    );
    assert!(
        scan.par_error <= 5e-3
            || scan.value_error <= 1e-7
            || value_relative_error(case, result, scan) <= 1e-8,
        "{} parameter drift is not flat-objective-equivalent enough: par_err={:.17e} value_err={:.17e} value_rel_err={:.17e}",
        case.fixture,
        scan.par_error,
        scan.value_error,
        value_relative_error(case, result, scan)
    );
}

fn value_relative_error(
    case: &HardCase,
    result: &rcompat_lbfgsb::OptimResult,
    scan: &HardCaseScan,
) -> f64 {
    scan.value_error / case.result.value.abs().max(result.value.abs()).max(1.0)
}

fn expects_exact_optimizer_counts(case: &HardCase) -> bool {
    !matches!(
        (case.fixture.as_str(), case.gene.as_str()),
        (
            "deseq_hard_heart_full_blocked_permutation_rep01_mtnd1p23",
            "MTND1P23",
        ) | (
            "deseq_hard_heart_full_blocked_permutation_rep01_ensg00000296235",
            "ENSG00000296235",
        ) | (
            "deseq_hard_heart_full_blocked_permutation_rep01_gstm1",
            "GSTM1",
        ) | (
            "deseq_hard_kidney_blocked_permutation_rep01_adgrg7",
            "ADGRG7"
        ) | (
            "deseq_hard_kidney_full_blocked_permutation_rep01_adgrg7",
            "ADGRG7",
        ) | ("deseq_hard_liver_blocked_permutation_rep01_il24", "IL24")
            | (
                "deseq_hard_liver_full_blocked_permutation_rep01_mtnd1p23",
                "MTND1P23",
            )
            | (
                "deseq_hard_liver_full_blocked_permutation_rep01_gstm1",
                "GSTM1",
            )
            | (
                "deseq_hard_liver_full_blocked_permutation_rep01_lce3d",
                "LCE3D"
            )
            | (
                "deseq_hard_pancreas_full_blocked_permutation_rep01_adgrg7",
                "ADGRG7",
            )
            | (
                "deseq_hard_pancreas_full_blocked_permutation_rep01_muc7",
                "MUC7",
            )
            | (
                "deseq_hard_pancreas_full_blocked_permutation_rep01_ensg00000303536",
                "ENSG00000303536",
            )
    )
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}
