#![allow(missing_docs)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};

const DATA_ROOT: &str = "data/lbfgsb_synthetic_stress_2026-06-06";
const DESEQ2_BETA_SCALE: f64 = std::f64::consts::LN_2;

#[derive(Debug)]
struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    fn column(&self, name: &str) -> usize {
        self.headers
            .iter()
            .position(|header| header == name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    }
}

#[derive(Debug, Clone)]
struct SyntheticCase {
    case_id: String,
    optim_value: f64,
    optim_convergence: i32,
    optim_message: String,
    optim_fn_count: usize,
    optim_gr_count: usize,
    projected_gradient_norm: f64,
}

#[derive(Debug, Clone)]
struct SyntheticProblem {
    case: SyntheticCase,
    counts: Vec<f64>,
    size_factors: Vec<f64>,
    dispersion: f64,
    design: Vec<Vec<f64>>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    ridge: Vec<f64>,
    initial: Vec<f64>,
    optim_par: Vec<f64>,
}

#[derive(Debug)]
struct ScanFailure {
    case_id: String,
    par_error: f64,
    value_error: f64,
    value_relative_error: f64,
    count_delta: usize,
    result_projected_gradient: f64,
    target_projected_gradient: f64,
    actual_counts: (usize, usize),
    expected_counts: (usize, usize),
    convergence: i32,
}

#[test]
fn synthetic_stress_can_scan_ignored_reference_data() {
    if env::var_os("LBFGSB_SYNTHETIC_STRESS_SCAN").is_none() {
        return;
    }

    let root = Path::new(DATA_ROOT);
    assert!(
        root.exists(),
        "set LBFGSB_SYNTHETIC_STRESS_SCAN only when {DATA_ROOT} is available"
    );

    let limit = env_usize("LBFGSB_SYNTHETIC_STRESS_LIMIT").unwrap_or(512);
    let par_tolerance = env_f64("LBFGSB_SYNTHETIC_STRESS_PAR_TOL").unwrap_or(5e-3);
    let value_tolerance = env_f64("LBFGSB_SYNTHETIC_STRESS_VALUE_TOL").unwrap_or(1e-5);
    let value_relative_tolerance = env_f64("LBFGSB_SYNTHETIC_STRESS_VALUE_REL_TOL").unwrap_or(1e-8);
    let strict = env::var_os("LBFGSB_SYNTHETIC_STRESS_STRICT").is_some();
    let verbose = env::var_os("LBFGSB_SYNTHETIC_STRESS_VERBOSE").is_some();
    let only_case = env::var("LBFGSB_SYNTHETIC_STRESS_ONLY_CASE").ok();

    let cases = read_cases(&root.join("cases.tsv"));
    let coefficients = read_coefficients(&root.join("coefficients.tsv"));
    let samples = read_samples(&root.join("samples.tsv.gz"));

    let mut total = 0_usize;
    let mut par_close = 0_usize;
    let mut value_close = 0_usize;
    let mut both_close = 0_usize;
    let mut exact_count_matches = 0_usize;
    let mut failures = Vec::new();

    for case in cases.into_iter().take(limit) {
        if only_case
            .as_deref()
            .is_some_and(|expected| expected != case.case_id)
        {
            continue;
        }
        let problem = SyntheticProblem::from_parts(case, &coefficients, &samples);
        let dimension = problem.initial.len();
        let ndeps_scale = env_f64("LBFGSB_SYNTHETIC_STRESS_NDEPS_SCALE").unwrap_or(1.0);
        let result = optim_lbfgsb(
            problem.initial.clone(),
            Bounds::new(problem.lower.clone(), problem.upper.clone()).unwrap(),
            |beta| {
                nb_nll_without_constants(
                    beta,
                    &problem.counts,
                    &problem.design,
                    &problem.size_factors,
                    problem.dispersion,
                    &problem.ridge,
                )
            },
            OptimControl {
                maxit: env_usize("LBFGSB_SYNTHETIC_STRESS_MAXIT").unwrap_or(100),
                fnscale: 1.0,
                parscale: vec![1.0; dimension],
                ndeps: vec![1e-3 * ndeps_scale; dimension],
                factr: env_f64("LBFGSB_SYNTHETIC_STRESS_FACTR").unwrap_or(1e7),
                pgtol: 0.0,
                lmm: env_usize("LBFGSB_SYNTHETIC_STRESS_LMM").unwrap_or(5),
                trace: env_usize("LBFGSB_SYNTHETIC_STRESS_BACKEND_TRACE").unwrap_or(0),
                report: 1,
            },
        )
        .unwrap_or_else(|error| panic!("{} failed: {error}", problem.case.case_id));

        let result_gradient = nb_nll_gradient(
            &result.par,
            &problem.counts,
            &problem.design,
            &problem.size_factors,
            problem.dispersion,
            &problem.ridge,
        );
        let par_error = max_abs_delta(&result.par, &problem.optim_par);
        let value_error = (result.value - problem.case.optim_value).abs();
        let value_relative_error = value_error
            / result
                .value
                .abs()
                .max(problem.case.optim_value.abs())
                .max(1.0);
        let count_delta = result.counts.function.abs_diff(problem.case.optim_fn_count)
            + result.counts.gradient.abs_diff(problem.case.optim_gr_count);
        let result_projected_gradient = projected_gradient_norm(
            &result.par,
            &result_gradient,
            &problem.lower,
            &problem.upper,
        );
        let row_par_close = par_error <= par_tolerance;
        let row_value_close =
            value_error <= value_tolerance || value_relative_error <= value_relative_tolerance;

        total += 1;
        par_close += usize::from(row_par_close);
        value_close += usize::from(row_value_close);
        both_close += usize::from(row_par_close && row_value_close);
        exact_count_matches += usize::from(count_delta == 0);

        if verbose || !(row_par_close && row_value_close) {
            println!(
                "SYNTHETIC_STRESS_SCAN\t{}\tpar_close={}\tvalue_close={}\tpar_err={:.17e}\tvalue_err={:.17e}\tvalue_rel_err={:.17e}\tcount_delta={}\tactual_counts={}/{}\texpected_counts={}/{}\tpg={:.17e}\ttarget_pg={:.17e}\tconv={}\texpected_conv={}",
                problem.case.case_id,
                row_par_close,
                row_value_close,
                par_error,
                value_error,
                value_relative_error,
                count_delta,
                result.counts.function,
                result.counts.gradient,
                problem.case.optim_fn_count,
                problem.case.optim_gr_count,
                result_projected_gradient,
                problem.case.projected_gradient_norm,
                result.convergence,
                problem.case.optim_convergence
            );
            if verbose {
                println!(
                    "SYNTHETIC_STRESS_PAR\t{}\tactual={}\texpected={}\tmessage={:?}\texpected_message={:?}",
                    problem.case.case_id,
                    format_vector(&result.par),
                    format_vector(&problem.optim_par),
                    result.message,
                    problem.case.optim_message
                );
            }
        }

        if !(row_par_close && row_value_close) {
            failures.push(ScanFailure {
                case_id: problem.case.case_id,
                par_error,
                value_error,
                value_relative_error,
                count_delta,
                result_projected_gradient,
                target_projected_gradient: problem.case.projected_gradient_norm,
                actual_counts: (result.counts.function, result.counts.gradient),
                expected_counts: (problem.case.optim_fn_count, problem.case.optim_gr_count),
                convergence: result.convergence,
            });
        }
    }

    failures.sort_by(|left, right| {
        right
            .par_error
            .total_cmp(&left.par_error)
            .then_with(|| right.value_error.total_cmp(&left.value_error))
    });
    println!(
        "SYNTHETIC_STRESS_SUMMARY\ttotal={total}\tpar_close={par_close}\tvalue_close={value_close}\tboth_close={both_close}\texact_count_matches={exact_count_matches}\tfailures={}",
        failures.len()
    );
    for failure in failures.iter().take(24) {
        println!(
            "SYNTHETIC_STRESS_FAILURE\t{}\tpar_err={:.17e}\tvalue_err={:.17e}\tvalue_rel_err={:.17e}\tcount_delta={}\tactual_counts={}/{}\texpected_counts={}/{}\tpg={:.17e}\ttarget_pg={:.17e}\tconv={}",
            failure.case_id,
            failure.par_error,
            failure.value_error,
            failure.value_relative_error,
            failure.count_delta,
            failure.actual_counts.0,
            failure.actual_counts.1,
            failure.expected_counts.0,
            failure.expected_counts.1,
            failure.result_projected_gradient,
            failure.target_projected_gradient,
            failure.convergence
        );
    }

    if strict {
        assert_eq!(both_close, total, "synthetic stress parity scan failures");
    }
}

impl SyntheticProblem {
    fn from_parts(
        case: SyntheticCase,
        coefficients: &HashMap<String, Vec<CoefficientRow>>,
        samples: &HashMap<String, Vec<SampleRow>>,
    ) -> Self {
        let mut coefficient_rows = coefficients
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing coefficients for {}", case.case_id))
            .clone();
        coefficient_rows.sort_by_key(|row| row.index);

        let sample_rows = samples
            .get(&case.case_id)
            .unwrap_or_else(|| panic!("missing samples for {}", case.case_id));
        let lower = coefficient_rows.iter().map(|row| row.lower).collect();
        let upper = coefficient_rows.iter().map(|row| row.upper).collect();
        let ridge = coefficient_rows.iter().map(|row| row.ridge).collect();
        let initial = coefficient_rows.iter().map(|row| row.beta_start).collect();
        let optim_par = coefficient_rows.iter().map(|row| row.optim_par).collect();
        let coefficient_names = coefficient_rows
            .iter()
            .map(|row| row.name.as_str())
            .collect::<Vec<_>>();

        Self {
            case,
            counts: sample_rows.iter().map(|row| row.count).collect(),
            size_factors: sample_rows.iter().map(|row| row.size_factor).collect(),
            dispersion: sample_rows.first().expect("case has samples").dispersion,
            design: sample_rows
                .iter()
                .map(|row| {
                    coefficient_names
                        .iter()
                        .map(|&name| *row.design.get(name).unwrap_or(&0.0))
                        .collect()
                })
                .collect(),
            lower,
            upper,
            ridge,
            initial,
            optim_par,
        }
    }
}

#[derive(Debug, Clone)]
struct CoefficientRow {
    index: usize,
    name: String,
    lower: f64,
    upper: f64,
    ridge: f64,
    beta_start: f64,
    optim_par: f64,
}

#[derive(Debug, Clone)]
struct SampleRow {
    count: f64,
    size_factor: f64,
    dispersion: f64,
    design: HashMap<String, f64>,
}

fn read_cases(path: &Path) -> Vec<SyntheticCase> {
    let table = parse_tsv_file(path);
    let case_column = table.column("case_id");
    let value_column = table.column("optim_value");
    let convergence_column = table.column("optim_convergence");
    let message_column = table.column("optim_message");
    let fn_count_column = table.column("optim_fn_count");
    let gr_count_column = table.column("optim_gr_count");
    let pg_column = table.column("projected_gradient_norm");

    table
        .rows
        .iter()
        .map(|row| SyntheticCase {
            case_id: row[case_column].clone(),
            optim_value: parse_f64(&row[value_column]),
            optim_convergence: row[convergence_column].parse().unwrap(),
            optim_message: row[message_column].clone(),
            optim_fn_count: row[fn_count_column].parse().unwrap(),
            optim_gr_count: row[gr_count_column].parse().unwrap(),
            projected_gradient_norm: parse_f64(&row[pg_column]),
        })
        .collect()
}

fn read_coefficients(path: &Path) -> HashMap<String, Vec<CoefficientRow>> {
    let table = parse_tsv_file(path);
    let case_column = table.column("case_id");
    let index_column = table.column("coefficient_index_1based");
    let name_column = table.column("coefficient");
    let lower_column = table.column("lower");
    let upper_column = table.column("upper");
    let ridge_column = table.column("ridge_log2");
    let start_column = table.column("beta_start");
    let optim_column = table.column("optim_par");
    let mut by_case = HashMap::<String, Vec<CoefficientRow>>::new();

    for row in &table.rows {
        by_case
            .entry(row[case_column].clone())
            .or_default()
            .push(CoefficientRow {
                index: row[index_column].parse().unwrap(),
                name: row[name_column].clone(),
                lower: parse_f64(&row[lower_column]),
                upper: parse_f64(&row[upper_column]),
                ridge: parse_f64(&row[ridge_column]),
                beta_start: parse_f64(&row[start_column]),
                optim_par: parse_f64(&row[optim_column]),
            });
    }

    by_case
}

fn read_samples(path: &Path) -> HashMap<String, Vec<SampleRow>> {
    let table = parse_tsv_gz_file(path);
    let case_column = table.column("case_id");
    let count_column = table.column("count");
    let size_factor_column = table.column("size_factor");
    let dispersion_column = table.column("dispersion");
    let design_start = table.column("(Intercept)");
    let mut by_case = HashMap::<String, Vec<SampleRow>>::new();

    for row in &table.rows {
        let design = table.headers[design_start..]
            .iter()
            .zip(row[design_start..].iter())
            .filter_map(|(name, value)| {
                parse_optional_f64(value).map(|value| (name.clone(), value))
            })
            .collect();
        by_case
            .entry(row[case_column].clone())
            .or_default()
            .push(SampleRow {
                count: parse_f64(&row[count_column]),
                size_factor: parse_f64(&row[size_factor_column]),
                dispersion: parse_f64(&row[dispersion_column]),
                design,
            });
    }

    by_case
}

fn parse_tsv_file(path: &Path) -> Table {
    let text = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    });
    parse_tsv(&text)
}

fn parse_tsv_gz_file(path: &Path) -> Table {
    let output = Command::new("gzip")
        .args(["-cd", path.to_str().expect("UTF-8 data path")])
        .output()
        .unwrap_or_else(|error| panic!("failed to run gzip for {}: {error}", path.display()));
    assert!(
        output.status.success(),
        "gzip failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("gzip output is UTF-8");
    parse_tsv(&text)
}

fn parse_tsv(text: &str) -> Table {
    let mut lines = text.lines();
    let headers = lines
        .next()
        .expect("table header")
        .split('\t')
        .map(str::to_string)
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split('\t').map(str::to_string).collect())
        .collect();
    Table { headers, rows }
}

fn nb_nll_without_constants(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
    ridge: &[f64],
) -> f64 {
    let size = 1.0 / dispersion;
    let nll = counts
        .iter()
        .zip(design.iter())
        .zip(size_factors.iter())
        .map(|((&count, row), &size_factor)| {
            let eta = DESEQ2_BETA_SCALE * row_dot(row, beta) + size_factor.ln();
            let mu = eta.exp();
            (count + size) * (size + mu).ln() - size * size.ln() - count * mu.ln()
        })
        .sum::<f64>();
    let penalty = beta
        .iter()
        .zip(ridge.iter())
        .map(|(&beta, &ridge)| 0.5 * ridge * beta * beta)
        .sum::<f64>();
    nll + penalty
}

fn nb_nll_gradient(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
    ridge: &[f64],
) -> Vec<f64> {
    let size = 1.0 / dispersion;
    let mut gradient = vec![0.0; beta.len()];
    for ((&count, row), &size_factor) in counts.iter().zip(design.iter()).zip(size_factors.iter()) {
        let eta = DESEQ2_BETA_SCALE * row_dot(row, beta) + size_factor.ln();
        let mu = eta.exp();
        let residual = (count + size) * mu / (size + mu) - count;
        for (target, &x) in gradient.iter_mut().zip(row.iter()) {
            *target += DESEQ2_BETA_SCALE * x * residual;
        }
    }
    for ((target, &beta), &ridge) in gradient.iter_mut().zip(beta.iter()).zip(ridge.iter()) {
        *target += ridge * beta;
    }
    gradient
}

fn row_dot(row: &[f64], beta: &[f64]) -> f64 {
    row.iter()
        .zip(beta.iter())
        .map(|(&x, &parameter)| x * parameter)
        .sum()
}

fn projected_gradient_norm(x: &[f64], gradient: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    x.iter()
        .zip(gradient.iter())
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|(((&value, &gradient), &lower), &upper)| {
            let mut component = gradient;
            if component < 0.0 && upper.is_finite() {
                component = component.max(value - upper);
            } else if component > 0.0 && lower.is_finite() {
                component = component.min(value - lower);
            }
            component.abs()
        })
        .fold(0.0, f64::max)
}

fn max_abs_delta(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f64::max)
}

fn format_vector(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or_else(|error| {
        panic!("failed to parse {value:?} as f64: {error}");
    })
}

fn parse_optional_f64(value: &str) -> Option<f64> {
    if value == "NA" || value.is_empty() {
        None
    } else {
        Some(parse_f64(value))
    }
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().map(|value| {
        value
            .parse::<usize>()
            .unwrap_or_else(|error| panic!("{name}={value:?} is not a usize: {error}"))
    })
}

fn env_f64(name: &str) -> Option<f64> {
    env::var(name).ok().map(|value| {
        value
            .parse::<f64>()
            .unwrap_or_else(|error| panic!("{name}={value:?} is not an f64: {error}"))
    })
}
