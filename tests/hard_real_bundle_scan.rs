#![allow(missing_docs)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};

const DATA_ROOT: &str = "data/lbfgsb_hard_real_2026-06-01";
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
struct HardRow {
    contrast: String,
    gene: String,
    hard_score: f64,
    force_optim_iter: Option<usize>,
    actual_optim_routed: bool,
}

#[derive(Debug)]
struct ContrastData {
    design: Vec<Vec<f64>>,
    size_factors: Vec<f64>,
    counts_by_gene: HashMap<String, Vec<f64>>,
    dispersion_by_gene: HashMap<String, f64>,
    coefficients_by_gene: HashMap<String, Coefficients>,
}

#[derive(Debug)]
struct Coefficients {
    initial: Vec<f64>,
    force_optim: Vec<f64>,
}

#[derive(Debug)]
struct ScanFailure {
    contrast: String,
    gene: String,
    hard_score: f64,
    par_error: f64,
    value_error: f64,
    value_relative_error: f64,
    result_projected_gradient: f64,
    target_projected_gradient: f64,
    convergence: i32,
    counts_function: usize,
    counts_gradient: usize,
    force_optim_iter: Option<usize>,
}

#[test]
fn hard_real_bundle_can_scan_ignored_reference_data() {
    if env::var_os("LBFGSB_HARD_REAL_SCAN").is_none() {
        return;
    }

    // This opt-in scan compares against the DESeq2 wrapper's saved
    // betaForceOptim rows, not against a freshly generated direct R optim()
    // oracle. It is useful for routing diagnostics over the ignored hard-real
    // bundle; the committed subset test owns direct objective-only R parity.
    let root = Path::new(DATA_ROOT);
    assert!(
        root.exists(),
        "set LBFGSB_HARD_REAL_SCAN only when {DATA_ROOT} is available"
    );

    let limit = env_usize("LBFGSB_HARD_REAL_LIMIT").unwrap_or(512);
    let par_tolerance = env_f64("LBFGSB_HARD_REAL_PAR_TOL").unwrap_or(1e-7);
    let value_tolerance = env_f64("LBFGSB_HARD_REAL_VALUE_TOL").unwrap_or(1e-6);
    let value_relative_tolerance = env_f64("LBFGSB_HARD_REAL_VALUE_REL_TOL").unwrap_or(1e-12);
    let strict = env::var_os("LBFGSB_HARD_REAL_STRICT").is_some();
    let verbose = env::var_os("LBFGSB_HARD_REAL_VERBOSE").is_some();
    let only_contrast = env::var("LBFGSB_HARD_REAL_ONLY_CONTRAST").ok();
    let only_gene = env::var("LBFGSB_HARD_REAL_ONLY_GENE").ok();

    let rows = read_hard_rows(&root.join("global_hardest_512.tsv"));
    let mut contrast_cache = HashMap::<String, ContrastData>::new();
    let mut total = 0_usize;
    let mut par_close = 0_usize;
    let mut value_close = 0_usize;
    let mut both_close = 0_usize;
    let mut failures = Vec::new();

    for row in rows.iter().take(limit) {
        if only_contrast
            .as_deref()
            .is_some_and(|expected| expected != row.contrast)
            || only_gene
                .as_deref()
                .is_some_and(|expected| expected != row.gene)
        {
            continue;
        }
        let contrast_data = contrast_cache
            .entry(row.contrast.clone())
            .or_insert_with(|| ContrastData::load(root, &row.contrast));
        let counts = contrast_data
            .counts_by_gene
            .get(&row.gene)
            .unwrap_or_else(|| panic!("missing counts for {}", row.gene));
        let dispersion = *contrast_data
            .dispersion_by_gene
            .get(&row.gene)
            .unwrap_or_else(|| panic!("missing dispersion for {}", row.gene));
        let coefficients = contrast_data
            .coefficients_by_gene
            .get(&row.gene)
            .unwrap_or_else(|| panic!("missing coefficients for {}", row.gene));

        let dimension = coefficients.initial.len();
        let bounds = Bounds::new(vec![-30.0; dimension], vec![30.0; dimension]).unwrap();
        let control = OptimControl {
            maxit: env_usize("LBFGSB_HARD_REAL_MAXIT").unwrap_or(100),
            fnscale: 1.0,
            parscale: vec![1.0; dimension],
            ndeps: vec![1e-3 * env_f64("LBFGSB_HARD_REAL_NDEPS_SCALE").unwrap_or(1.0); dimension],
            factr: env_f64("LBFGSB_HARD_REAL_FACTR").unwrap_or(1e7),
            pgtol: 0.0,
            lmm: env_usize("LBFGSB_HARD_REAL_LMM").unwrap_or(5),
            trace: env_usize("LBFGSB_HARD_REAL_BACKEND_TRACE").unwrap_or(0),
            report: 1,
        };

        let result = optim_lbfgsb(
            coefficients.initial.clone(),
            bounds,
            |beta| {
                nb_nll_without_constants(
                    beta,
                    counts,
                    &contrast_data.design,
                    &contrast_data.size_factors,
                    dispersion,
                )
            },
            control,
        )
        .unwrap_or_else(|error| panic!("{} {} failed: {error}", row.contrast, row.gene));

        let target_value = nb_nll_without_constants(
            &coefficients.force_optim,
            counts,
            &contrast_data.design,
            &contrast_data.size_factors,
            dispersion,
        );
        let target_gradient = nb_nll_gradient(
            &coefficients.force_optim,
            counts,
            &contrast_data.design,
            &contrast_data.size_factors,
            dispersion,
        );
        let result_gradient = nb_nll_gradient(
            &result.par,
            counts,
            &contrast_data.design,
            &contrast_data.size_factors,
            dispersion,
        );

        let par_error = max_abs_delta(&result.par, &coefficients.force_optim);
        let value_error = (result.value - target_value).abs();
        let value_relative_error =
            value_error / target_value.abs().max(result.value.abs()).max(1.0);
        let target_projected_gradient = projected_gradient_norm(
            &coefficients.force_optim,
            &target_gradient,
            &vec![-30.0; dimension],
            &vec![30.0; dimension],
        );
        let result_projected_gradient = projected_gradient_norm(
            &result.par,
            &result_gradient,
            &vec![-30.0; dimension],
            &vec![30.0; dimension],
        );
        let row_par_close = par_error <= par_tolerance;
        let row_value_close =
            value_error <= value_tolerance || value_relative_error <= value_relative_tolerance;
        total += 1;
        par_close += usize::from(row_par_close);
        value_close += usize::from(row_value_close);
        both_close += usize::from(row_par_close && row_value_close);

        if verbose || !(row_par_close && row_value_close) {
            if verbose {
                println!(
                    "HARD_REAL_PAR\t{}\t{}\tactual={}\texpected={}",
                    row.contrast,
                    row.gene,
                    format_vector(&result.par),
                    format_vector(&coefficients.force_optim)
                );
            }
            println!(
                "HARD_REAL_SCAN\t{}\t{}\trouted={}\tpar_close={}\tvalue_close={}\tpar_err={:.17e}\tvalue_err={:.17e}\tvalue_rel_err={:.17e}\tpg={:.17e}\ttarget_pg={:.17e}\tconv={}\tcounts={}/{}\tforce_iter={}",
                row.contrast,
                row.gene,
                row.actual_optim_routed,
                row_par_close,
                row_value_close,
                par_error,
                value_error,
                value_relative_error,
                result_projected_gradient,
                target_projected_gradient,
                result.convergence,
                result.counts.function,
                result.counts.gradient,
                row.force_optim_iter
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "NA".to_string())
            );
        }

        if !(row_par_close && row_value_close) {
            failures.push(ScanFailure {
                contrast: row.contrast.clone(),
                gene: row.gene.clone(),
                hard_score: row.hard_score,
                par_error,
                value_error,
                value_relative_error,
                result_projected_gradient,
                target_projected_gradient,
                convergence: result.convergence,
                counts_function: result.counts.function,
                counts_gradient: result.counts.gradient,
                force_optim_iter: row.force_optim_iter,
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
        "HARD_REAL_SUMMARY\ttotal={total}\tpar_close={par_close}\tvalue_close={value_close}\tboth_close={both_close}\tfailures={}",
        failures.len()
    );
    for failure in failures.iter().take(16) {
        println!(
            "HARD_REAL_FAILURE\t{}\t{}\thard_score={:.17e}\tpar_err={:.17e}\tvalue_err={:.17e}\tvalue_rel_err={:.17e}\tpg={:.17e}\ttarget_pg={:.17e}\tconv={}\tcounts={}/{}\tforce_iter={}",
            failure.contrast,
            failure.gene,
            failure.hard_score,
            failure.par_error,
            failure.value_error,
            failure.value_relative_error,
            failure.result_projected_gradient,
            failure.target_projected_gradient,
            failure.convergence,
            failure.counts_function,
            failure.counts_gradient,
            failure
                .force_optim_iter
                .map(|value| value.to_string())
                .unwrap_or_else(|| "NA".to_string())
        );
    }

    if strict {
        assert_eq!(both_close, total, "hard real parity scan failures");
    }
}

fn format_vector(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.17e}"))
        .collect::<Vec<_>>()
        .join(",")
}

impl ContrastData {
    fn load(root: &Path, contrast: &str) -> Self {
        let base = root.join("contrasts").join(contrast).join("lbfgsb");
        let design_table = parse_tsv_file(&base.join("design_matrix.tsv"));
        let design = design_table
            .rows
            .iter()
            .map(|row| row[1..].iter().map(|value| parse_f64(value)).collect())
            .collect::<Vec<Vec<f64>>>();
        let dimension = design
            .first()
            .map(Vec::len)
            .unwrap_or_else(|| panic!("empty design matrix for {contrast}"));

        let size_factor_table = parse_tsv_file(&base.join("size_factors.tsv"));
        let size_factors = size_factor_table
            .rows
            .iter()
            .map(|row| parse_f64(&row[1]))
            .collect::<Vec<f64>>();
        assert_eq!(
            design.len(),
            size_factors.len(),
            "design/size-factor row mismatch for {contrast}"
        );

        let counts_table = parse_tsv_file(&base.join("selected_counts.tsv"));
        assert_eq!(
            counts_table.headers.len() - 1,
            size_factors.len(),
            "count/size-factor row mismatch for {contrast}"
        );
        let counts_by_gene = counts_table
            .rows
            .into_iter()
            .map(|row| {
                let gene = row[0].clone();
                let counts = row[1..].iter().map(|value| parse_f64(value)).collect();
                (gene, counts)
            })
            .collect::<HashMap<_, _>>();

        let dispersion_table = parse_tsv_file(&base.join("selected_dispersions.tsv"));
        let gene_column = dispersion_table.column("gene");
        let dispersion_column = dispersion_table.column("dispersion");
        let dispersion_by_gene = dispersion_table
            .rows
            .into_iter()
            .map(|row| (row[gene_column].clone(), parse_f64(&row[dispersion_column])))
            .collect::<HashMap<_, _>>();

        let coefficients_table = parse_tsv_file(&base.join("selected_coefficients_long.tsv"));
        let coefficients_by_gene = coefficients_by_gene(&coefficients_table, dimension);

        Self {
            design,
            size_factors,
            counts_by_gene,
            dispersion_by_gene,
            coefficients_by_gene,
        }
    }
}

fn read_hard_rows(path: &Path) -> Vec<HardRow> {
    let table = parse_tsv_file(path);
    let contrast_column = table.column("contrast");
    let gene_column = table.column("gene");
    let hard_score_column = table.column("hardScore");
    let force_iter_column = table.column("forceOptimIter");
    let actual_routed_column = table.column("actualOptimRouted");
    table
        .rows
        .iter()
        .map(|row| HardRow {
            contrast: row[contrast_column].clone(),
            gene: row[gene_column].clone(),
            hard_score: parse_f64(&row[hard_score_column]),
            force_optim_iter: parse_optional_usize(&row[force_iter_column]),
            actual_optim_routed: parse_boolish(&row[actual_routed_column]),
        })
        .collect()
}

fn coefficients_by_gene(table: &Table, dimension: usize) -> HashMap<String, Coefficients> {
    let gene_column = table.column("gene");
    let index_column = table.column("coefficient_index_1based");
    let initial_column = table.column("betaNoOptim");
    let force_column = table.column("betaForceOptim");
    let mut indexed = HashMap::<String, Vec<(usize, f64, f64)>>::new();

    for row in &table.rows {
        indexed.entry(row[gene_column].clone()).or_default().push((
            row[index_column].parse::<usize>().unwrap(),
            parse_f64(&row[initial_column]),
            parse_f64(&row[force_column]),
        ));
    }

    indexed
        .into_iter()
        .map(|(gene, mut rows)| {
            rows.sort_by_key(|(index, _, _)| *index);
            assert_eq!(
                rows.len(),
                dimension,
                "coefficient count mismatch for {gene}"
            );
            let initial = rows.iter().map(|(_, value, _)| *value).collect();
            let force_optim = rows.into_iter().map(|(_, _, value)| value).collect();
            (
                gene,
                Coefficients {
                    initial,
                    force_optim,
                },
            )
        })
        .collect()
}

fn parse_tsv_file(path: &Path) -> Table {
    let text = fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read {}: {error}", path.display());
    });
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

fn nb_nll_gradient(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
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

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or_else(|error| {
        panic!("failed to parse {value:?} as f64: {error}");
    })
}

fn parse_optional_usize(value: &str) -> Option<usize> {
    if value == "NA" || value.is_empty() {
        None
    } else {
        Some(value.parse::<usize>().unwrap())
    }
}

fn parse_boolish(value: &str) -> bool {
    matches!(value, "TRUE" | "True" | "true" | "1")
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
