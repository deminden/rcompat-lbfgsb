#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb_with_gradient, Bounds, OptimControl, OptimResult};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DeseqFixture {
    fixture: String,
    samples: usize,
    coefficients: usize,
    cases: Vec<DeseqCase>,
}

#[derive(Debug, Deserialize)]
struct DeseqCase {
    fixture: String,
    gene: String,
    dispersion: f64,
    initial_par: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    control: DeseqControl,
    result: DeseqResult,
}

#[derive(Debug, Deserialize)]
struct DeseqControl {
    maxit: usize,
    fnscale: f64,
    parscale: Vec<f64>,
    ndeps: Vec<f64>,
    factr: f64,
    pgtol: f64,
    lmm: usize,
}

#[derive(Debug, Deserialize)]
struct DeseqResult {
    par: Vec<f64>,
    value: f64,
    counts: DeseqCounts,
    convergence: i32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct DeseqCounts {
    function: usize,
    gradient: usize,
}

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

    fn rows_by_gene<'a>(&'a self, gene: &str) -> Vec<&'a [String]> {
        let gene_column = self.column("gene");
        self.rows
            .iter()
            .filter(|row| row[gene_column] == gene)
            .map(Vec::as_slice)
            .collect()
    }

    fn single_row_by_gene<'a>(&'a self, gene: &str) -> &'a [String] {
        let rows = self.rows_by_gene(gene);
        assert_eq!(rows.len(), 1, "expected one row for {gene}");
        rows[0]
    }
}

#[test]
fn deseq_fixture_subset_is_consistent() {
    let fixture = deseq_fixture();
    let tables = deseq_tables();

    assert_eq!(fixture.fixture, "deseq_nb_real_subset");
    assert_eq!(fixture.samples, tables.design.rows.len());
    assert_eq!(fixture.samples, tables.size_factors.rows.len());
    assert_eq!(fixture.coefficients, tables.design.headers.len() - 1);
    assert_eq!(fixture.cases.len(), tables.counts.rows.len());

    let count_samples = &tables.counts.headers[1..];
    let size_factor_samples: Vec<&str> = tables
        .size_factors
        .rows
        .iter()
        .map(|row| row[0].as_str())
        .collect();
    assert_eq!(count_samples, size_factor_samples.as_slice());

    for case in &fixture.cases {
        let gene_case = tables.gene_cases.single_row_by_gene(&case.gene);
        let dispersion_column = tables.gene_cases.column("dispersion");
        assert_close(
            case.dispersion,
            parse_f64(&gene_case[dispersion_column]),
            1e-14,
            &case.fixture,
            "dispersion",
        );

        let initial = initial_parameters_from_coefficients(&tables.coefficients, &case.gene);
        assert_eq!(case.initial_par.len(), fixture.coefficients);
        assert_eq!(case.result.par.len(), fixture.coefficients);
        assert_eq!(case.lower, vec![-30.0; fixture.coefficients]);
        assert_eq!(case.upper, vec![30.0; fixture.coefficients]);
        assert!(case.result.value.is_finite(), "{}", case.fixture);
        assert!(case.result.counts.function > 0, "{}", case.fixture);
        assert!(case.result.counts.gradient > 0, "{}", case.fixture);
        for (actual, expected) in case.initial_par.iter().zip(initial.iter()) {
            assert_close(*actual, *expected, 1e-14, &case.fixture, "initial_par");
        }
    }
}

#[test]
fn deseq_real_gradient_optim_cases_match_r() {
    let fixture = deseq_fixture();
    let tables = deseq_tables();
    let design = design_matrix(&tables.design);
    let size_factors = size_factors(&tables.size_factors);

    for case in &fixture.cases {
        let counts = counts_for_gene(&tables.counts, &case.gene);
        let result = optim_lbfgsb_with_gradient(
            case.initial_par.clone(),
            Bounds::new(case.lower.clone(), case.upper.clone()).unwrap(),
            |beta| nb_nll_without_constants(beta, &counts, &design, &size_factors, case.dispersion),
            |beta| nb_nll_gradient(beta, &counts, &design, &size_factors, case.dispersion),
            control_from_case(&case.control),
        )
        .unwrap();

        assert_result_close(case, &result, 1e-8, 1e-8);
    }
}

fn deseq_fixture() -> DeseqFixture {
    serde_json::from_str(include_str!(
        "../fixtures/deseq_real_subset/optim_cases.json"
    ))
    .unwrap()
}

fn deseq_tables() -> DeseqTables {
    DeseqTables {
        gene_cases: parse_tsv(include_str!("../fixtures/deseq_real_subset/gene_cases.tsv")),
        counts: parse_tsv(include_str!("../fixtures/deseq_real_subset/counts.tsv")),
        coefficients: parse_tsv(include_str!(
            "../fixtures/deseq_real_subset/coefficients_long.tsv"
        )),
        design: parse_tsv(include_str!(
            "../fixtures/deseq_real_subset/design_matrix.tsv"
        )),
        size_factors: parse_tsv(include_str!(
            "../fixtures/deseq_real_subset/size_factors.tsv"
        )),
    }
}

struct DeseqTables {
    gene_cases: Table,
    counts: Table,
    coefficients: Table,
    design: Table,
    size_factors: Table,
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

fn initial_parameters_from_coefficients(coefficients: &Table, gene: &str) -> Vec<f64> {
    let gene_column = coefficients.column("gene");
    let index_column = coefficients.column("coefficient_index_1based");
    let beta_column = coefficients.column("betaNoOptim");
    let mut indexed = coefficients
        .rows
        .iter()
        .filter(|row| row[gene_column] == gene)
        .map(|row| {
            (
                row[index_column].parse::<usize>().unwrap(),
                parse_f64(&row[beta_column]),
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, value)| value).collect()
}

fn design_matrix(design: &Table) -> Vec<Vec<f64>> {
    design
        .rows
        .iter()
        .map(|row| row[1..].iter().map(|value| parse_f64(value)).collect())
        .collect()
}

fn size_factors(size_factors: &Table) -> Vec<f64> {
    size_factors
        .rows
        .iter()
        .map(|row| parse_f64(&row[1]))
        .collect()
}

fn counts_for_gene(counts: &Table, gene: &str) -> Vec<f64> {
    counts.single_row_by_gene(gene)[1..]
        .iter()
        .map(|value| parse_f64(value))
        .collect()
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
            let eta = row_dot(row, beta) + size_factor.ln();
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
        let eta = row_dot(row, beta) + size_factor.ln();
        let mu = eta.exp();
        let residual = (count + size) * mu / (size + mu) - count;
        for (target, &x) in gradient.iter_mut().zip(row.iter()) {
            *target += x * residual;
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

fn control_from_case(control: &DeseqControl) -> OptimControl {
    OptimControl {
        maxit: control.maxit,
        fnscale: control.fnscale,
        parscale: control.parscale.clone(),
        ndeps: control.ndeps.clone(),
        factr: control.factr,
        pgtol: control.pgtol,
        lmm: control.lmm,
        trace: 0,
        report: 10,
    }
}

fn assert_result_close(
    case: &DeseqCase,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
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
    for (actual, expected) in result.par.iter().zip(case.result.par.iter()) {
        assert_close(*actual, *expected, par_tolerance, &case.fixture, "par");
    }
    assert_close(
        result.value,
        case.result.value,
        value_tolerance,
        &case.fixture,
        "value",
    );
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, fixture: &str, field: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{fixture} {field} mismatch: actual={actual:?}, expected={expected:?}"
    );
}

fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().unwrap()
}
