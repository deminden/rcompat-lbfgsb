#![allow(missing_docs)]
mod support;

use rcompat_lbfgsb::{Bounds, OptimControl, OptimResult};
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
#[cfg_attr(
    not(all(target_arch = "x86_64", target_os = "linux")),
    ignore = "full NB replay requires the recorded Linux x86 callback math; mandatory in pinned CI"
)]
fn deseq_real_gradient_optim_cases_match_r() {
    replay_gradient_cases(true);
}

#[test]
fn deseq_real_gradient_optim_cases_have_portable_scientific_accuracy() {
    replay_gradient_cases(false);
}

fn replay_gradient_cases(full_result: bool) {
    let fixture = deseq_fixture();
    let tables = deseq_tables();
    let design = design_matrix(&tables.design);
    let size_factors = size_factors(&tables.size_factors);
    let scan_only = std::env::var_os("DESEQ_PARITY_SCAN").is_some();
    let scan_verbose = std::env::var_os("DESEQ_PARITY_SCAN_VERBOSE").is_some();
    let trace_gene = std::env::var("DESEQ_TRACE_GENE").ok();

    let mut failures = Vec::new();
    assert_eq!(fixture.cases.len(), 68);
    for case in &fixture.cases {
        let counts = counts_for_gene(&tables.counts, &case.gene);
        let should_trace = trace_gene.as_deref() == Some(case.gene.as_str());
        let mut trace_call = 0_usize;
        let mut trace_gradient_call = 0_usize;
        let result = optim_lbfgsb_with_gradient(
            case.initial_par.clone(),
            Bounds::new(case.lower.clone(), case.upper.clone()).unwrap(),
            |beta| {
                if should_trace {
                    trace_call += 1;
                    println!(
                        "DESEQ_TRACE\t{}\t{}\t{}",
                        case.gene,
                        trace_call,
                        beta.iter()
                            .map(|value| format!("{value:.17e}"))
                            .collect::<Vec<_>>()
                            .join("\t")
                    );
                }
                let value = nb_nll_without_constants(
                    beta,
                    &counts,
                    &design,
                    &size_factors,
                    case.dispersion,
                );
                if should_trace {
                    println!(
                        "DESEQ_TRACE_FN\t{}\t{}\t{value:.17e}\t{:016x}",
                        case.gene,
                        trace_call,
                        value.to_bits()
                    );
                }
                value
            },
            |beta| {
                let gradient =
                    nb_nll_gradient(beta, &counts, &design, &size_factors, case.dispersion);
                if should_trace {
                    trace_gradient_call += 1;
                    println!(
                        "DESEQ_TRACE_GR\t{}\t{}\t{}",
                        case.gene,
                        trace_gradient_call,
                        gradient
                            .iter()
                            .map(|value| format!("{value:.17e}/{:016x}", value.to_bits()))
                            .collect::<Vec<_>>()
                            .join("\t")
                    );
                }
                gradient
            },
            control_from_case(&case.control),
        )
        .unwrap();

        if scan_only {
            let scan = scan_result(case, &result, 1e-8, 1e-8);
            println!(
                "DESEQ_SCAN\t{}\t{}\tpar_err={:.17e}\tvalue_err={:.17e}\tcount_delta={}\tactual_counts={}/{}\texpected_counts={}/{}",
                case.gene,
                scan.passed,
                scan.par_error,
                scan.value_error,
                scan.count_delta,
                result.counts.function,
                result.counts.gradient,
                case.result.counts.function,
                case.result.counts.gradient
            );
            if scan_verbose && !scan.passed {
                println!("DESEQ_SCAN_ACTUAL_PAR\t{}\t{:?}", case.gene, result.par);
                println!(
                    "DESEQ_SCAN_EXPECTED_PAR\t{}\t{:?}",
                    case.gene, case.result.par
                );
            }
        }
        let issues = if full_result {
            result_failures(case, &result, 1e-8, 1e-8)
        } else {
            let comparison =
                support::compare_scientific(&result, &expected_result(case), 1e-8, 1e-8);
            if !comparison.bookkeeping_differences.is_empty() {
                eprintln!("{}: {:?}", case.fixture, comparison.bookkeeping_differences);
            }
            comparison.failures
        };
        if !issues.is_empty() {
            failures.push(format!("{}: {}", case.fixture, issues.join("; ")));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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
    r_style_sum(
        counts
            .iter()
            .zip(design.iter())
            .zip(size_factors.iter())
            .map(|((&count, row), &size_factor)| {
                let eta = row_dot(row, beta) + size_factor.ln();
                let mu = eta.exp();
                (count + size) * (size + mu).ln() - size * size.ln() - count * mu.ln()
            }),
    )
}

fn r_style_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return values.iter().copied().sum();
    }
    values
        .into_iter()
        .fold(ExtendedAccumulator::ZERO, ExtendedAccumulator::add_f64)
        .to_f64()
}

/// Safe software model of the 64-bit-significand accumulator used by R's
/// platform `long double`. Inputs are binary64 and every addition rounds to
/// the x87 extended precision before the final binary64 conversion.
#[derive(Debug, Clone, Copy)]
struct ExtendedAccumulator {
    negative: bool,
    exponent: i32,
    significand: u64,
}

impl ExtendedAccumulator {
    const ZERO: Self = Self {
        negative: false,
        exponent: 0,
        significand: 0,
    };

    fn from_f64(value: f64) -> Self {
        debug_assert!(value.is_finite());
        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let exponent_field = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1_u64 << 52) - 1);
        if exponent_field == 0 {
            if fraction == 0 {
                return Self::ZERO;
            }
            let shift = fraction.leading_zeros();
            return Self {
                negative,
                exponent: -1011 - shift as i32,
                significand: fraction << shift,
            };
        }
        Self {
            negative,
            exponent: exponent_field - 1023,
            significand: ((1_u64 << 52) | fraction) << 11,
        }
    }

    fn add_f64(self, value: f64) -> Self {
        self.add(Self::from_f64(value))
    }

    fn add(self, other: Self) -> Self {
        if self.significand == 0 {
            return other;
        }
        if other.significand == 0 {
            return self;
        }

        let (larger, smaller) = if self.exponent > other.exponent
            || (self.exponent == other.exponent && self.significand >= other.significand)
        {
            (self, other)
        } else {
            (other, self)
        };
        let exponent_delta = (larger.exponent - smaller.exponent) as u32;
        let larger_extended = (larger.significand as u128) << 3;
        let smaller_extended = shift_right_jam((smaller.significand as u128) << 3, exponent_delta);
        let mut exponent = larger.exponent;

        let extended = if larger.negative == smaller.negative {
            let mut sum = larger_extended + smaller_extended;
            if sum >= 1_u128 << 67 {
                sum = shift_right_jam(sum, 1);
                exponent += 1;
            }
            sum
        } else {
            let mut difference = larger_extended - smaller_extended;
            if difference == 0 {
                return Self::ZERO;
            }
            while difference < 1_u128 << 66 {
                difference <<= 1;
                exponent -= 1;
            }
            difference
        };

        Self::round_extended(larger.negative, exponent, extended)
    }

    fn round_extended(negative: bool, mut exponent: i32, extended: u128) -> Self {
        let mut significand = extended >> 3;
        let round_bits = extended & 0b111;
        if round_bits > 0b100 || (round_bits == 0b100 && significand & 1 != 0) {
            significand += 1;
        }
        if significand >= 1_u128 << 64 {
            significand >>= 1;
            exponent += 1;
        }
        Self {
            negative,
            exponent,
            significand: significand as u64,
        }
    }

    fn to_f64(self) -> f64 {
        if self.significand == 0 {
            return 0.0;
        }
        let sign = (self.negative as u64) << 63;
        if self.exponent > 1023 {
            return f64::from_bits(sign | (0x7ff_u64 << 52));
        }

        if self.exponent >= -1022 {
            let mut exponent = self.exponent;
            let mut rounded = round_shift_nearest_even(self.significand as u128, 11);
            if rounded >= 1_u128 << 53 {
                rounded >>= 1;
                exponent += 1;
                if exponent > 1023 {
                    return f64::from_bits(sign | (0x7ff_u64 << 52));
                }
            }
            let exponent_bits = ((exponent + 1023) as u64) << 52;
            let fraction = rounded as u64 & ((1_u64 << 52) - 1);
            return f64::from_bits(sign | exponent_bits | fraction);
        }

        let shift = (-self.exponent - 1011) as u32;
        let rounded = round_shift_nearest_even(self.significand as u128, shift);
        if rounded >= 1_u128 << 52 {
            return f64::from_bits(sign | (1_u64 << 52));
        }
        f64::from_bits(sign | rounded as u64)
    }
}

fn shift_right_jam(value: u128, shift: u32) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift >= 128 {
        return u128::from(value != 0);
    }
    let discarded = value & ((1_u128 << shift) - 1);
    (value >> shift) | u128::from(discarded != 0)
}

fn round_shift_nearest_even(value: u128, shift: u32) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift >= 128 {
        return 0;
    }
    let truncated = value >> shift;
    let remainder = value & ((1_u128 << shift) - 1);
    let halfway = 1_u128 << (shift - 1);
    truncated + u128::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0))
}

fn nb_nll_gradient(
    beta: &[f64],
    counts: &[f64],
    design: &[Vec<f64>],
    size_factors: &[f64],
    dispersion: f64,
) -> Vec<f64> {
    let size = 1.0 / dispersion;
    let residuals = counts
        .iter()
        .zip(design.iter())
        .zip(size_factors.iter())
        .map(|((&count, row), &size_factor)| {
            let eta = row_dot(row, beta) + size_factor.ln();
            let mu = eta.exp();
            (count + size) * mu / (size + mu) - count
        })
        .collect::<Vec<_>>();

    (0..beta.len())
        .map(|column| openblas_cooperlake_crossprod_column(design, column, &residuals))
        .collect()
}

fn openblas_cooperlake_crossprod_column(
    design: &[Vec<f64>],
    column: usize,
    residuals: &[f64],
) -> f64 {
    assert_eq!(design.len(), residuals.len());

    // R's `crossprod(x, residual)` reaches the four-lane Cooperlake/Haswell
    // OpenBLAS DGEMV-T microkernel for this contiguous 78-row matrix. Each
    // lane is accumulated with FMA, then the four lanes are reduced in this
    // order; the final one-to-three rows are added by the scalar tail.
    let bulk_len = residuals.len() & !3;
    let mut lanes = [0.0_f64; 4];
    for index in 0..bulk_len {
        let lane = index & 3;
        lanes[lane] = design[index][column].mul_add(residuals[index], lanes[lane]);
    }
    let mut result = (lanes[2] + lanes[0]) + (lanes[3] + lanes[1]);
    if bulk_len < residuals.len() {
        let tail = (bulk_len..residuals.len())
            .map(|index| design[index][column] * residuals[index])
            .sum::<f64>();
        result += tail;
    }
    result
}

fn row_dot(row: &[f64], beta: &[f64]) -> f64 {
    // The pinned R 4.6.1 generator evaluates `x %*% beta` through the configured
    // short-column OpenBLAS DGEMV. Reproduce its four-column accumulation so
    // this test isolates optimizer behavior from callback arithmetic.
    let mut sum = 0.0;
    let mut index = 0;
    while index + 4 <= row.len() {
        let mut group = row[index] * beta[index];
        group = row[index + 1].mul_add(beta[index + 1], group);
        group = row[index + 2].mul_add(beta[index + 2], group);
        group = row[index + 3].mul_add(beta[index + 3], group);
        sum += group;
        index += 4;
    }
    if index + 2 <= row.len() {
        let group = row[index + 1].mul_add(beta[index + 1], row[index] * beta[index]);
        sum += group;
        index += 2;
    }
    if index < row.len() {
        sum += row[index] * beta[index];
    }
    sum
}

fn control_from_case(control: &DeseqControl) -> OptimControl {
    let trace = std::env::var("DESEQ_BACKEND_TRACE")
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

fn result_failures(
    case: &DeseqCase,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
) -> Vec<String> {
    support::compare(
        result,
        &expected_result(case),
        support::x86_reference_contract(par_tolerance, value_tolerance),
    )
}

fn expected_result(case: &DeseqCase) -> OptimResult {
    OptimResult {
        par: case.result.par.clone(),
        value: case.result.value,
        counts: rcompat_lbfgsb::OptimCounts {
            function: case.result.counts.function,
            gradient: case.result.counts.gradient,
        },
        convergence: case.result.convergence,
        message: case.result.message.clone(),
    }
}

#[derive(Debug)]
struct ScanResult {
    passed: bool,
    par_error: f64,
    value_error: f64,
    count_delta: usize,
}

fn scan_result(
    case: &DeseqCase,
    result: &OptimResult,
    par_tolerance: f64,
    value_tolerance: f64,
) -> ScanResult {
    let par_error = support::parameter_error(&result.par, &case.result.par);
    let value_error = (result.value - case.result.value).abs();
    let function_delta = result.counts.function.abs_diff(case.result.counts.function);
    let gradient_delta = result.counts.gradient.abs_diff(case.result.counts.gradient);
    let passed = result_failures(case, result, par_tolerance, value_tolerance).is_empty();

    ScanResult {
        passed,
        par_error,
        value_error,
        count_delta: function_delta + gradient_delta,
    }
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

use support::optim_x86_with_gradient as optim_lbfgsb_with_gradient;
