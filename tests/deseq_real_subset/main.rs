#![allow(missing_docs)]
#[path = "../support/mod.rs"]
mod support;

use rcompat_lbfgsb::{Bounds, OptimControl, OptimResult};
use serde::Deserialize;
use support::optim_reference_with_gradient as optim_lbfgsb_with_gradient;
mod data;
use data::*;
mod objective;
use objective::*;

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
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "full NB replay requires the recorded Apple Silicon callback math; mandatory in pinned CI"
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
    let mut differences = support::MaxAbsoluteDifferences::default();
    let mut failures = Vec::new();
    assert_eq!(fixture.cases.len(), 68);
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

        differences.record(&result, &expected_result(case));
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
    differences.report("supplied_gradient_glms");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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
        support::Contract::Numerical {
            parameters: par_tolerance,
            value: value_tolerance,
        },
    )
}

fn expected_result(case: &DeseqCase) -> OptimResult {
    case.result.to_result()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, fixture: &str, field: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{fixture} {field} mismatch: actual={actual:?}, expected={expected:?}"
    );
}
