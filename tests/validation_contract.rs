//! Tests that the reference comparator detects corrupted evidence.
mod support;
use rcompat_lbfgsb::{OptimCounts, OptimResult};
use support::{compare, Contract};

fn reference() -> OptimResult {
    OptimResult {
        par: vec![1.0, 2.0],
        value: 3.0,
        counts: OptimCounts {
            function: 4,
            gradient: 4,
        },
        convergence: 0,
        message: "done".to_owned(),
    }
}

#[test]
fn exact_contract_detects_every_public_field_mutation() {
    let expected = reference();
    assert!(compare(&expected, &expected, Contract::Exact).is_empty());
    for mutation in 0..7 {
        let mut actual = expected.clone();
        match mutation {
            0 => actual.par[0] = f64::from_bits(actual.par[0].to_bits() + 1),
            1 => actual.value = f64::from_bits(actual.value.to_bits() + 1),
            2 => actual.counts.function += 1,
            3 => actual.counts.gradient += 1,
            4 => actual.convergence = 52,
            5 => actual.message.push('!'),
            _ => {
                actual.par.pop();
            }
        }
        assert!(
            !compare(&actual, &expected, Contract::Exact).is_empty(),
            "missed mutation {mutation}"
        );
    }
}

#[test]
fn numerical_contract_rejects_nonfinite_values_and_wrong_lengths() {
    let expected = reference();
    let contract = Contract::Numerical {
        parameters: 1e-8,
        value: 1e-8,
    };
    for mutation in 0..4 {
        let mut actual = expected.clone();
        match mutation {
            0 => actual.par[0] = f64::NAN,
            1 => actual.value = f64::INFINITY,
            2 => {
                actual.par.pop();
            }
            _ => actual.par.push(0.0),
        }
        assert!(!compare(&actual, &expected, contract).is_empty());
    }
}

#[test]
fn invalid_tolerances_cannot_disable_a_validation_gate() {
    let expected = reference();
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        for contract in [
            Contract::Numerical {
                parameters: invalid,
                value: 0.0,
            },
            Contract::Numerical {
                parameters: 0.0,
                value: invalid,
            },
        ] {
            assert!(!compare(&expected, &expected, contract).is_empty());
        }
    }
}

#[test]
fn scientific_accuracy_distinguishes_failures_from_observed_bookkeeping() {
    let expected = reference();
    for field in 0..8 {
        let mut actual = expected.clone();
        match field {
            0 => actual.par[0] = f64::NAN,
            1 => actual.value = f64::INFINITY,
            2 => {
                actual.par.pop();
            }
            3 => actual.par[0] += 0.1,
            4 => actual.value += 0.1,
            5 => actual.convergence = 52,
            6 => actual.counts.function += 1,
            _ => actual.message.push('!'),
        }
        let result = support::compare_scientific(&actual, &expected, 1e-8, 1e-8);
        assert_eq!(result.failures.is_empty(), field >= 6, "field {field}");
        assert_eq!(
            result.bookkeeping_differences.is_empty(),
            field < 6,
            "field {field}"
        );
        assert!(!result.max_parameter_error.is_nan());
        assert!(!result.value_error.is_nan());
    }
}
