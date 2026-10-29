#![allow(dead_code)]

use rcompat_lbfgsb::OptimResult;

/// The contract a recorded reference promises for floating-point fields.
#[derive(Clone, Copy)]
pub enum Contract {
    Exact,
    Numerical { parameters: f64, value: f64 },
}

/// Compare every public result field without masking NaNs or truncating vectors.
fn compare_floating_fields(
    actual: &OptimResult,
    expected: &OptimResult,
    contract: Contract,
) -> Vec<String> {
    let mut failures = Vec::new();
    if let Contract::Numerical { parameters, value } = contract {
        if !parameters.is_finite() || parameters < 0.0 || !value.is_finite() || value < 0.0 {
            return vec!["comparison tolerances must be finite and nonnegative".to_owned()];
        }
    }
    if actual.par.len() != expected.par.len() {
        failures.push(format!(
            "parameter length: {} != {}",
            actual.par.len(),
            expected.par.len()
        ));
    }
    let close = |a: f64, b: f64, tolerance: f64| {
        a.is_finite()
            && b.is_finite()
            && match contract {
                Contract::Exact => a.to_bits() == b.to_bits(),
                Contract::Numerical { .. } => (a - b).abs() <= tolerance,
            }
    };
    let (par_tol, value_tol) = match contract {
        Contract::Exact => (0.0, 0.0),
        Contract::Numerical { parameters, value } => (parameters, value),
    };
    for (index, (&a, &b)) in actual.par.iter().zip(&expected.par).enumerate() {
        if !close(a, b, par_tol) {
            failures.push(format!("parameter {index}: {a:?} != {b:?}"));
        }
    }
    if !close(actual.value, expected.value, value_tol) {
        failures.push(format!("value: {:?} != {:?}", actual.value, expected.value));
    }
    failures
}

/// Compare all floating fields and all R-style bookkeeping.
pub fn compare(actual: &OptimResult, expected: &OptimResult, contract: Contract) -> Vec<String> {
    let mut failures = compare_floating_fields(actual, expected, contract);
    if actual.counts != expected.counts {
        failures.push(format!(
            "counts: {:?} != {:?}",
            actual.counts, expected.counts
        ));
    }
    if actual.convergence != expected.convergence {
        failures.push(format!(
            "convergence: {} != {}",
            actual.convergence, expected.convergence
        ));
    }
    if actual.message != expected.message {
        failures.push(format!(
            "message: {:?} != {:?}",
            actual.message, expected.message
        ));
    }
    failures
}

/// Execute a fixture using its declared x86 reference arithmetic, regardless of features.
pub fn optim_x86<F>(
    par: Vec<f64>,
    bounds: rcompat_lbfgsb::Bounds,
    objective: F,
    control: rcompat_lbfgsb::OptimControl,
) -> Result<OptimResult, rcompat_lbfgsb::OptimError>
where
    F: FnMut(&[f64]) -> f64,
{
    rcompat_lbfgsb::optim_lbfgsb_with_profile(
        par,
        bounds,
        objective,
        control,
        rcompat_lbfgsb::ArithmeticProfile::R461X86,
    )
}

/// Supplied-gradient version of the explicitly pinned x86 fixture adapter.
pub fn optim_x86_with_gradient<F, G>(
    par: Vec<f64>,
    bounds: rcompat_lbfgsb::Bounds,
    objective: F,
    gradient: G,
    control: rcompat_lbfgsb::OptimControl,
) -> Result<OptimResult, rcompat_lbfgsb::OptimError>
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    rcompat_lbfgsb::optim_lbfgsb_with_gradient_and_profile(
        par,
        bounds,
        objective,
        gradient,
        control,
        rcompat_lbfgsb::ArithmeticProfile::R461X86,
    )
}

/// Enable the complete bit contract explicitly on the pinned reference runner.
/// Diagnostic scan flags never weaken this or the default numerical contract.
pub fn x86_reference_contract(parameters: f64, value: f64) -> Contract {
    match std::env::var("LBFGSB_EXACT_REFERENCE") {
        Ok(value) if value == "1" => Contract::Exact,
        Err(std::env::VarError::NotPresent) => Contract::Numerical { parameters, value },
        _ => panic!("LBFGSB_EXACT_REFERENCE must be absent or exactly 1"),
    }
}

/// Accuracy evidence from a package fit whose optimizer path may legitimately differ.
/// This is deliberately distinct from a full-result parity contract.
pub struct ScientificComparison {
    pub failures: Vec<String>,
    pub bookkeeping_differences: Vec<String>,
    pub max_parameter_error: f64,
    pub value_error: f64,
}

/// Scientific accuracy requires finite, correctly shaped endpoints, objective
/// accuracy and identical convergence status. Counts and termination messages
/// are explicitly recorded observations, not exact-parity assertions.
pub fn compare_scientific(
    actual: &OptimResult,
    expected: &OptimResult,
    parameters: f64,
    value: f64,
) -> ScientificComparison {
    let mut failures =
        compare_floating_fields(actual, expected, Contract::Numerical { parameters, value });
    if actual.convergence != expected.convergence {
        failures.push(format!(
            "convergence: {} != {}",
            actual.convergence, expected.convergence
        ));
    }
    let mut bookkeeping_differences = Vec::new();
    if actual.counts != expected.counts {
        bookkeeping_differences.push(format!(
            "counts: {:?} != {:?}",
            actual.counts, expected.counts
        ));
    }
    if actual.message != expected.message {
        bookkeeping_differences.push(format!(
            "message: {:?} != {:?}",
            actual.message, expected.message
        ));
    }
    let max_parameter_error = parameter_error(&actual.par, &expected.par);
    let value_error = if actual.value.is_finite() && expected.value.is_finite() {
        (actual.value - expected.value).abs()
    } else {
        f64::INFINITY
    };
    ScientificComparison {
        failures,
        bookkeeping_differences,
        max_parameter_error,
        value_error,
    }
}

/// Decode both named optim counts and the positional package-capture encoding.
pub fn recorded_result(value: &serde_json::Value) -> OptimResult {
    let counts = &value["counts"];
    let (function, gradient) = if let Some(counts) = counts.as_array() {
        assert_eq!(counts.len(), 2, "optimizer count shape");
        (counts[0].as_u64().unwrap(), counts[1].as_u64().unwrap())
    } else {
        (
            counts["function"].as_u64().unwrap(),
            counts["gradient"].as_u64().unwrap(),
        )
    };
    OptimResult {
        par: value["par"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
        value: value["value"].as_f64().unwrap(),
        counts: rcompat_lbfgsb::OptimCounts {
            function: function.try_into().unwrap(),
            gradient: gradient.try_into().unwrap(),
        },
        convergence: value["convergence"].as_i64().unwrap().try_into().unwrap(),
        message: value["message"].as_str().unwrap().to_owned(),
    }
}

/// Endpoint distance for diagnostics without a complete saved optimizer result.
/// Shape mismatches and any non-finite coordinate are failures, never zero error.
pub fn parameter_error(actual: &[f64], expected: &[f64]) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                (a - b).abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}
