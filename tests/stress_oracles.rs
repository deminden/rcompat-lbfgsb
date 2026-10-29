//! Complete-result replay of the self-contained replacement stress corpus.
#![allow(missing_docs)]
mod support;
use rcompat_lbfgsb::*;
use serde_json::Value;
use std::collections::HashSet;

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| match v.as_str() {
            Some("Inf") => f64::INFINITY,
            Some("-Inf") => f64::NEG_INFINITY,
            _ => v.as_f64().unwrap(),
        })
        .collect()
}
struct Problem {
    n: usize,
    family: String,
    weights: Vec<f64>,
    target: Vec<f64>,
    sf: Vec<f64>,
    counts: Vec<f64>,
    dispersion: f64,
    sign: f64,
}
impl Problem {
    fn new(c: &Value) -> Self {
        let n = c["n"].as_u64().unwrap() as usize;
        let out = Self {
            n,
            family: c["family"].as_str().unwrap().to_owned(),
            weights: numbers(&c["weights"]),
            target: numbers(&c["target"]),
            sf: numbers(&c["size_factors"]),
            counts: numbers(&c["counts"]),
            dispersion: c["dispersion"].as_f64().unwrap(),
            sign: c["control"]["fnscale"].as_f64().unwrap(),
        };
        assert_eq!(out.weights.len(), n);
        assert_eq!(out.target.len(), n);
        assert_eq!(out.sf.len(), 2 * n);
        assert_eq!(out.counts.len(), 2 * n);
        out
    }
    fn value(&self, x: &[f64]) -> f64 {
        let mut total = 0.0;
        match self.family.as_str() {
            "correlated_quadratic" | "ill_conditioned_quadratic" => {
                let mut combined = 0.0;
                for (i, xi) in x.iter().enumerate() {
                    let d = xi - self.target[i];
                    combined += d;
                    total += 0.5 * self.weights[i] * d * d;
                }
                total += 0.125 * combined * combined;
            }
            "double_well" => {
                for xi in x {
                    let q = xi * xi - 1.0;
                    total += q * q;
                }
                if self.n > 1 {
                    for i in 0..self.n {
                        let j = (i + 1) % self.n;
                        let d = x[i] - x[j];
                        total += 0.125 * d * d;
                    }
                }
            }
            "negative_binomial" => {
                let size = 1.0 / self.dispersion;
                for row in 0..2 * self.n {
                    let i = row % self.n;
                    let j = (i + 1) % self.n;
                    let mut eta = x[i];
                    if row >= self.n && j != i {
                        eta += x[j];
                    }
                    let mu = self.sf[row] * eta.exp();
                    let term = (self.counts[row] + size) * (size + mu).ln()
                        - size * size.ln()
                        - self.counts[row] * mu.ln();
                    total += term;
                }
            }
            _ => panic!("unknown stress family {}", self.family),
        }
        self.sign * total
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut g = vec![0.0; self.n];
        match self.family.as_str() {
            "correlated_quadratic" | "ill_conditioned_quadratic" => {
                let mut combined = 0.0;
                for (i, xi) in x.iter().enumerate() {
                    combined += xi - self.target[i];
                }
                for i in 0..self.n {
                    g[i] = self.weights[i] * (x[i] - self.target[i]) + 0.25 * combined;
                }
            }
            "double_well" => {
                for i in 0..self.n {
                    g[i] = 4.0 * x[i] * (x[i] * x[i] - 1.0);
                }
                if self.n > 1 {
                    for i in 0..self.n {
                        let j = (i + 1) % self.n;
                        let d = x[i] - x[j];
                        g[i] += 0.25 * d;
                        g[j] -= 0.25 * d;
                    }
                }
            }
            "negative_binomial" => {
                let size = 1.0 / self.dispersion;
                for row in 0..2 * self.n {
                    let i = row % self.n;
                    let j = (i + 1) % self.n;
                    let mut eta = x[i];
                    if row >= self.n && j != i {
                        eta += x[j];
                    }
                    let mu = self.sf[row] * eta.exp();
                    let residual = (size / (size + mu)) * (mu - self.counts[row]);
                    g[i] += residual;
                    if row >= self.n && j != i {
                        g[j] += residual;
                    }
                }
            }
            _ => panic!("unknown stress family"),
        }
        for v in &mut g {
            *v *= self.sign;
        }
        g
    }
}
fn fixture() -> Value {
    let path = std::env::var_os("LBFGSB_STRESS_ORACLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/synthetic_v1/cases.json")
        });
    serde_json::from_str(&std::fs::read_to_string(path).expect("stress oracle is required"))
        .unwrap()
}

fn active_set_trace_issues(actual: &OptimResult, calls: &[Value], witness: &Value) -> Vec<String> {
    let mut issues = support::compare(
        actual,
        &support::recorded_result(&witness["result"]),
        support::Contract::Exact,
    );
    let expected_calls = witness["raw_calls"].as_u64().unwrap() as usize;
    if calls.len() != expected_calls {
        issues.push(format!(
            "active-set raw calls: {} != {expected_calls}",
            calls.len()
        ));
        return issues;
    }
    let mut digest = 0xcbf29ce484222325_u64;
    let mut add = |byte: u8| {
        digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    };
    for call in calls {
        let x = call["x"].as_array().unwrap();
        for byte in (x.len() as u64).to_le_bytes() {
            add(byte);
        }
        for value in x
            .iter()
            .map(|v| v.as_f64().unwrap())
            .chain(std::iter::once(call["value"].as_f64().unwrap()))
        {
            for byte in value.to_bits().to_le_bytes() {
                add(byte);
            }
        }
    }
    if format!("{digest:016x}") != witness["raw_callback_fnv1a64"].as_str().unwrap() {
        issues.push("active-set raw callback argument/value digest differs".to_owned());
    }
    let bases = witness["base_calls"].as_array().unwrap();
    assert_eq!(bases.len(), 64);
    assert_eq!(calls.len(), 23 * bases.len());
    for (ordinal, expected) in bases.iter().enumerate() {
        assert_eq!(expected["ordinal"].as_u64().unwrap() as usize, ordinal);
        let actual = &calls[23 * ordinal];
        let x = numbers(&actual["x"]);
        let reference = numbers(&expected["x"]);
        if x.len() != reference.len()
            || x.iter()
                .zip(&reference)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            issues.push(format!(
                "active-set base callback {ordinal} differs in parameter bits"
            ));
        }
        if actual["value"].as_f64().unwrap().to_bits()
            != expected["value"].as_f64().unwrap().to_bits()
        {
            issues.push(format!(
                "active-set base callback {ordinal} differs in value bits"
            ));
        }
    }
    issues
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "full R replay requires the recorded macOS ARM callback math; mandatory in the ARM CI job"
)]
fn stress_corpus_matches_r() {
    replay_corpus(false);
}

#[test]
fn polynomial_stress_corpus_matches_r_on_portable_hosts() {
    replay_corpus(true);
}

fn replay_corpus(polynomial_only: bool) {
    let fixture = fixture();
    assert_eq!(fixture["identity"], "synthetic_v1");
    assert_eq!(
        fixture["arithmetic_profile"],
        ArithmeticProfile::R461MacArm.id()
    );
    let cases = fixture["cases"].as_array().unwrap();
    let witness: Value =
        serde_json::from_str(include_str!("../fixtures/active_set/case_trace.json")).unwrap();
    assert_eq!(
        witness["arithmetic_profile"],
        ArithmeticProfile::R461MacArm.id()
    );
    assert!(cases.iter().any(|c| c["id"] == witness["id"]));
    assert_eq!(cases.len(), 512, "incomplete stress corpus");
    let mut identities = HashSet::new();
    for c in cases {
        assert!(
            identities.insert(c["id"].as_str().unwrap()),
            "duplicate stress case"
        );
    }
    let selected = std::env::var("LBFGSB_STRESS_CASE").ok();
    let mut executed = 0;
    let mut failures = Vec::new();
    let mut contract_matches = 0;
    for c in cases {
        let id = c["id"].as_str().unwrap();
        if polynomial_only && c["family"] == "negative_binomial" {
            continue;
        }
        if selected.as_ref().is_some_and(|s| s != id) {
            continue;
        }
        executed += 1;
        let p = Problem::new(c);
        let initial = numbers(&c["initial"]);
        assert_eq!(initial.len(), p.n);
        let bounds = Bounds::new(numbers(&c["lower"]), numbers(&c["upper"])).unwrap();
        let mut control = OptimControl::default_for_dimension(p.n);
        control.maxit = c["control"]["maxit"].as_u64().unwrap() as usize;
        control.fnscale = p.sign;
        control.parscale = numbers(&c["control"]["parscale"]);
        control.ndeps = numbers(&c["control"]["ndeps"]);
        control.factr = c["control"]["factr"].as_f64().unwrap();
        control.pgtol = c["control"]["pgtol"].as_f64().unwrap();
        control.lmm = c["control"]["lmm"].as_u64().unwrap() as usize;
        let trace_path = std::env::var_os("LBFGSB_STRESS_TRACE");
        let active_set_witness = c["id"] == witness["id"];
        let capture_calls = trace_path.is_some() || active_set_witness;
        assert!(
            trace_path.is_none() || selected.is_some(),
            "a callback trace requires one selected case"
        );
        let mut value_calls = Vec::new();
        let mut gradient_calls = Vec::new();
        let f = |x: &[f64]| {
            let value = p.value(x);
            if capture_calls {
                value_calls.push(serde_json::json!({"x":x,"value":value}));
            }
            value
        };
        let g = |x: &[f64]| {
            let gradient = p.gradient(x);
            if capture_calls {
                gradient_calls.push(serde_json::json!({"x":x,"gradient":gradient}));
            }
            gradient
        };
        let mut trace_events = Vec::new();
        let result = if trace_path.is_some() && c["gradient"].as_bool().unwrap() {
            optim_lbfgsb_with_gradient_observed(
                initial,
                bounds,
                f,
                g,
                control,
                ArithmeticProfile::R461MacArm,
                |event| trace_events.push(format!("{event:?}")),
            )
        } else if trace_path.is_some() {
            optim_lbfgsb_observed(
                initial,
                bounds,
                f,
                control,
                ArithmeticProfile::R461MacArm,
                |event| trace_events.push(format!("{event:?}")),
            )
        } else if c["gradient"].as_bool().unwrap() {
            optim_lbfgsb_with_gradient_and_profile(
                initial,
                bounds,
                f,
                g,
                control,
                ArithmeticProfile::R461MacArm,
            )
        } else {
            optim_lbfgsb_with_profile(initial, bounds, f, control, ArithmeticProfile::R461MacArm)
        };
        if let Some(path) = trace_path {
            std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"id":id,"value_calls":value_calls,"gradient_calls":gradient_calls,"events":trace_events})).unwrap()).unwrap();
        }
        let expected = support::recorded_result(&c["result"]);
        match result {
            Ok(actual) => {
                let mut issues = support::compare(
                    &actual,
                    &expected,
                    support::Contract::Numerical {
                        parameters: 1e-8,
                        value: 1e-8,
                    },
                );
                if active_set_witness {
                    issues.extend(active_set_trace_issues(&actual, &value_calls, &witness));
                }
                if issues.is_empty() {
                    contract_matches += 1;
                } else {
                    failures.push(format!("{id}: {}", issues.join("; ")));
                }
            }
            Err(e) => failures.push(format!("{id}: {e}")),
        }
    }
    assert_eq!(
        executed,
        if selected.is_some() {
            1
        } else if polynomial_only {
            384
        } else {
            512
        },
        "stress selection must execute every requested case"
    );
    eprintln!("stress full-result matches: {contract_matches}/{executed}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn negative_binomial_stress_reaches_the_independent_global_optimum() {
    // Each row has count == size_factor. Its negative log likelihood is
    // strictly convex in eta, with unique minimum eta=0. The first n rows
    // contain the identity design, so the unique coefficient optimum is x=0.
    // This checks accuracy independently of the R trajectory or its libm.
    let fixture = fixture();
    let mut executed = 0;
    for c in fixture["cases"].as_array().unwrap() {
        if c["family"] != "negative_binomial" {
            continue;
        }
        let p = Problem::new(c);
        assert_eq!(p.counts, p.sf);
        let lower = numbers(&c["lower"]);
        let upper = numbers(&c["upper"]);
        assert!(lower.iter().zip(&upper).all(|(&l, &u)| l < 0.0 && u > 0.0));
        let optimum = p.value(&vec![0.0; p.n]);
        for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
            let mut control = OptimControl::default_for_dimension(p.n);
            control.fnscale = p.sign;
            control.parscale = numbers(&c["control"]["parscale"]);
            control.ndeps = numbers(&c["control"]["ndeps"]);
            control.lmm = c["control"]["lmm"].as_u64().unwrap() as usize;
            // Accuracy controls are intentionally separate from the unchanged
            // full R-result replay. Do not compare their counts or messages.
            control.maxit = 500;
            control.factr = 100.0;
            control.pgtol = 1e-7;
            let initial = numbers(&c["initial"]);
            let bounds = Bounds::new(lower.clone(), upper.clone()).unwrap();
            let f = |x: &[f64]| p.value(x);
            let result = if c["gradient"].as_bool().unwrap() {
                optim_lbfgsb_with_gradient_and_profile(
                    initial,
                    bounds,
                    f,
                    |x| p.gradient(x),
                    control,
                    profile,
                )
            } else {
                optim_lbfgsb_with_profile(initial, bounds, f, control, profile)
            }
            .unwrap();
            let id = c["id"].as_str().unwrap();
            assert_eq!(result.convergence, 0, "{id}: {profile:?}: {result:?}");
            assert_eq!(result.par.len(), p.n);
            for (i, &x) in result.par.iter().enumerate() {
                assert!(x.is_finite() && x >= lower[i] && x <= upper[i]);
                assert!(x.abs() <= 1e-5, "{id}: {profile:?}: parameter {i}: {x}");
            }
            assert!(result.value.is_finite());
            let excess = p.sign * (result.value - optimum);
            assert!(
                excess.abs() <= 1e-9 * (1.0 + optimum.abs()),
                "{id}: {profile:?}: objective excess {excess}"
            );
            executed += 1;
        }
    }
    assert_eq!(
        executed, 256,
        "all 128 NB cases must run under both profiles"
    );
}

#[test]
fn strict_stress_gate_rejects_corrupted_result_evidence() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temporary(std::path::PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let directory = Temporary(std::env::temp_dir().join(format!(
        "rcompat-stress-gate-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let original = fixture();
    let id = original["cases"][0]["id"].as_str().unwrap();
    let path = directory.0.join("oracle.json");
    for mutation in 0..10 {
        let mut data = original.clone();
        match mutation {
            0 => {}
            1 => data["cases"][0]["result"]["par"][0] = Value::from(123.0),
            2 => data["cases"][0]["result"]["value"] = Value::from(123.0),
            3 => data["cases"][0]["result"]["counts"]["function"] = Value::from(999),
            4 => data["cases"][0]["result"]["counts"]["gradient"] = Value::from(999),
            5 => data["cases"][0]["result"]["convergence"] = Value::from(52),
            6 => data["cases"][0]["result"]["message"] = Value::from("corrupted evidence"),
            7 => {
                data["cases"].as_array_mut().unwrap().remove(0);
            }
            8 => {
                data["cases"][0]["result"]["par"]
                    .as_array_mut()
                    .unwrap()
                    .clear();
            }
            _ => data["cases"][0]["result"]["par"][0] = Value::String("NaN".to_owned()),
        }
        std::fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "stress_corpus_matches_r",
                "--include-ignored",
                "--nocapture",
            ])
            .env("LBFGSB_STRESS_ORACLE", &path)
            .env("LBFGSB_STRESS_CASE", id)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            mutation == 0,
            "mutation {mutation}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
