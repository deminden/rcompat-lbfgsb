//! Private API prototypes using the same source modules as the library.
//! This does not export a new API or change the existing callback contract.
#![allow(dead_code)]
// Cargo enables cfg(test) for custom benches but omits their test functions.
// Imports in the included library test modules consequently become unused.
#![allow(unused_imports)]

// Including the production modules lets this benchmark override the private
// BackendProblem combined-evaluation hook without making internals public.
#[path = "../src/backend/mod.rs"]
mod backend;
#[path = "../src/bounds.rs"]
mod bounds;
#[path = "../src/compat.rs"]
mod compat;
#[path = "../src/control.rs"]
mod control;
#[path = "../src/diagnostics.rs"]
mod diagnostics;
#[path = "../src/error.rs"]
mod error;
#[path = "../src/finite_diff.rs"]
mod finite_diff;
#[path = "../src/objective.rs"]
mod objective;
#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/result.rs"]
mod result;
#[path = "../src/scaling.rs"]
mod scaling;

pub use bounds::Bounds;
pub use control::OptimControl;
pub use diagnostics::{ModelSnapshot, OptimEvent, TerminationReason};
pub use error::OptimError;
pub use profile::ArithmeticProfile;
pub use result::OptimCounts;

use backend::{BackendControl, BackendProblem, BackendResult, LbfgsbBackend, NativeBackend};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;
use std::cell::Cell;
use std::time::Instant;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

struct Model {
    weights: Vec<f64>,
    work: usize,
    exponentials: Cell<usize>,
}

impl Model {
    fn value(&self, x: &[f64]) -> f64 {
        let mut value = 0.0;
        for _ in 0..self.work {
            value = 0.0;
            for (&x, &weight) in std::hint::black_box(x).iter().zip(&self.weights) {
                let residual = x.exp() - 1.0;
                value += 0.5 * weight * residual * residual;
            }
            std::hint::black_box(value);
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        value
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut gradient = vec![0.0; x.len()];
        for _ in 0..self.work {
            for ((slot, &x), &weight) in gradient
                .iter_mut()
                .zip(std::hint::black_box(x))
                .zip(&self.weights)
            {
                let mu = x.exp();
                *slot = weight * (mu - 1.0) * mu;
            }
            std::hint::black_box(&gradient);
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        gradient
    }

    fn joint_into(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        let mut value = 0.0;
        for _ in 0..self.work {
            value = 0.0;
            for ((slot, &x), &weight) in gradient
                .iter_mut()
                .zip(std::hint::black_box(x))
                .zip(&self.weights)
            {
                let mu = x.exp();
                let residual = mu - 1.0;
                value += 0.5 * weight * residual * residual;
                *slot = weight * residual * mu;
            }
            std::hint::black_box((&value, &gradient));
        }
        self.exponentials
            .set(self.exponentials.get() + x.len() * self.work);
        value
    }

    fn joint(&self, x: &[f64]) -> (f64, Vec<f64>) {
        let mut gradient = vec![0.0; x.len()];
        let value = self.joint_into(x, &mut gradient);
        (value, gradient)
    }

    fn fallible(&self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        let (value, gradient) = self.joint(x);
        if value.is_finite() {
            Ok((value, gradient))
        } else {
            Err(OptimError::NonFiniteObjective { value })
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Joint,
    Fallible,
    Buffer,
}

struct Joint<'a> {
    model: &'a Model,
    mode: Mode,
    scaling: scaling::Scaling,
    parameters: Vec<f64>,
    gradient_buffer: Vec<f64>,
}

impl BackendProblem for Joint<'_> {
    fn value(&mut self, _: &[f64]) -> Result<f64, OptimError> {
        unreachable!("joint prototype must use the combined evaluation hook")
    }
    fn gradient(&mut self, _: &[f64]) -> Result<Vec<f64>, OptimError> {
        unreachable!("joint prototype must use the combined evaluation hook")
    }
    fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
        self.scaling
            .internal_to_user_par_into(x, &mut self.parameters)?;
        let (value, gradient) = match self.mode {
            Mode::Joint => self.model.joint(&self.parameters),
            Mode::Fallible => self.model.fallible(&self.parameters)?,
            Mode::Buffer => {
                let value = self
                    .model
                    .joint_into(&self.parameters, &mut self.gradient_buffer);
                if !value.is_finite() {
                    return Err(OptimError::NonFiniteObjective { value });
                }
                let value = self.scaling.user_value_to_internal(value)?;
                let gradient = self
                    .scaling
                    .user_gradient_to_internal_owned(self.gradient_buffer.clone())?;
                return Ok((value, gradient));
            }
        };
        if !value.is_finite() {
            return Err(OptimError::NonFiniteObjective { value });
        }
        Ok((
            self.scaling.user_value_to_internal(value)?,
            self.scaling.user_gradient_to_internal_owned(gradient)?,
        ))
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Complete {
    x: Vec<u64>,
    value: u64,
    counts: OptimCounts,
    convergence: i32,
    message: String,
    iterations: usize,
    reason: TerminationReason,
}

impl Complete {
    fn fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        for value in self.x.iter().copied().chain([
            self.value,
            self.counts.function as u64,
            self.counts.gradient as u64,
            self.convergence as u64,
            self.iterations as u64,
        ]) {
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        }
        for byte in self.message.bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        hash
    }
}

impl From<BackendResult> for Complete {
    fn from(r: BackendResult) -> Self {
        Self {
            x: r.x.iter().map(|v| v.to_bits()).collect(),
            value: r.value.to_bits(),
            counts: r.counts,
            convergence: r.convergence,
            message: r.message,
            iterations: r.iterations,
            reason: r.reason,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Cost {
    allocations: usize,
    bytes: usize,
    exponentials: usize,
}

fn measure<P: BackendProblem>(
    problem: &mut P,
    initial: &[f64],
    bounds: &Bounds,
    control: BackendControl,
    model: &Model,
) -> (Complete, Cost, u128) {
    model.exponentials.set(0);
    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let result = NativeBackend
        .minimize(problem, initial, &bounds.lower, &bounds.upper, control)
        .unwrap();
    let elapsed = start.elapsed().as_nanos();
    let stats = region.change();
    (
        result.into(),
        Cost {
            allocations: stats.allocations,
            bytes: stats.bytes_allocated,
            exponentials: model.exponentials.get(),
        },
        elapsed,
    )
}

#[derive(Clone, Copy)]
struct Setup<'a> {
    initial: &'a [f64],
    bounds: &'a Bounds,
    control: BackendControl,
    model: &'a Model,
    samples: usize,
    stationary: bool,
}

impl Setup<'_> {
    fn record<P: BackendProblem>(
        &self,
        problem: &mut P,
        expected: &mut Option<Complete>,
        mode: &str,
    ) {
        let Self {
            initial,
            bounds,
            control,
            model,
            samples,
            stationary,
        } = *self;
        let (warm, cost, _) = measure(problem, initial, bounds, control, model);
        if let Some(reference) = expected {
            assert_eq!(
                reference, &warm,
                "callback prototype changed complete optimizer results"
            );
        } else {
            *expected = Some(warm);
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let (result, actual, elapsed) = measure(problem, initial, bounds, control, model);
            assert_eq!(&result, expected.as_ref().unwrap());
            assert_eq!(actual, cost);
            times.push(elapsed);
        }
        times.sort_unstable();
        let result = expected.as_ref().unwrap();
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:016x}",
            initial.len(),
            control.lmm,
            control.profile.id(),
            model.work,
            stationary,
            mode,
            times[samples / 2],
            cost.allocations,
            cost.bytes,
            cost.exponentials,
            result.counts.function,
            result.counts.gradient,
            result.iterations,
            result.fingerprint()
        );
    }
}

fn verify_prototype_failures() {
    let model = Model {
        weights: vec![4096.0; 16],
        work: 1,
        exponentials: Cell::new(0),
    };
    let bounds = Bounds::unbounded(16);
    let control = BackendControl {
        profile: ArithmeticProfile::R461MacArm,
        maxit: 24,
        factr: 0.0,
        pgtol: 0.0,
        lmm: 10,
        trace: 0,
        report: 0,
        #[cfg(test)]
        has_user_gradient: true,
    };
    for (initial, fnscale, operation) in [
        (1000.0, 1.0, None),
        (0.75, 1e-308, Some("objective scaling")),
        (1e-5, 1e-312, Some("gradient scaling")),
    ] {
        let mut expected = None;
        for mode in [Mode::Joint, Mode::Fallible, Mode::Buffer] {
            let mut joint = Joint {
                model: &model,
                mode,
                scaling: scaling::Scaling::new(fnscale, vec![1.0; 16]),
                parameters: Vec::with_capacity(16),
                gradient_buffer: vec![0.0; 16],
            };
            let error = NativeBackend
                .minimize(
                    &mut joint,
                    &[initial; 16],
                    &bounds.lower,
                    &bounds.upper,
                    control,
                )
                .unwrap_err();
            match (operation, &error) {
                (None, OptimError::NonFiniteInitialValue { value }) => assert!(value.is_infinite()),
                (Some(expected), OptimError::NumericalFailure { operation, .. }) => {
                    assert_eq!(operation, &expected)
                }
                _ => panic!("unexpected prototype failure: {error:?}"),
            }
            if let Some(expected) = &expected {
                assert_eq!(&error, expected);
            } else {
                expected = Some(error);
            }
        }
    }
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    verify_prototype_failures();
    let samples = std::env::var("LBFGSB_API_SAMPLES")
        .unwrap_or_else(|_| "11".to_owned())
        .parse::<usize>()
        .unwrap();
    assert!(samples > 0);
    println!("dimension,lmm,profile,callback_work,stationary,mode,elapsed_ns,allocations,bytes_allocated,exponentials,function_calls,gradient_calls,iterations,result_fingerprint");
    for n in [16, 128, 1024] {
        for m in [1, 10] {
            for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
                for work in [1, 16] {
                    let model = Model {
                        weights: (0..n).map(|i| 2.0_f64.powi((i % 13) as i32)).collect(),
                        work,
                        exponentials: Cell::new(0),
                    };
                    let bounds = Bounds::new(vec![-2.0; n], vec![2.0; n]).unwrap();
                    let scaling = scaling::Scaling::new(1.0, vec![1.0; n]);
                    let control = BackendControl {
                        profile,
                        maxit: 24,
                        factr: 0.0,
                        pgtol: 0.0,
                        lmm: m,
                        trace: 0,
                        report: 0,
                        #[cfg(test)]
                        has_user_gradient: true,
                    };
                    for stationary in [false, true] {
                        let initial = vec![if stationary { 0.0 } else { 0.75 }; n];
                        let mut expected = None;
                        let mut separate = objective::ScaledProblem::new(
                            |x: &[f64]| model.value(x),
                            Some(|x: &[f64]| model.gradient(x)),
                            scaling.clone(),
                            bounds.clone(),
                            vec![1e-3; n],
                            diagnostics::NoObserver,
                        );
                        let setup = Setup {
                            initial: &initial,
                            bounds: &bounds,
                            control,
                            model: &model,
                            samples,
                            stationary,
                        };
                        setup.record(&mut separate, &mut expected, "separate");
                        for mode in [Mode::Joint, Mode::Fallible, Mode::Buffer] {
                            let mut joint = Joint {
                                model: &model,
                                mode,
                                scaling: scaling.clone(),
                                parameters: Vec::with_capacity(n),
                                gradient_buffer: vec![0.0; n],
                            };
                            setup.record(
                                &mut joint,
                                &mut expected,
                                match mode {
                                    Mode::Joint => "joint",
                                    Mode::Fallible => "fallible",
                                    Mode::Buffer => "buffer",
                                },
                            );
                        }
                    }
                }
            }
        }
    }
}
