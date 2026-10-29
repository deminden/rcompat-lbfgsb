//! Private API prototypes using the same source modules as the library.
//! This does not export a new API or change the existing callback contract.
#![allow(dead_code)]
// Cargo enables cfg(test) for custom benches but omits their test functions.
// Imports in the included library test modules consequently become unused.
#![allow(unused_imports)]

// Including the production modules lets this benchmark override the private
// BackendProblem combined-evaluation hook without making internals public.
#[path = "../../src/backend/mod.rs"]
mod backend;
#[path = "../../src/bounds.rs"]
mod bounds;
#[path = "../../src/compat.rs"]
mod compat;
#[path = "../../src/control.rs"]
mod control;
#[path = "../../src/diagnostics.rs"]
mod diagnostics;
#[path = "../../src/error.rs"]
mod error;
#[path = "../../src/finite_diff.rs"]
mod finite_diff;
#[path = "../../src/objective.rs"]
mod objective;
#[path = "../../src/profile.rs"]
mod profile;
#[path = "../../src/result.rs"]
mod result;
#[path = "../../src/scaling.rs"]
mod scaling;

pub use bounds::Bounds;
pub use control::OptimControl;
pub use diagnostics::{ModelSnapshot, OptimEvent, TerminationReason};
pub use error::OptimError;
pub use profile::ArithmeticProfile;
pub use result::OptimCounts;

use backend::{BackendControl, BackendProblem, BackendResult, LbfgsbBackend, NativeBackend};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};
use std::alloc::System;
use std::cell::Cell;
use std::time::Instant;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[path = "../support/fingerprint.rs"]
mod fingerprint;
mod model;
use model::Model;
mod adapters;
use adapters::{Joint, Mode};
mod measurement;
use measurement::*;

fn verify_prototype_failures() {
    let model = Model {
        weights: vec![4096.0; 16],
        work: 1,
        exponentials: Cell::new(0),
    };
    let bounds = Bounds::unbounded(16);
    let control = BackendControl {
        profile: ArithmeticProfile::AppleSiliconOpenBlas,
        maxit: 24,
        factr: 0.0,
        pgtol: 0.0,
        lmm: 10,
        trace: 0,
        report: 0,
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
    println!(
        "dimension,lmm,profile,callback_work,stationary,mode,elapsed_ns,allocations,bytes_allocated,exponentials,function_calls,gradient_calls,iterations,result_fingerprint"
    );
    for n in [16, 128, 1024] {
        for m in [1, 10] {
            for profile in [
                ArithmeticProfile::X86OpenBlas,
                ArithmeticProfile::AppleSiliconOpenBlas,
            ] {
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
