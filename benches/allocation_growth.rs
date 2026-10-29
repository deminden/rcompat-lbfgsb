//! Fixed-budget allocation and scaling probe. No workload runs under cargo test.
use rcompat_lbfgsb::*;
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::alloc::System;
use std::cell::Cell;
use std::time::Instant;

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[derive(Debug, PartialEq, Eq)]
struct Evidence {
    result: (Vec<u64>, u64, OptimCounts, i32, String),
    allocations: usize,
    bytes: usize,
    reallocations: usize,
    callback_allocations: usize,
    callback_bytes: usize,
    raw_f: usize,
    raw_g: usize,
}

impl Evidence {
    fn result_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325_u64;
        let mut add = |byte: u8| {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        };
        for value in self.result.0.iter().copied().chain([
            self.result.1,
            self.result.2.function as u64,
            self.result.2.gradient as u64,
            self.result.3 as u64,
        ]) {
            for byte in value.to_le_bytes() {
                add(byte);
            }
        }
        for byte in self.result.4.bytes() {
            add(byte);
        }
        hash
    }
}

fn measure(
    n: usize,
    m: usize,
    route: &str,
    work: usize,
    profile: ArithmeticProfile,
) -> (Evidence, u128, u128, u128) {
    // Inputs and controls belong to the caller and are prepared before sampling.
    let initial = vec![2.0; n];
    let bounds = match route {
        "unbounded" => Bounds::unbounded(n),
        "mixed" => Bounds::new(
            (0..n)
                .map(|i| if i % 2 == 0 { -10.0 } else { f64::NEG_INFINITY })
                .collect(),
            (0..n)
                .map(|i| if i % 3 == 0 { 10.0 } else { f64::INFINITY })
                .collect(),
        )
        .unwrap(),
        _ => Bounds::new(vec![-10.0; n], vec![10.0; n]).unwrap(),
    };
    let mut control = OptimControl::default_for_dimension(n);
    control.lmm = m;
    control.maxit = 24;
    control.factr = 0.0;
    control.pgtol = 0.0;
    let weights: Vec<_> = (0..n).map(|i| 2.0_f64.powi((i % 13) as i32)).collect();
    let f_ns = Cell::new(0);
    let g_ns = Cell::new(0);
    let raw_f = Cell::new(0);
    let raw_g = Cell::new(0);
    let callback_allocations = Cell::new(0);
    let callback_bytes = Cell::new(0);
    let f = |x: &[f64]| {
        let start = Instant::now();
        let mut value = 0.0;
        for _ in 0..work {
            value = std::hint::black_box(x)
                .iter()
                .zip(&weights)
                .map(|(v, w)| 0.5 * w * (v - 1.0).powi(2))
                .sum();
            std::hint::black_box(value);
        }
        f_ns.set(f_ns.get() + start.elapsed().as_nanos());
        raw_f.set(raw_f.get() + 1);
        value
    };
    let g = |x: &[f64]| {
        let callback_region = Region::new(GLOBAL);
        let start = Instant::now();
        let mut gradient = Vec::with_capacity(n);
        for _ in 0..work {
            gradient.clear();
            gradient.extend(
                std::hint::black_box(x)
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| w * (v - 1.0)),
            );
            std::hint::black_box(&gradient);
        }
        g_ns.set(g_ns.get() + start.elapsed().as_nanos());
        raw_g.set(raw_g.get() + 1);
        let stats = callback_region.change();
        callback_allocations.set(callback_allocations.get() + stats.allocations);
        callback_bytes.set(callback_bytes.get() + stats.bytes_allocated);
        gradient
    };
    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let result = if route == "finite_difference" {
        optim_lbfgsb_with_profile(initial, bounds, f, control, profile)
    } else {
        optim_lbfgsb_with_gradient_and_profile(initial, bounds, f, g, control, profile)
    }
    .unwrap();
    let elapsed = start.elapsed().as_nanos();
    let stats = region.change();
    assert!(result.value.is_finite() && result.par.iter().all(|x| x.is_finite()));
    assert_eq!(result.convergence, 1, "fixed budget must be exhausted");
    let evidence = Evidence {
        result: (
            result.par.iter().map(|v| v.to_bits()).collect(),
            result.value.to_bits(),
            result.counts,
            result.convergence,
            result.message,
        ),
        allocations: stats.allocations,
        bytes: stats.bytes_allocated,
        reallocations: stats.reallocations,
        callback_allocations: callback_allocations.get(),
        callback_bytes: callback_bytes.get(),
        raw_f: raw_f.get(),
        raw_g: raw_g.get(),
    };
    (evidence, elapsed, f_ns.get(), g_ns.get())
}

fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    let dimensions: Vec<usize> = std::env::var("LBFGSB_BENCH_DIMENSIONS")
        .unwrap_or_else(|_| "32,128,512,2048".to_owned())
        .split(',')
        .map(|v| v.parse().expect("positive dimension"))
        .collect();
    let samples: usize = std::env::var("LBFGSB_BENCH_SAMPLES")
        .unwrap_or_else(|_| "5".to_owned())
        .parse()
        .unwrap();
    assert!(samples > 0);
    println!("dimension,lmm,route,callback_work,profile,elapsed_ns,objective_ns,gradient_ns,allocations,bytes_allocated,reallocations,callback_allocations,callback_bytes,raw_objective_calls,raw_gradient_calls,optimizer_function_calls,optimizer_gradient_calls,convergence,result_fingerprint");
    for n in dimensions {
        assert!(n > 0);
        for m in [1, 5, 10, 20] {
            for route in ["finite", "mixed", "unbounded", "finite_difference"] {
                // Numerical gradients cost O(n) callback evaluations; keep the
                // default profiling run small enough for routine reproduction.
                if route == "finite_difference" && n > 128 {
                    continue;
                }
                for work in [1, 16] {
                    for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
                        let (expected, ..) = measure(n, m, route, work, profile);
                        let mut times = Vec::with_capacity(samples);
                        for _ in 0..samples {
                            let (evidence, elapsed, f, g) = measure(n, m, route, work, profile);
                            assert_eq!(
                                evidence, expected,
                                "measurement changed outputs or allocation counts"
                            );
                            times.push((elapsed, f, g));
                        }
                        times.sort_unstable();
                        let (elapsed, f, g) = times[samples / 2];
                        println!("{n},{m},{route},{work},{},{elapsed},{f},{g},{},{},{},{},{},{},{},{},{},{},{:016x}",profile.id(),expected.allocations,expected.bytes,expected.reallocations,expected.callback_allocations,expected.callback_bytes,expected.raw_f,expected.raw_g,expected.result.2.function,expected.result.2.gradient,expected.result.3,expected.result_fingerprint());
                    }
                }
            }
        }
    }
}
