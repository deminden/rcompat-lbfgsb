//! Repeatable optimizer/callback cost probe; outputs CSV without runtime dependencies.
use rcompat_lbfgsb::{Bounds, OptimControl, optim_lbfgsb, optim_lbfgsb_with_gradient};
use std::cell::Cell;
use std::time::Instant;

fn main() {
    // cargo test --all-targets must not execute the measurement workload.
    if !std::env::args().any(|arg| arg == "--bench") {
        return;
    }
    let dimensions: Vec<usize> = std::env::var("LBFGSB_BENCH_DIMENSIONS")
        .unwrap_or_else(|_| "16,128,512".to_owned())
        .split(',')
        .map(|v| v.parse().expect("positive dimensions"))
        .collect();
    println!(
        "dimension,lmm,route,callback_work,trace,elapsed_ns,objective_ns,raw_objective_calls,optimizer_calls"
    );
    for n in dimensions {
        assert!(n > 0);
        for lmm in [1, 5, 10] {
            for route in [
                "finite_gradient",
                "finite_difference",
                "mixed_gradient",
                "unbounded_gradient",
            ] {
                for work in [1, 16] {
                    for trace in [0, 2] {
                        let mut samples = Vec::new();
                        for _ in 0..5 {
                            let mut c = OptimControl::default_for_dimension(n);
                            c.lmm = lmm;
                            c.maxit = 2;
                            c.trace = trace;
                            c.report = 0;
                            let bounds = match route {
                                "unbounded_gradient" => Bounds::unbounded(n),
                                "mixed_gradient" => {
                                    let mut b = Bounds::new(vec![-10.0; n], vec![10.0; n]).unwrap();
                                    b.lower[0] = f64::NEG_INFINITY;
                                    b
                                }
                                _ => Bounds::new(vec![-10.0; n], vec![10.0; n]).unwrap(),
                            };
                            let callback_ns = Cell::new(0_u128);
                            let raw_calls = Cell::new(0_usize);
                            let objective = |x: &[f64]| {
                                let start = Instant::now();
                                let mut value = 0.0;
                                for _ in 0..work {
                                    value = std::hint::black_box(x)
                                        .iter()
                                        .enumerate()
                                        .map(|(i, v)| {
                                            (1.0 + i as f64 / n as f64) * (v - 1.0).powi(2)
                                        })
                                        .sum();
                                    std::hint::black_box(value);
                                }
                                callback_ns.set(callback_ns.get() + start.elapsed().as_nanos());
                                raw_calls.set(raw_calls.get() + 1);
                                value
                            };
                            let gradient = |x: &[f64]| {
                                x.iter()
                                    .enumerate()
                                    .map(|(i, v)| 2.0 * (1.0 + i as f64 / n as f64) * (v - 1.0))
                                    .collect()
                            };
                            let start = Instant::now();
                            let result = if route == "finite_difference" {
                                optim_lbfgsb(vec![2.0; n], bounds, objective, c)
                            } else {
                                optim_lbfgsb_with_gradient(
                                    vec![2.0; n],
                                    bounds,
                                    objective,
                                    gradient,
                                    c,
                                )
                            }
                            .unwrap();
                            let elapsed = start.elapsed().as_nanos();
                            assert!(
                                result.value.is_finite()
                                    && result.par.iter().all(|v| v.is_finite())
                            );
                            samples.push((
                                elapsed,
                                callback_ns.get(),
                                raw_calls.get(),
                                result.counts.function,
                            ));
                        }
                        samples.sort_unstable();
                        let (elapsed, callback, raw, optimizer) = samples[2];
                        println!(
                            "{n},{lmm},{route},{work},{trace},{elapsed},{callback},{raw},{optimizer}"
                        );
                    }
                }
            }
        }
    }
}
