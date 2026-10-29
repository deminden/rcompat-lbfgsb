//! CSV observer overhead probe; run with cargo bench --bench observer_cost.
use rcompat_lbfgsb::*;
use std::cell::Cell;
use std::time::Instant;
fn main() {
    if !std::env::args().any(|a| a == "--bench") {
        return;
    }
    println!("dimension,profile,observed,median_ns,event_count");
    for n in [16, 128, 1024] {
        for profile in [
            ArithmeticProfile::X86OpenBlas,
            ArithmeticProfile::AppleSiliconOpenBlas,
        ] {
            let mut expected = None;
            for observed in [false, true] {
                let mut times = Vec::new();
                let event_count = Cell::new(0_usize);
                for _ in 0..31 {
                    event_count.set(0);
                    let f = |x: &[f64]| {
                        x.iter()
                            .enumerate()
                            .map(|(i, v)| (1.0 + i as f64 / n as f64) * (v - 1.0).powi(2))
                            .sum()
                    };
                    let g = |x: &[f64]| {
                        x.iter()
                            .enumerate()
                            .map(|(i, v)| 2.0 * (1.0 + i as f64 / n as f64) * (v - 1.0))
                            .collect()
                    };
                    let b = Bounds::new(vec![-10.0; n], vec![10.0; n]).unwrap();
                    let mut c = OptimControl::default_for_dimension(n);
                    c.maxit = 2;
                    let start = Instant::now();
                    let result = if observed {
                        optim_lbfgsb_with_gradient_observed(
                            vec![2.0; n],
                            b,
                            f,
                            g,
                            c,
                            profile,
                            |e| {
                                event_count.set(event_count.get() + 1);
                                std::hint::black_box(e);
                            },
                        )
                    } else {
                        optim_lbfgsb_with_gradient_and_profile(vec![2.0; n], b, f, g, c, profile)
                    }
                    .unwrap();
                    times.push(start.elapsed().as_nanos());
                    let bits = (
                        result.par.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        result.value.to_bits(),
                        result.counts,
                        result.convergence,
                        result.message,
                    );
                    if let Some(reference) = &expected {
                        assert_eq!(&bits, reference);
                    } else {
                        expected = Some(bits);
                    }
                }
                times.sort_unstable();
                println!(
                    "{n},{},{observed},{},{}",
                    profile.id(),
                    times[15],
                    event_count.get()
                );
            }
        }
    }
}
