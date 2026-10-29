//! Independent closed-form constrained optima over dimensions and bound routes.
use rcompat_lbfgsb::*;
use std::cell::Cell;

#[test]
fn constant_objectives_preserve_projected_parameters_and_initial_evaluations() {
    let mut executed = 0;
    for n in [1, 2, 11, 128] {
        for profile in [
            ArithmeticProfile::X86OpenBlas,
            ArithmeticProfile::AppleSiliconOpenBlas,
        ] {
            for route in 0..3 {
                for numeric in [false, true] {
                    for fnscale in [2.0, -2.0] {
                        let lower = vec![if route == 0 { f64::NEG_INFINITY } else { -1.0 }; n];
                        let upper: Vec<_> = (0..n)
                            .map(|i| {
                                if route == 2 || (route == 1 && i % 2 == 0) {
                                    1.0
                                } else {
                                    f64::INFINITY
                                }
                            })
                            .collect();
                        let initial: Vec<f64> = (0..n)
                            .map(|i| if i % 2 == 0 { -3.0 } else { 3.0 })
                            .collect();
                        let expected: Vec<_> = (0..n)
                            .map(|i| initial[i].clamp(lower[i], upper[i]))
                            .collect();
                        let bounds = Bounds::new(lower, upper).unwrap();
                        let mut control = OptimControl::default_for_dimension(n);
                        control.maxit = 0;
                        control.factr = 0.0;
                        control.fnscale = fnscale;
                        control.parscale =
                            (0..n).map(|i| if i % 2 == 0 { 0.5 } else { 2.0 }).collect();
                        let raw_f = Cell::new(0);
                        let raw_g = Cell::new(0);
                        let f = |x: &[f64]| {
                            assert_eq!(x.len(), n);
                            raw_f.set(raw_f.get() + 1);
                            42.0
                        };
                        let g = |x: &[f64]| {
                            raw_g.set(raw_g.get() + 1);
                            vec![0.0; x.len()]
                        };
                        let result = if numeric {
                            optim_lbfgsb_with_profile(initial, bounds, f, control, profile)
                        } else {
                            optim_lbfgsb_with_gradient_and_profile(
                                initial, bounds, f, g, control, profile,
                            )
                        }
                        .unwrap();
                        assert_eq!(
                            result.par.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                        assert_eq!(result.value.to_bits(), 42.0_f64.to_bits());
                        assert_eq!(result.convergence, 0);
                        assert_eq!(
                            result.message,
                            TerminationReason::ProjectedGradient.message()
                        );
                        assert_eq!(
                            result.counts,
                            OptimCounts {
                                function: 1,
                                gradient: 1
                            }
                        );
                        assert_eq!(raw_f.get(), if numeric { 2 * n + 1 } else { 1 });
                        assert_eq!(raw_g.get(), if numeric { 0 } else { 1 });
                        executed += 1;
                    }
                }
            }
        }
    }
    assert_eq!(executed, 96);
}

#[test]
fn diagonal_quadratics_reach_known_box_solutions_across_routes() {
    let mut executed = 0;
    for n in [1, 2, 4, 9, 10, 11, 13, 128, 1024] {
        for profile in [
            ArithmeticProfile::X86OpenBlas,
            ArithmeticProfile::AppleSiliconOpenBlas,
        ] {
            for route in 0..3 {
                for lmm in [1, 5] {
                    for scale in [1e-8, 1.0, 1e8] {
                        for numeric in [false, true] {
                            if numeric && n > 13 {
                                continue;
                            }
                            executed += 1;
                            let mut lower = vec![f64::NEG_INFINITY; n];
                            let mut upper = vec![f64::INFINITY; n];
                            if route != 0 {
                                for i in 0..n {
                                    lower[i] = -1.0;
                                    if route == 2 || i % 2 == 0 {
                                        upper[i] = 2.0;
                                    }
                                }
                            }
                            let target: Vec<f64> = (0..n)
                                .map(|i| if i % 2 == 0 { 3.0 } else { -2.0 })
                                .collect();
                            let expected: Vec<f64> = (0..n)
                                .map(|i| target[i].clamp(lower[i], upper[i]))
                                .collect();
                            let initial: Vec<f64> =
                                (0..n).map(|i| if i % 2 == 0 { 0.5 } else { 0.0 }).collect();
                            let bounds = Bounds::new(lower, upper).unwrap();
                            let f = |x: &[f64]| {
                                let mut sum = 0.0;
                                for i in 0..n {
                                    let d = x[i] - target[i];
                                    sum += (1_u32 << (i % 3)) as f64 * d * d;
                                }
                                sum * scale
                            };
                            let g = |x: &[f64]| {
                                (0..n)
                                    .map(|i| {
                                        2.0 * (1_u32 << (i % 3)) as f64 * (x[i] - target[i]) * scale
                                    })
                                    .collect()
                            };
                            let mut c = OptimControl::default_for_dimension(n);
                            c.lmm = lmm;
                            c.fnscale = scale;
                            c.factr = 1e3;
                            c.pgtol = 1e-7;
                            c.maxit = 1000;
                            c.ndeps = vec![1e-5; n];
                            let result = if numeric {
                                optim_lbfgsb_with_profile(initial, bounds, f, c, profile)
                            } else {
                                optim_lbfgsb_with_gradient_and_profile(
                                    initial, bounds, f, g, c, profile,
                                )
                            }
                            .unwrap();
                            let error = result
                                .par
                                .iter()
                                .zip(&expected)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0_f64, f64::max);
                            assert!(
                                result.is_success() && error < 1e-5,
                                "n={n}, route={route}, lmm={lmm}, scale={scale}, numeric={numeric}, profile={profile:?}: error={error}; {result:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(executed, 576);
}
