use super::*;

#[test]
fn more_thuente_stage_two_waits_for_nonnegative_directional_derivative() {
    assert!(!more_thuente_enters_stage_two(true, 0.0, 0.0, -1e-12));
    assert!(more_thuente_enters_stage_two(true, 0.0, 0.0, 0.0));
    assert!(!more_thuente_enters_stage_two(false, 0.0, 0.0, 1.0));
}

#[test]
fn more_thuente_retry_warning_requires_narrow_bracket() {
    assert!(more_thuente_retry_warning(10.0, 14.9, true));
    assert!(!more_thuente_retry_warning(10.0, 21.0, true));
    assert!(!more_thuente_retry_warning(10.0, 14.9, false));
    assert!(!more_thuente_retry_warning(0.0, 0.0, true));
}

#[test]
fn more_thuente_retry_restore_requires_the_task_warning_width() {
    assert!(more_thuente_retry_restore(10.0, 11.0, true));
    assert!(!more_thuente_retry_restore(10.0, 11.2, true));
    assert!(!more_thuente_retry_restore(10.0, 11.0, false));
    assert!(!more_thuente_retry_restore(0.0, 0.0, true));
}

#[test]
fn more_thuente_cubic_steps_are_mirror_symmetric() {
    let right_best = MoreThuentePoint {
        alpha: 0.0,
        value: 0.0,
        derivative: -1.0,
    };
    let left_best = MoreThuentePoint {
        alpha: 0.0,
        value: 0.0,
        derivative: 1.0,
    };

    assert_mirrored_steps(
        more_thuente_cubic_case_two(
            ArithmeticProfile::legacy(),
            right_best,
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.25,
                derivative: 0.2,
            },
        ),
        more_thuente_cubic_case_two(
            ArithmeticProfile::legacy(),
            left_best,
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.25,
                derivative: -0.2,
            },
        ),
    );

    assert_mirrored_steps(
        more_thuente_cubic_case_three(
            ArithmeticProfile::legacy(),
            right_best,
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.5,
                derivative: -0.1,
            },
        ),
        more_thuente_cubic_case_three(
            ArithmeticProfile::legacy(),
            left_best,
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.5,
                derivative: 0.1,
            },
        ),
    );

    assert_mirrored_steps(
        more_thuente_cubic_case_four(
            ArithmeticProfile::legacy(),
            MoreThuentePoint {
                alpha: 2.0,
                value: 0.5,
                derivative: 0.7,
            },
            MoreThuentePoint {
                alpha: 1.0,
                value: -0.25,
                derivative: -1.2,
            },
        ),
        more_thuente_cubic_case_four(
            ArithmeticProfile::legacy(),
            MoreThuentePoint {
                alpha: -2.0,
                value: 0.5,
                derivative: -0.7,
            },
            MoreThuentePoint {
                alpha: -1.0,
                value: -0.25,
                derivative: 1.2,
            },
        ),
    );
}

#[test]
fn more_thuente_case_three_respects_exact_feasible_step_cap() {
    fn replay(
        x_value: u64,
        direction_value: u64,
        best: MoreThuentePoint,
        trial: MoreThuentePoint,
        expected: u64,
    ) {
        let x = [f64::from_bits(x_value)];
        let direction = [f64::from_bits(direction_value)];
        let step_cap = geometric_feasible_step(&x, &direction, &[-30.0], &[30.0]);
        assert_eq!(step_cap.to_bits(), expected);

        let cubic = more_thuente_cubic_case_three(ArithmeticProfile::legacy(), best, trial)
            .expect("case-three cubic step");
        assert!(cubic > step_cap);
        let mut state = MoreThuenteStepState {
            best,
            other: best,
            bracketed: false,
        };
        let next = more_thuente_step(
            ArithmeticProfile::legacy(),
            &mut state,
            trial,
            0.0,
            step_cap,
        );
        assert_eq!(next.to_bits(), expected);
    }

    replay(
        0x403e_0000_0000_0000,
        0xc018_8f82_b6b9_8f84,
        MoreThuentePoint {
            alpha: f64::from_bits(0x3ff0_0000_0000_0000),
            value: f64::from_bits(0x4093_7fd6_92a2_9b17),
            derivative: f64::from_bits(0xc06a_9aea_9c7c_d910),
        },
        MoreThuentePoint {
            alpha: f64::from_bits(0x4014_0000_0000_0000),
            value: f64::from_bits(0x4079_0b70_fbbe_3573),
            derivative: f64::from_bits(0xc068_88b4_4079_5f3f),
        },
        0x4023_8b23_327e_4338,
    );

    replay(
        0x4032_2977_b96a_9ed3,
        0xc006_4fa0_78dd_4b90,
        MoreThuentePoint {
            alpha: f64::from_bits(0x3ff0_0000_0000_0000),
            value: f64::from_bits(0x4060_74e7_ac0b_b1fb),
            derivative: f64::from_bits(0xc026_eaa7_7487_2309),
        },
        MoreThuentePoint {
            alpha: f64::from_bits(0x4014_0000_0000_0000),
            value: f64::from_bits(0x4055_74f7_76e2_5e47),
            derivative: f64::from_bits(0xc026_e2bd_8334_f324),
        },
        0x4031_44f0_338c_48bb,
    );
}

#[test]
fn fourth_iteration_more_thuente_extrapolates_to_r_trace_step() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let lower = vec![-5.0, -5.0];
    let upper = vec![5.0, 5.0];
    let mut problem = RosenbrockProblem;
    let x0 = fixture.first_points[0].clone();
    let x1 = fixture.first_points[3].clone();
    let x2 = fixture.first_points[4].clone();
    let x3 = fixture.first_points[5].clone();
    let g0 = problem.gradient(&x0).expect("Rosenbrock gradient");
    let g1 = problem.gradient(&x1).expect("Rosenbrock gradient");
    let g2 = problem.gradient(&x2).expect("Rosenbrock gradient");
    let g3 = problem.gradient(&x3).expect("Rosenbrock gradient");
    let mut history = Vec::new();

    update_history(
        &mut history,
        5,
        difference(&x1, &x0),
        difference(&g1, &g0),
        &g0,
    );
    update_history(
        &mut history,
        5,
        difference(&x2, &x1),
        difference(&g2, &g1),
        &g1,
    );
    update_history(
        &mut history,
        5,
        difference(&x3, &x2),
        difference(&g3, &g2),
        &g2,
    );

    let cauchy = generalized_cauchy_point_limited_memory(&x3, &g3, &lower, &upper, &history);
    let target = subspace_minimizer_limited_memory(&x3, &g3, &lower, &upper, &history, &cauchy)
        .expect("fourth iteration positive-definite model");
    let direction = difference(&target.x, &x3);
    let value = problem.value(&x3).expect("Rosenbrock value");
    let mut recording = RecordingRosenbrockProblem::default();
    let mut counts = OptimCounts::default();
    let step = more_thuente_line_search(
        &mut recording,
        LineSearchRequest {
            profile: ArithmeticProfile::legacy(),
            x: &x3,
            value,
            gradient: &g3,
            direction: &direction,
            unit_step_target: None,
            lower: &lower,
            upper: &upper,
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            exact_operations: false,
        },
        &mut counts,
    )
    .expect("line search should not error")
    .expect("line search should accept the R trace step");
    assert_eq!(history.len(), 3);
    assert_eq!(counts.function, 5);
    assert_eq!(counts.gradient, 5);
    assert_eq!(step.line_search_trials, 4);
    for (actual, expected) in recording
        .points
        .iter()
        .zip(fixture.first_points.iter().skip(6))
    {
        assert_vec_close(actual, expected, 1e-10);
    }
    assert_vec_close(&step.x, &fixture.first_points[10], 1e-10);
}

#[test]
fn interpolated_brackets_enclose_independent_convex_minima() {
    // Check geometric invariants against independently known minimizers rather
    // than asserting the interpolation helper's own formula as its expectation.
    let mut checked = 0;
    for center in [0.03125_f64, 0.125, 0.75, 2.0, 19.0] {
        for initial in [0.01_f64, 0.1, 1.0, 10.0, 100.0] {
            for quartic in [false, true] {
                let phi = |alpha: f64| {
                    let d = alpha - center;
                    MoreThuentePoint {
                        alpha,
                        value: if quartic {
                            d.powi(4) + 0.01 * d * d
                        } else {
                            d * d
                        },
                        derivative: if quartic {
                            4.0 * d.powi(3) + 0.02 * d
                        } else {
                            2.0 * d
                        },
                    }
                };
                let mut state = MoreThuenteStepState {
                    best: phi(0.0),
                    other: phi(0.0),
                    bracketed: false,
                };
                let mut alpha = initial;
                for _ in 0..20 {
                    if phi(alpha).derivative.abs() <= 1e-12 || alpha == state.best.alpha {
                        break;
                    }
                    let old_interval = state.bracketed.then(|| {
                        (
                            state.best.alpha.min(state.other.alpha),
                            state.best.alpha.max(state.other.alpha),
                        )
                    });
                    let next = more_thuente_step(
                        ArithmeticProfile::legacy(),
                        &mut state,
                        phi(alpha),
                        0.0,
                        1e4,
                    );
                    checked += 1;
                    assert!(next.is_finite() && (0.0..=1e4).contains(&next));
                    if state.bracketed {
                        let low = state.best.alpha.min(state.other.alpha);
                        let high = state.best.alpha.max(state.other.alpha);
                        let tolerance = 1e-12 * high.max(1.0);
                        assert!(
                            low <= center + tolerance && high >= center - tolerance,
                            "minimum {center} outside [{low}, {high}]"
                        );
                        assert!(
                            next >= low - tolerance && next <= high + tolerance,
                            "trial {next} outside [{low}, {high}]"
                        );
                        if let Some((old_low, old_high)) = old_interval {
                            assert!(low >= old_low - tolerance && high <= old_high + tolerance);
                        }
                    }
                    alpha = next;
                }
            }
        }
    }
    assert!(checked >= 100, "too few bracket updates: {checked}");
}

#[test]
#[allow(clippy::excessive_precision)] // Preserve independent native-probe decimal round trips.
fn cubic_trials_preserve_reference_affine_contraction() {
    // Independent evaluation of the published cubic formula with contraction
    // enabled/disabled in the native reference compiler produces these bits.
    let best = MoreThuentePoint {
        alpha: 5.205847622847678,
        value: 1607.180651510217,
        derivative: -105.20576784510523,
    };
    let trial = MoreThuentePoint {
        alpha: 15.629988191768211,
        value: 48684.11983223793,
        derivative: 21387.900169328255,
    };
    for (profile, expected) in [
        (ArithmeticProfile::X86OpenBlas, 7.4644440618718697_f64),
        (
            ArithmeticProfile::AppleSiliconOpenBlas,
            7.4644440618718679_f64,
        ),
    ] {
        let mut state = MoreThuenteStepState {
            best,
            other: MoreThuentePoint {
                alpha: 21.0,
                value: 297786.65788028465,
                derivative: 54024.02073409168,
            },
            bracketed: true,
        };
        assert_eq!(
            more_thuente_step(profile, &mut state, trial, best.alpha, 21.0).to_bits(),
            expected.to_bits(),
        );
    }
}
