use super::*;

#[test]
fn exact_more_thuente_warning_restores_and_accepts_only_at_task_width() {
    let best = f64::from_bits(0x4066_9849_a456_50ea);
    let narrow_other = f64::from_bits(0x4064_989a_67d2_0d2b);
    let wider_other = f64::from_bits(0x4063_5766_db0c_a2ee);

    assert!(more_thuente_restores_best(
        true,
        false,
        best,
        narrow_other,
        true
    ));
    assert!(more_thuente_accepts_warning(
        true,
        false,
        best,
        narrow_other,
        true
    ));
    assert!(!more_thuente_restores_best(
        true,
        false,
        best,
        wider_other,
        true
    ));
    assert!(!more_thuente_accepts_warning(
        true,
        false,
        best,
        wider_other,
        true
    ));

    assert!(!more_thuente_accepts_warning(
        true,
        true,
        best,
        wider_other,
        true
    ));
    assert!(more_thuente_accepts_warning(
        false,
        true,
        best,
        wider_other,
        true
    ));
    assert!(!more_thuente_accepts_warning(
        false,
        false,
        best,
        narrow_other,
        true
    ));
    assert!(!more_thuente_restores_best(
        true,
        false,
        best,
        narrow_other,
        false
    ));
}

#[test]
fn line_search_candidate_copies_exact_unit_step_target() {
    let x = [-25.060655267031027];
    let target = [30.0];
    let direction = [target[0] - x[0]];
    let recomputed = bounded_step(
        ArithmeticProfile::legacy(),
        &x,
        &direction,
        1.0,
        &[-30.0],
        &[30.0],
    );
    assert_ne!(recomputed[0].to_bits(), target[0].to_bits());

    let candidate = line_search_candidate(
        &LineSearchRequest {
            profile: ArithmeticProfile::legacy(),
            x: &x,
            value: 0.0,
            gradient: &[-1.0],
            direction: &direction,
            unit_step_target: Some(&target),
            lower: &[-30.0],
            upper: &[30.0],
            max_step_cap: None,
            cap_initial_unbounded_step: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            exact_operations: false,
        },
        1.0,
    );

    assert_eq!(candidate[0].to_bits(), target[0].to_bits());
}

#[test]
fn exact_line_search_candidate_keeps_feasible_bound_roundoff() {
    fn replay(x_value: u64, direction_value: u64, alpha: u64) -> (u64, u64) {
        let x = [f64::from_bits(x_value)];
        let direction = [f64::from_bits(direction_value)];
        let lower = [-30.0];
        let upper = [30.0];
        let candidate = |exact_operations| {
            line_search_candidate(
                &LineSearchRequest {
                    profile: ArithmeticProfile::legacy(),
                    x: &x,
                    value: 0.0,
                    gradient: &[-1.0],
                    direction: &direction,
                    unit_step_target: None,
                    lower: &lower,
                    upper: &upper,
                    max_step_cap: None,
                    cap_initial_unbounded_step: false,
                    allow_retry_warning_accept: false,
                    min_step: MAIN_PATH_MIN_STEP,
                    exact_operations,
                },
                f64::from_bits(alpha),
            )[0]
            .to_bits()
        };
        (candidate(true), candidate(false))
    }

    // R 4.6.0 black-box traces, unchanged under R 4.6.2: synthetic cases
    // 0135/call 93 and 0370/call
    // 109. The safeguarded scalar is already exact; the affine update retains
    // the one- or two-ULP bound overshoot, while the tolerant path clamps it.
    let (case_0135_exact, case_0135_clamped) = replay(
        0x403e_0000_0000_0000,
        0xc018_8f82_b6b9_8f84,
        0x4023_8b23_327e_4338,
    );
    assert_eq!(case_0135_exact, 0xc03e_0000_0000_0002);
    assert_eq!(case_0135_clamped, (-30.0_f64).to_bits());

    let (case_0370_exact, case_0370_clamped) = replay(
        0x4032_2977_b96a_9ed3,
        0xc006_4fa0_78dd_4b90,
        0x4031_44f0_338c_48bb,
    );
    assert_eq!(case_0370_exact, 0xc03e_0000_0000_0001);
    assert_eq!(case_0370_clamped, (-30.0_f64).to_bits());
}

#[test]
fn more_thuente_accepts_r_max_step_warning_case() {
    let mut problem = LinearDescent;
    let mut counts = OptimCounts::default();
    let step = more_thuente_line_search(
        &mut problem,
        LineSearchRequest {
            profile: ArithmeticProfile::legacy(),
            x: &[0.0],
            value: 0.0,
            gradient: &[-1.0],
            direction: &[1.0],
            unit_step_target: None,
            lower: &[f64::NEG_INFINITY],
            upper: &[10.0],
            max_step_cap: Some(1.0),
            cap_initial_unbounded_step: false,
            allow_retry_warning_accept: false,
            min_step: MAIN_PATH_MIN_STEP,
            exact_operations: false,
        },
        &mut counts,
    )
    .expect("More-Thuente should not error")
    .expect("R accepts STPMAX warning as a new iterate");

    assert_eq!(counts.function, 1);
    assert_eq!(counts.gradient, 1);
    assert_eq!(step.alpha, 1.0);
    assert_eq!(step.max_alpha, 1.0);
    assert!(!step.wolfe_curvature_satisfied);
    assert_vec_close(&step.x, &[1.0], 1e-15);
}

#[test]
fn cauchy_more_thuente_full_probe_matches_r_trace_prefix_and_counts() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    let mut backend = NativeBackend;
    let mut problem = RecordingRosenbrockProblem::default();
    let result = backend
        .minimize(
            &mut problem,
            &[-1.2, 1.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            BackendControl {
                profile: ArithmeticProfile::legacy(),
                maxit: 100,
                factr: 1e7,
                pgtol: 0.0,
                lmm: 5,
                trace: 0,
                report: 10,
            },
        )
        .expect("Rosenbrock backend optimization");

    assert_eq!(result.convergence, 0, "{result:?}");
    assert_eq!(result.counts.function, 48, "{result:?}");
    assert_eq!(result.counts.function, result.counts.gradient, "{result:?}");
    assert!(
        problem.points.len() >= 8,
        "expected at least the first third-iteration trial: {:?}",
        problem.points
    );
    for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-10);
    }
}
