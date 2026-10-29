use super::*;

const PROFILES: [ArithmeticProfile; 2] = [
    ArithmeticProfile::X86OpenBlas,
    ArithmeticProfile::AppleSiliconOpenBlas,
];

fn quadratic_request<'a>(
    profile: ArithmeticProfile,
    direction: &'a [f64],
    upper: &'a [f64],
) -> LineSearchRequest<'a> {
    LineSearchRequest {
        profile,
        x: &[0.0],
        value: 4.0,
        gradient: &[-4.0],
        direction,
        unit_step_target: None,
        lower: &[f64::NEG_INFINITY],
        upper,
        max_step_cap: None,
        cap_initial_unbounded_step: false,
        allow_retry_warning_accept: false,
        min_step: MAIN_PATH_MIN_STEP,
        exact_operations: true,
    }
}

fn quadratic_search(profile: ArithmeticProfile, direction: f64, upper: f64) -> (Step, OptimCounts) {
    let mut counts = OptimCounts::default();
    let step = more_thuente_line_search(
        &mut OneDimensionalQuadratic,
        quadratic_request(profile, &[direction], &[upper]),
        &mut counts,
    )
    .expect("quadratic evaluation")
    .expect("bounded quadratic has an admissible step");
    assert_eq!(counts.function, counts.gradient);
    assert_eq!(step.line_search_trials + 1, counts.function);
    (step, counts)
}

fn assert_quadratic_wolfe(step: &Step, direction: f64) {
    // Evaluate the conditions from the known polynomial, independently of the
    // search's cached derivatives and curvature diagnostics.
    let value = (step.x[0] - 2.0).powi(2);
    let initial_derivative = -4.0 * direction;
    let final_derivative = 2.0 * (step.x[0] - 2.0) * direction;
    assert!(value <= 4.0 + 1e-3 * step.alpha * initial_derivative);
    assert!(final_derivative.abs() <= 0.9 * initial_derivative.abs());
    assert_eq!(step.acceptance, StepAcceptance::StrongWolfe);
    assert!(step.wolfe_curvature_satisfied);
    assert_eq!(step.value, value);
}

#[test]
fn production_search_accepts_a_full_step_without_extra_trials() {
    for profile in PROFILES {
        let (step, counts) = quadratic_search(profile, 1.0, f64::INFINITY);
        assert_eq!(counts.function, 1);
        assert_eq!(step.alpha, 1.0);
        assert_eq!(step.x, vec![1.0]);
        assert_quadratic_wolfe(&step, 1.0);
    }
}

#[test]
fn production_search_contracts_an_overlarge_trial() {
    for profile in PROFILES {
        let (step, counts) = quadratic_search(profile, 10.0, f64::INFINITY);
        assert!(counts.function >= 2);
        assert!(step.alpha > 0.0 && step.alpha < 1.0);
        assert!(step.value < 4.0);
        assert_quadratic_wolfe(&step, 10.0);
    }
}

#[test]
fn production_search_expands_a_step_that_has_insufficient_curvature() {
    // At alpha=1, x=0.1 gives derivative ratio 0.95, above the 0.9 limit.
    for profile in PROFILES {
        let (step, counts) = quadratic_search(profile, 0.1, 1.0);
        assert!(counts.function >= 2);
        assert!(step.alpha > 1.0);
        assert!(step.x[0] >= 0.2 && step.x[0] <= 1.0);
        assert_quadratic_wolfe(&step, 0.1);
    }
}

#[test]
fn production_search_reports_a_feasible_cap_with_insufficient_curvature() {
    for profile in PROFILES {
        let (step, counts) = quadratic_search(profile, 1.0, 0.1);
        assert_eq!(counts.function, 1);
        assert_eq!(step.x, vec![0.1]);
        assert_eq!(step.alpha, 0.1);
        assert_eq!(step.acceptance, StepAcceptance::StepLimit);
        assert!(!step.wolfe_curvature_satisfied);
        let derivative = 2.0 * (step.x[0] - 2.0);
        assert!(derivative.abs() > 0.9 * 4.0);
        assert!(step.value <= 4.0 - 1e-3 * step.alpha * 4.0);
    }
}

#[test]
fn production_search_propagates_an_evaluation_error_without_counting_success() {
    for profile in PROFILES {
        let mut counts = OptimCounts::default();
        let error = more_thuente_line_search(
            &mut AlwaysNonFiniteObjective,
            quadratic_request(profile, &[1.0], &[f64::INFINITY]),
            &mut counts,
        )
        .expect_err("objective failure must propagate");
        assert!(matches!(error, OptimError::NonFiniteObjective { value } if value.is_nan()));
        assert_eq!(counts.function, 0);
        assert_eq!(counts.gradient, 0);
    }
}
