use super::*;

#[test]
fn rosenbrock_default_evaluation_trace_matches_r_prefix() {
    let fixture: TraceFixture = serde_json::from_str(include_str!(
        "../../../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
    ))
    .expect("R trace fixture JSON");
    assert_eq!(fixture.fixture, "rosenbrock_default_trace");
    assert_eq!(fixture.source_fixture, "rosenbrock");
    assert_eq!(fixture.trace_kind, "function_evaluation_points");

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

    assert_eq!(result.counts.function, 48, "{result:?}");
    assert_eq!(result.counts.gradient, 48, "{result:?}");
    assert!(
        problem.points.len() >= fixture.first_points.len(),
        "{:?}",
        problem.points
    );

    for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
        assert_vec_close(actual, expected, 1e-10);
    }
}
