use super::*;
#[test]
fn initial_stationarity_preserves_evaluation_counts_and_event_order() {
    for profile in [
        ArithmeticProfile::X86OpenBlas,
        ArithmeticProfile::AppleSiliconOpenBlas,
    ] {
        for numeric in [false, true] {
            for (slope, tolerance) in [(0.0, 0.0), (0.001, 0.01)] {
                let calls = RefCell::new(Vec::new());
                let mut events = Vec::new();
                let f = |x: &[f64]| {
                    calls.borrow_mut().push(false);
                    x.iter().map(|v| v * v + slope * v).sum()
                };
                let g = |x: &[f64]| {
                    calls.borrow_mut().push(true);
                    x.iter().map(|v| 2.0 * v + slope).collect()
                };
                let mut control = OptimControl::default_for_dimension(2);
                control.parscale = vec![0.5, 2.0];
                control.pgtol = tolerance;
                let bounds =
                    Bounds::new(vec![-1.0, f64::NEG_INFINITY], vec![1.0, f64::INFINITY]).unwrap();
                let result = if numeric {
                    optim_lbfgsb_observed(vec![0.0; 2], bounds, f, control, profile, |e| {
                        events.push(e)
                    })
                } else {
                    optim_lbfgsb_with_gradient_observed(
                        vec![0.0; 2],
                        bounds,
                        f,
                        g,
                        control,
                        profile,
                        |e| events.push(e),
                    )
                }
                .unwrap();
                assert_eq!(result.par, vec![0.0; 2]);
                assert_eq!(result.value.to_bits(), 0.0_f64.to_bits());
                assert_eq!(
                    result.counts,
                    OptimCounts {
                        function: 1,
                        gradient: 1
                    }
                );
                assert_eq!(result.convergence, 0);
                assert_eq!(
                    result.message,
                    TerminationReason::ProjectedGradient.message()
                );
                assert_eq!(
                    *calls.borrow(),
                    if numeric {
                        vec![false; 5]
                    } else {
                        vec![false, true]
                    }
                );
                assert_eq!(events.len(), 2);
                assert!(
                    matches!(events[0], OptimEvent::Started {projected_gradient,counts,..} if projected_gradient <= tolerance && counts == result.counts)
                );
                assert!(
                    matches!(events[1], OptimEvent::Finished {iterations:0,reason:TerminationReason::ProjectedGradient,counts} if counts == result.counts)
                );
            }
        }
    }
}

#[test]
fn typed_termination_covers_iteration_limit_and_unrepresentable_target() {
    for (initial, maxit, reason) in [
        (0.0, 0, TerminationReason::IterationLimit),
        (4.0, 100, TerminationReason::AbnormalLineSearch),
    ] {
        let mut events = Vec::new();
        let mut c = OptimControl::default_for_dimension(1);
        c.maxit = maxit;
        c.factr = 0.0;
        let result = optim_lbfgsb_with_gradient_observed(
            vec![initial],
            Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
            |x| 1.0 - 1e-20 * x[0],
            |_| vec![-1e-20],
            c,
            ArithmeticProfile::AppleSiliconOpenBlas,
            |e| events.push(e),
        )
        .unwrap();
        assert_eq!(result.convergence, reason.convergence());
        assert_eq!(result.message, reason.message());
        assert!(matches!(events.last(),Some(OptimEvent::Finished{reason:r,..}) if *r==reason));
    }
}

#[test]
fn observer_can_run_an_independent_nested_optimization() {
    let expected = replay(false, false, ArithmeticProfile::X86OpenBlas, 0).result;
    let mut first = true;
    let result = optim_lbfgsb_with_gradient_observed(
        vec![-1.2, 1.0],
        Bounds::new(vec![-5.0; 2], vec![5.0; 2]).unwrap(),
        |x| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
        |x| {
            vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ]
        },
        OptimControl::default_for_dimension(2),
        ArithmeticProfile::X86OpenBlas,
        |_| {
            if first {
                first = false;
                let nested = replay(false, true, ArithmeticProfile::AppleSiliconOpenBlas, 1);
                assert!(nested.result.value.is_finite());
            }
        },
    )
    .unwrap();
    assert_eq!(result, expected);
}

#[test]
fn history_skips_and_line_search_restarts_are_explicit() {
    let mut linear_events = Vec::new();
    let mut c = OptimControl::default_for_dimension(2);
    c.maxit = 1;
    c.factr = 0.0;
    optim_lbfgsb_with_gradient_observed(
        vec![0.0; 2],
        Bounds::new(vec![-10.0; 2], vec![10.0; 2]).unwrap(),
        |x| -x.iter().sum::<f64>(),
        |_| vec![-1.0; 2],
        c,
        ArithmeticProfile::X86OpenBlas,
        |e| linear_events.push(e),
    )
    .unwrap();
    assert!(linear_events.iter().any(|e| matches!(
        e,
        OptimEvent::History {
            outcome: HistoryUpdate::SkippedCurvature,
            ..
        }
    )));

    // A stateful callback becomes incompatible with the previously accepted
    // objective after the first step. Recovery must clear memory exactly once.
    let mut calls = 0;
    let mut events = Vec::new();
    let result = optim_lbfgsb_with_gradient_observed(
        vec![4.0, -5.0],
        Bounds::unbounded(2),
        |x| {
            calls += 1;
            let offset = if calls > 2 { 1e6 } else { 0.0 };
            (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2) + offset
        },
        |x| vec![2.0 * (x[0] - 1.0), 4.0 * (x[1] + 2.0)],
        OptimControl::default_for_dimension(2),
        ArithmeticProfile::X86OpenBlas,
        |e| events.push(e),
    )
    .unwrap();
    assert_eq!(result.convergence, 52);
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(
                e,
                OptimEvent::MemoryReset {
                    reason: RecoveryReason::LineSearch,
                    ..
                }
            ))
            .count(),
        1
    );
    assert!(matches!(
        events.last(),
        Some(OptimEvent::Finished {
            iterations: 1,
            reason: TerminationReason::AbnormalLineSearch,
            ..
        })
    ));
}

#[test]
fn diagnostic_values_use_scaled_units() {
    let mut events = Vec::new();
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 2.0;
    c.parscale[0] = 4.0;
    let result = optim_lbfgsb_with_gradient_observed(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0]).unwrap(),
        |x| (x[0] - 2.0).powi(2),
        |x| vec![2.0 * (x[0] - 2.0)],
        c,
        ArithmeticProfile::X86OpenBlas,
        |e| events.push(e),
    )
    .unwrap();
    assert!(matches!(
        events.first(),
        Some(OptimEvent::Started {
            value: 2.0,
            projected_gradient: 2.5,
            ..
        })
    ));
    assert!((result.par[0] - 2.0).abs() < 1e-8);
    assert!(result.value < 1e-12);
}
