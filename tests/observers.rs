//! Observer contracts are checked against externally observable callback behavior.
use rcompat_lbfgsb::*;
use std::cell::RefCell;

#[derive(Debug, PartialEq)]
struct Replay {
    result: OptimResult,
    calls: Vec<(bool, Vec<f64>)>,
    events: Vec<OptimEvent>,
}
fn replay(observed: bool, numeric: bool, profile: ArithmeticProfile, route: usize) -> Replay {
    let calls = RefCell::new(Vec::new());
    let mut events = Vec::new();
    let f = |x: &[f64]| {
        calls.borrow_mut().push((false, x.to_vec()));
        100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2)
    };
    let g = |x: &[f64]| {
        calls.borrow_mut().push((true, x.to_vec()));
        vec![
            -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ]
    };
    let mut b = Bounds::new(vec![-5.0; 2], vec![5.0; 2]).unwrap();
    if route == 1 {
        b.lower[0] = f64::NEG_INFINITY;
    } else if route == 2 {
        b = Bounds::unbounded(2);
    }
    let mut c = OptimControl::default_for_dimension(2);
    c.maxit = 1000;
    let p = vec![-1.2, 1.0];
    let result = if observed {
        if numeric {
            optim_lbfgsb_observed(p, b, f, c, profile, |event| events.push(event))
        } else {
            optim_lbfgsb_with_gradient_observed(p, b, f, g, c, profile, |event| events.push(event))
        }
    } else if numeric {
        optim_lbfgsb_with_profile(p, b, f, c, profile)
    } else {
        optim_lbfgsb_with_gradient_and_profile(p, b, f, g, c, profile)
    }
    .unwrap();
    Replay {
        result,
        calls: calls.into_inner(),
        events,
    }
}

#[test]
fn observer_preserves_every_callback_and_result_bit() {
    for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
        for numeric in [false, true] {
            for route in 0..3 {
                let plain = replay(false, numeric, profile, route);
                let observed = replay(true, numeric, profile, route);
                assert_eq!(plain.calls, observed.calls);
                assert_eq!(plain.result, observed.result);
                assert_eq!(
                    plain.result.value.to_bits(),
                    observed.result.value.to_bits()
                );
                for (a, b) in plain.result.par.iter().zip(&observed.result.par) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
                let mut accepted = 0;
                let mut previous_value = None;
                assert!(
                    matches!(observed.events.first(),Some(OptimEvent::Started {profile:p,..}) if *p==profile)
                );
                for event in &observed.events {
                    match event {
                        OptimEvent::Started { value, .. } => previous_value = Some(*value),
                        OptimEvent::Direction {
                            model: Some(model), ..
                        } => {
                            assert_eq!(model.active + model.free, 2);
                            assert!(model.history <= 5);
                        }
                        OptimEvent::Accepted {
                            iteration,
                            alpha,
                            max_alpha,
                            search_evaluations,
                            acceptance,
                            initial_derivative,
                            final_derivative,
                            value,
                            ..
                        } => {
                            accepted += 1;
                            assert_eq!(*iteration, accepted);
                            assert!(*alpha > 0.0 && alpha <= max_alpha);
                            assert!(*search_evaluations > 0);
                            assert!(*initial_derivative < 0.0);
                            let old = previous_value.unwrap();
                            assert!(
                                *value
                                    <= old
                                        + alpha * (1e-3 * initial_derivative)
                                        + 32.0 * f64::EPSILON * old.abs().max(1.0)
                            );
                            match acceptance {
                                StepAcceptance::StrongWolfe => assert!(
                                    final_derivative.abs()
                                        <= 0.9 * initial_derivative.abs() + f64::MIN_POSITIVE
                                ),
                                StepAcceptance::StepLimit => {
                                    assert_eq!(alpha, max_alpha);
                                    assert!(*final_derivative <= 1e-3 * initial_derivative);
                                }
                                StepAcceptance::BracketWarning => {}
                                StepAcceptance::Armijo => {
                                    panic!("legacy Armijo probe reached public API")
                                }
                            }
                            previous_value = Some(*value);
                        }
                        OptimEvent::History { history, .. } => assert!(*history <= 5),
                        _ => {}
                    }
                }
                assert!(
                    matches!(observed.events.last(),Some(OptimEvent::Finished{iterations,counts,reason}) if *iterations==accepted && *counts==observed.result.counts && reason.convergence()==observed.result.convergence && reason.message()==observed.result.message)
                );
            }
        }
    }
}

#[test]
fn observer_reports_empty_inputs_and_evaluation_errors() {
    let mut events = Vec::new();
    let result = optim_lbfgsb_observed(
        vec![],
        Bounds::unbounded(0),
        |_| 7.0,
        OptimControl::default_for_dimension(0),
        ArithmeticProfile::R461X86,
        |e| events.push(e),
    )
    .unwrap();
    assert_eq!(
        events,
        vec![OptimEvent::Finished {
            iterations: 0,
            reason: TerminationReason::NothingToDo,
            counts: result.counts
        }]
    );
    events.clear();
    assert!(optim_lbfgsb_observed(
        vec![0.0],
        Bounds::unbounded(1),
        |_| f64::NAN,
        OptimControl::default_for_dimension(1),
        ArithmeticProfile::R461X86,
        |e| events.push(e)
    )
    .is_err());
    assert_eq!(
        events,
        vec![OptimEvent::EvaluationFailed {
            counts: OptimCounts::default()
        }]
    );
    events.clear();
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 0.0;
    assert!(optim_lbfgsb_observed(
        vec![0.0],
        Bounds::unbounded(1),
        |_| panic!("invalid input reached callback"),
        c,
        ArithmeticProfile::R461X86,
        |e| events.push(e)
    )
    .is_err());
    assert!(events.is_empty());
}

#[test]
fn initial_stationarity_preserves_evaluation_counts_and_event_order() {
    for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
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
            ArithmeticProfile::R461MacArm,
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
    let expected = replay(false, false, ArithmeticProfile::R461X86, 0).result;
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
        ArithmeticProfile::R461X86,
        |_| {
            if first {
                first = false;
                let nested = replay(false, true, ArithmeticProfile::R461MacArm, 1);
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
        ArithmeticProfile::R461X86,
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
        ArithmeticProfile::R461X86,
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
        ArithmeticProfile::R461X86,
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

#[test]
fn failed_evaluations_and_output_conversion_do_not_emit_success() {
    let mut calls = 0;
    let mut events = Vec::new();
    let result = optim_lbfgsb_with_gradient_observed(
        vec![4.0, -5.0],
        Bounds::unbounded(2),
        |x| {
            calls += 1;
            if calls == 3 {
                f64::NAN
            } else {
                (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 2.0).powi(2)
            }
        },
        |x| vec![2.0 * (x[0] - 1.0), 4.0 * (x[1] + 2.0)],
        OptimControl::default_for_dimension(2),
        ArithmeticProfile::R461X86,
        |e| events.push(e),
    );
    assert!(matches!(result, Err(OptimError::NonFiniteObjective { .. })));
    assert!(matches!(
        events.last(),
        Some(OptimEvent::EvaluationFailed {
            counts: OptimCounts {
                function: 2,
                gradient: 2
            }
        })
    ));
    assert!(!events
        .iter()
        .any(|e| matches!(e, OptimEvent::Finished { .. })));
    events.clear();
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 1.0134;
    let result = optim_lbfgsb_with_gradient_observed(
        vec![0.0],
        Bounds::unbounded(1),
        |_| f64::MAX,
        |_| vec![0.0],
        c,
        ArithmeticProfile::R461X86,
        |e| events.push(e),
    );
    assert!(matches!(
        result,
        Err(OptimError::NumericalFailure {
            operation: "objective unscaling",
            ..
        })
    ));
    assert!(matches!(
        events.last(),
        Some(OptimEvent::OutputConversionFailed {
            counts: OptimCounts {
                function: 1,
                gradient: 1
            }
        })
    ));
    assert!(!events
        .iter()
        .any(|e| matches!(e, OptimEvent::Finished { .. })));
}
