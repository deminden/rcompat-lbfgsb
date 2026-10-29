use super::*;
#[test]
fn observer_reports_empty_inputs_and_evaluation_errors() {
    let mut events = Vec::new();
    let result = optim_lbfgsb_observed(
        vec![],
        Bounds::unbounded(0),
        |_| 7.0,
        OptimControl::default_for_dimension(0),
        ArithmeticProfile::X86OpenBlas,
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
    assert!(
        optim_lbfgsb_observed(
            vec![0.0],
            Bounds::unbounded(1),
            |_| f64::NAN,
            OptimControl::default_for_dimension(1),
            ArithmeticProfile::X86OpenBlas,
            |e| events.push(e)
        )
        .is_err()
    );
    assert_eq!(
        events,
        vec![OptimEvent::EvaluationFailed {
            counts: OptimCounts::default()
        }]
    );
    events.clear();
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 0.0;
    assert!(
        optim_lbfgsb_observed(
            vec![0.0],
            Bounds::unbounded(1),
            |_| panic!("invalid input reached callback"),
            c,
            ArithmeticProfile::X86OpenBlas,
            |e| events.push(e)
        )
        .is_err()
    );
    assert!(events.is_empty());
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
        ArithmeticProfile::X86OpenBlas,
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
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, OptimEvent::Finished { .. }))
    );
    events.clear();
    let mut c = OptimControl::default_for_dimension(1);
    c.fnscale = 1.0134;
    let result = optim_lbfgsb_with_gradient_observed(
        vec![0.0],
        Bounds::unbounded(1),
        |_| f64::MAX,
        |_| vec![0.0],
        c,
        ArithmeticProfile::X86OpenBlas,
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
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, OptimEvent::Finished { .. }))
    );
}
