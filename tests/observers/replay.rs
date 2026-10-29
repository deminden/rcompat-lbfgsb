use super::*;
#[derive(Debug, PartialEq)]
pub(super) struct Replay {
    pub(super) result: OptimResult,
    pub(super) calls: Vec<(bool, Vec<f64>)>,
    pub(super) events: Vec<OptimEvent>,
}
pub(super) fn replay(
    observed: bool,
    numeric: bool,
    profile: ArithmeticProfile,
    route: usize,
) -> Replay {
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
pub(super) fn observer_preserves_every_callback_and_result_bit() {
    for profile in [
        ArithmeticProfile::X86OpenBlas,
        ArithmeticProfile::AppleSiliconOpenBlas,
    ] {
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
