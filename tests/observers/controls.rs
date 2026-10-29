use super::*;

#[test]
fn requested_budgets_and_history_limits_are_observable_in_production() {
    for n in [11, 13] {
        for lmm in [1, 5] {
            for maxit in [0, 2, 8] {
                for numeric in [false, true] {
                    let mut events = Vec::new();
                    let weights: Vec<_> = (0..n).map(|i| 2.0_f64.powi(i as i32)).collect();
                    let f = |x: &[f64]| {
                        x.iter()
                            .zip(&weights)
                            .map(|(x, w)| 0.5 * w * (x - 1.0).powi(2))
                            .sum()
                    };
                    let g =
                        |x: &[f64]| x.iter().zip(&weights).map(|(x, w)| w * (x - 1.0)).collect();
                    let mut control = OptimControl::default_for_dimension(n);
                    control.maxit = maxit;
                    control.lmm = lmm;
                    control.factr = 0.0;
                    control.pgtol = 0.0;
                    let bounds = Bounds::new(vec![-10.0; n], vec![10.0; n]).unwrap();
                    let result = if numeric {
                        optim_lbfgsb_observed(
                            vec![2.0; n],
                            bounds,
                            f,
                            control,
                            ArithmeticProfile::AppleSiliconOpenBlas,
                            |e| events.push(e),
                        )
                    } else {
                        optim_lbfgsb_with_gradient_observed(
                            vec![2.0; n],
                            bounds,
                            f,
                            g,
                            control,
                            ArithmeticProfile::AppleSiliconOpenBlas,
                            |e| events.push(e),
                        )
                    }
                    .unwrap();
                    assert_eq!(
                        result.convergence, 1,
                        "n={n}, lmm={lmm}, maxit={maxit}, numeric={numeric}"
                    );
                    assert!(matches!(events.last(), Some(OptimEvent::Finished {
                        iterations, reason: TerminationReason::IterationLimit, ..
                    }) if *iterations == maxit+1));
                    let histories: Vec<_> = events
                        .iter()
                        .filter_map(|e| match e {
                            OptimEvent::History { history, .. } => Some(*history),
                            _ => None,
                        })
                        .collect();
                    assert!(histories.iter().all(|&history| history <= lmm));
                    if maxit >= lmm {
                        assert_eq!(histories.iter().copied().max(), Some(lmm));
                    }
                }
            }
        }
    }
}
