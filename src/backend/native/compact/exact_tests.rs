use super::*;

#[test]
fn exact_state_scales_nonunit_steps_before_cyclic_storage() {
    let lower = vec![-1.0, -1.0];
    let upper = vec![1.0, 1.0];
    let mut state = ExactCompactState::new(ArithmeticProfile::legacy(), 2, 2, &lower, &upper);

    assert_eq!(
        state.update(&[1.0, 2.0], 0.5, &[0.25, 0.5], -10.0, -8.0),
        HistoryUpdate::Stored
    );
    assert_eq!(state.col, 1);
    assert_eq!(state.head, 0);
    assert_eq!(state.itail, 0);
    assert_eq!(state.ws[0], vec![0.5, 1.0]);
    assert_eq!(state.wy[0], vec![0.25, 0.5]);
    assert_eq!(state.theta, 0.3125);
    assert_eq!(state.ss[0][0], 1.25);
}

#[test]
fn exact_subspace_uses_one_shared_feasible_alpha() {
    let point = vec![0.5, 0.5];
    let direction = vec![2.0, 0.5];
    let indices = vec![0, 1];
    let lower = vec![0.0, 0.0];
    let upper = vec![1.0, 1.0];

    let (alpha, binding_slot) =
        exact_shared_feasible_step(&point, &direction, &indices, 2, &lower, &upper);

    assert_eq!(alpha, 0.25);
    assert_eq!(binding_slot, Some(0));
    assert_eq!(point[1] + alpha * direction[1], 0.625);
}

#[test]
fn production_history_update_straddles_the_curvature_threshold() {
    for profile in [
        ArithmeticProfile::X86OpenBlas,
        ArithmeticProfile::AppleSiliconOpenBlas,
    ] {
        // Powers of two keep the cutoff and neighboring derivatives exact
        // while changing the scale of the initial directional derivative.
        for magnitude in [0.125, 1.0, 8.0] {
            let cutoff_derivative = magnitude * (-1.0 + f64::EPSILON);
            let below = f64::from_bits(cutoff_derivative.to_bits() + 1);
            let above = f64::from_bits(cutoff_derivative.to_bits() - 1);
            for alpha in [0.5, 1.0, 2.0] {
                for (final_derivative, expected) in [
                    (-1.5 * magnitude, HistoryUpdate::SkippedCurvature),
                    (-magnitude, HistoryUpdate::SkippedCurvature),
                    (below, HistoryUpdate::SkippedCurvature),
                    (cutoff_derivative, HistoryUpdate::SkippedCurvature),
                    (above, HistoryUpdate::Stored),
                ] {
                    let mut state = ExactCompactState::new(profile, 2, 2, &[-10.0; 2], &[10.0; 2]);
                    let delta = final_derivative + magnitude;
                    assert_eq!(
                        state.update(
                            &[1.0, 0.0],
                            alpha,
                            &[delta, 0.0],
                            -magnitude,
                            final_derivative
                        ),
                        expected,
                        "{profile:?}, magnitude={magnitude}, alpha={alpha}, final derivative={final_derivative:e}",
                    );
                    if expected == HistoryUpdate::Stored {
                        assert!(state.has_history());
                        assert!(state.updatd);
                        assert_eq!(state.col, 1);
                        assert_eq!(state.iupdat, 1);
                        assert_eq!(state.ws[0], [alpha, 0.0]);
                        assert_eq!(state.wy[0], [delta, 0.0]);
                        assert_eq!(state.sy[0][0], alpha * delta);
                        assert_eq!(state.theta, delta / alpha);
                        assert!(state.wt[0][0].is_finite() && state.wt[0][0] > 0.0);
                    } else {
                        assert!(!state.has_history());
                        assert!(!state.updatd);
                        assert_eq!(state.iupdat, 0);
                        assert_eq!(state.theta, 1.0);
                    }
                }
            }
        }
    }
}

#[test]
fn skipped_curvature_preserves_a_full_cyclic_history() {
    for profile in [
        ArithmeticProfile::X86OpenBlas,
        ArithmeticProfile::AppleSiliconOpenBlas,
    ] {
        for alpha in [0.5, 1.0, 2.0] {
            let mut state = ExactCompactState::new(profile, 2, 2, &[-10.0; 2], &[10.0; 2]);
            // Fill and wrap the cyclic buffer before attempting rejected updates.
            for (direction, difference) in [
                ([1.0, 0.0], [2.0, 0.0]),
                ([0.0, 1.0], [0.0, 2.0]),
                ([1.0, 1.0], [2.0, 2.0]),
            ] {
                let change = profile.short_dot(&direction, &difference);
                assert_eq!(
                    state.update(&direction, 1.0, &difference, -8.0, -8.0 + change),
                    HistoryUpdate::Stored
                );
            }
            assert_eq!(state.col, 2);
            assert_eq!(state.head, 1);
            let columns = (state.ws.clone(), state.wy.clone());
            let factors = (
                state.sy.clone(),
                state.ss.clone(),
                state.wt.clone(),
                state.wn.clone(),
                state.wn1.clone(),
            );
            let bookkeeping = (
                state.head,
                state.itail,
                state.col,
                state.iupdat,
                state.theta,
            );
            for delta in [-0.5, 0.0, f64::EPSILON / 2.0, f64::EPSILON] {
                assert_eq!(
                    state.update(&[1.0, 0.0], alpha, &[delta, 0.0], -1.0, -1.0 + delta),
                    HistoryUpdate::SkippedCurvature
                );
                assert!(!state.updatd);
                assert_eq!((&state.ws, &state.wy), (&columns.0, &columns.1));
                assert_eq!(
                    (&state.sy, &state.ss, &state.wt, &state.wn, &state.wn1),
                    (&factors.0, &factors.1, &factors.2, &factors.3, &factors.4)
                );
                assert_eq!(
                    (
                        state.head,
                        state.itail,
                        state.col,
                        state.iupdat,
                        state.theta
                    ),
                    bookkeeping
                );
            }
            // The next accepted correction must use exactly the next cyclic slot.
            assert_eq!(
                state.update(&[0.0, 1.0], 1.0, &[0.0, 2.0], -4.0, -2.0),
                HistoryUpdate::Stored
            );
            assert!(state.updatd);
            assert_eq!(
                (state.head, state.itail, state.col, state.iupdat),
                (0, 1, 2, 4)
            );
            assert_eq!(state.ws[1], [0.0, 1.0]);
            assert_eq!(state.wy[1], [0.0, 2.0]);
        }
    }
}
