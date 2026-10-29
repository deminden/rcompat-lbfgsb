//! Dense BFGS identities independent of the compact matrix formulas.
use super::*;

fn inner(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn product(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| inner(row, x)).collect()
}
fn dense_bfgs(n: usize, theta: f64, pairs: &[(Vec<f64>, Vec<f64>)]) -> Vec<Vec<f64>> {
    let mut b = vec![vec![0.0; n]; n];
    for (i, row) in b.iter_mut().enumerate() {
        row[i] = theta;
    }
    for (s, y) in pairs {
        let bs = product(&b, s);
        let sbs = inner(s, &bs);
        let sy = inner(s, y);
        assert!(sbs > 0.0 && sy > 0.0);
        for i in 0..n {
            for j in 0..n {
                b[i][j] += y[i] * y[j] / sy - bs[i] * bs[j] / sbs;
            }
        }
    }
    b
}
fn independent_solve(a: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    // Pivoted Gaussian elimination instead of the compact Cholesky machinery.
    let n = rhs.len();
    let mut a = a.to_vec();
    let mut b = rhs.to_vec();
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|i, j| a[*i][column].abs().total_cmp(&a[*j][column].abs()))
            .unwrap();
        a.swap(column, pivot);
        b.swap(column, pivot);
        assert!(a[column][column].abs() > 1e-14);
        let pivot_tail = a[column][column + 1..].to_vec();
        for row in column + 1..n {
            let ratio = a[row][column] / a[column][column];
            for (entry, pivot_entry) in a[row][column + 1..].iter_mut().zip(&pivot_tail) {
                *entry -= ratio * pivot_entry;
            }
            b[row] -= ratio * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x
}
fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-9 * (1.0 + b.abs()),
            "component {i}: {a} != {b}"
        );
    }
}

#[test]
fn cyclic_model_matches_dense_bfgs_and_unbounded_newton_steps() {
    let mut checked = 0;
    for n in [1, 2, 4, 9, 10, 11, 13, 32, 64] {
        for memory in [1, 3, 5] {
            for profile in [
                ArithmeticProfile::X86OpenBlas,
                ArithmeticProfile::AppleSiliconOpenBlas,
            ] {
                let lower = vec![f64::NEG_INFINITY; n];
                let upper = vec![f64::INFINITY; n];
                let mut state = ExactCompactState::new(profile, n, memory, &lower, &upper);
                let mut pairs = Vec::new();
                for iteration in 0..3 * memory + 2 {
                    let direction: Vec<_> = (0..n)
                        .map(|i| ((i * 11 + iteration * 7) % 19) as f64 / 16.0 - 0.53125)
                        .collect();
                    let alpha = if iteration % 2 == 0 { 1.0 } else { 0.25 };
                    let step: Vec<_> = direction.iter().map(|v| alpha * v).collect();
                    // Independent SPD A=diag(1+i/4)+11^T/16 gives y=A*s.
                    let sum = step.iter().sum::<f64>();
                    let y: Vec<_> = step
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (1.0 + i as f64 / 4.0) * s + sum / 16.0)
                        .collect();
                    let change = inner(&direction, &y);
                    assert_eq!(
                        state.update(&direction, alpha, &y, -change / 2.0, change / 2.0),
                        HistoryUpdate::Stored
                    );
                    pairs.push((step, y));
                    if pairs.len() > memory {
                        pairs.remove(0);
                    }
                    assert_eq!(state.col, pairs.len());
                    assert_eq!(
                        state.head,
                        if iteration < memory {
                            0
                        } else {
                            (iteration + 1 - memory) % memory
                        }
                    );
                    for (logical, (s, y)) in pairs.iter().enumerate() {
                        let stored = (state.head + logical) % memory;
                        assert_eq!(&state.ws[stored], s);
                        assert_eq!(&state.wy[stored], y);
                    }
                    let dense = dense_bfgs(n, state.theta, &pairs);
                    let latest = pairs.last().unwrap();
                    close(&product(&dense, &latest.0), &latest.1);
                    for sign in [-1.0, 1.0] {
                        let v: Vec<_> = (0..n).map(|i| sign * (i as f64 / 8.0 - 0.375)).collect();
                        let mut wtv: Vec<_> = pairs.iter().map(|(_, y)| inner(y, &v)).collect();
                        wtv.extend(pairs.iter().map(|(s, _)| state.theta * inner(s, &v)));
                        let middle = state.bmv(&wtv).unwrap();
                        let mut bv: Vec<_> = v.iter().map(|v| state.theta * v).collect();
                        for (j, (s, y)) in pairs.iter().enumerate() {
                            for i in 0..n {
                                bv[i] -=
                                    y[i] * middle[j] + state.theta * s[i] * middle[pairs.len() + j];
                            }
                        }
                        close(&bv, &product(&dense, &v));
                    }
                    let g: Vec<_> = (0..n)
                        .map(|i| ((i + iteration) % 7) as f64 / 8.0 - 0.375)
                        .collect();
                    let expected =
                        independent_solve(&dense, &g.iter().map(|v| -v).collect::<Vec<_>>());
                    let choice = state
                        .direction(&vec![0.0; n], &g, &lower, &upper, 1.0)
                        .unwrap();
                    close(&choice.direction, &expected);
                    assert_eq!(state.snapshot().free, n);
                    assert_eq!(state.snapshot().active, 0);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 594);
}

#[test]
fn free_partition_tracks_entering_and_leaving_sets() {
    for profile in [
        ArithmeticProfile::X86OpenBlas,
        ArithmeticProfile::AppleSiliconOpenBlas,
    ] {
        let mut state = ExactCompactState::new(profile, 8, 3, &[-1.0; 8], &[1.0; 8]);
        let mut previous = vec![true; 8];
        for mask in 0..256 {
            let free: Vec<_> = (0..8).map(|i| mask & (1 << i) != 0).collect();
            for (i, is_free) in free.iter().enumerate() {
                state.iwhere[i] = if *is_free { 0 } else { 1 };
            }
            let part = state.partition_free_variables();
            let expected_free: Vec<_> = (0..8).filter(|i| free[*i]).collect();
            assert_eq!(state.index[..part.nfree], expected_free);
            let mut indices = state.index.clone();
            indices.sort_unstable();
            assert_eq!(indices, (0..8).collect::<Vec<_>>());
            if mask > 0 {
                let expected_entering: Vec<_> =
                    (0..8).filter(|i| !previous[*i] && free[*i]).collect();
                let expected_leaving: Vec<_> =
                    (0..8).filter(|i| previous[*i] && !free[*i]).collect();
                let mut entering = state.indx2[..part.nenter].to_vec();
                entering.sort_unstable();
                let mut leaving = state.indx2[part.ileave..].to_vec();
                leaving.sort_unstable();
                assert_eq!(entering, expected_entering);
                assert_eq!(leaving, expected_leaving);
                assert_eq!(part.work, !entering.is_empty() || !leaving.is_empty());
            }
            previous = free;
            state.nfree_previous = part.nfree;
            state.iteration += 1;
        }
        state.clear_history();
        assert_eq!(state.snapshot().history, 0);
        assert_eq!(state.theta, 1.0);
        assert!(state.wt.iter().flatten().all(|x| *x == 0.0));
    }
}

#[test]
fn retained_storage_is_linear_in_dimension_at_fixed_history() {
    for n in [1, 2, 9, 10, 11, 13, 128, 1024] {
        for m in [1, 5, 10] {
            let state = ExactCompactState::new(
                ArithmeticProfile::X86OpenBlas,
                n,
                m,
                &vec![-1.0; n],
                &vec![1.0; n],
            );
            let slots: usize = [
                &state.ws, &state.wy, &state.sy, &state.ss, &state.wt, &state.wn, &state.wn1,
            ]
            .into_iter()
            .map(|matrix| matrix.iter().map(Vec::len).sum::<usize>())
            .sum();
            assert_eq!(slots, 2 * n * m + 11 * m * m);
            assert_eq!(state.iwhere.len(), n);
            assert_eq!(state.index.len(), n);
            assert_eq!(state.indx2.len(), n);
        }
    }
}
