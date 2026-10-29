use super::*;

#[test]
fn compact_factor_and_solve_recover_independently_constructed_solutions() {
    // Construct A = U^T U from a known upper factor; the production code
    // must recover a solve without sharing its factorization arithmetic.
    for n in [1, 2, 3, 4, 5, 8, 11, 16] {
        let upper: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if j < i {
                            0.0
                        } else if j == i {
                            2.0 + i as f64 / 8.0
                        } else {
                            (i + j + 1) as f64 / 32.0
                        }
                    })
                    .collect()
            })
            .collect();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] = (0..n).map(|k| upper[k][i] * upper[k][j]).sum();
            }
        }
        let expected: Vec<f64> = (0..n).map(|i| (i as f64 - 3.0) / 7.0).collect();
        let mut rhs: Vec<f64> = matrix
            .iter()
            .map(|row| row.iter().zip(&expected).map(|(a, b)| a * b).sum())
            .collect();
        coherent_cholesky_upper_in_place(ArithmeticProfile::legacy(), &mut matrix, 0, n).unwrap();
        coherent_solve_upper_transpose_in_place(
            ArithmeticProfile::legacy(),
            &matrix,
            0,
            n,
            &mut rhs,
        )
        .unwrap();
        coherent_solve_upper_in_place(&matrix, 0, n, &mut rhs).unwrap();
        for (actual, expected) in rhs.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "n={n}: {actual} != {expected}"
            );
        }
    }
}
