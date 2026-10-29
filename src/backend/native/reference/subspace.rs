use super::*;

pub(in crate::backend::native) fn subspace_minimizer_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
) -> Option<SubspacePoint> {
    subspace_minimizer_limited_memory_with_activity(
        ArithmeticProfile::legacy(),
        x,
        gradient,
        lower,
        upper,
        history,
        cauchy,
        BoundActivity::Tolerant,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::backend::native) fn subspace_minimizer_limited_memory_with_activity(
    profile: ArithmeticProfile,
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
    bound_activity: BoundActivity,
) -> Option<SubspacePoint> {
    subspace_minimizer_with_hessian_and_activity(
        profile,
        x,
        gradient,
        lower,
        upper,
        cauchy,
        bound_activity,
        |vector| lbfgs_hessian_product(vector, history),
    )
}

pub(in crate::backend::native) fn subspace_minimizer_with_hessian<F>(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    cauchy: &CauchyPoint,
    mut hessian_product: F,
) -> Option<SubspacePoint>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    subspace_minimizer_with_hessian_and_activity(
        ArithmeticProfile::legacy(),
        x,
        gradient,
        lower,
        upper,
        cauchy,
        BoundActivity::Tolerant,
        &mut hessian_product,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::backend::native) fn subspace_minimizer_with_hessian_and_activity<F>(
    profile: ArithmeticProfile,
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    cauchy: &CauchyPoint,
    bound_activity: BoundActivity,
    mut hessian_product: F,
) -> Option<SubspacePoint>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let free_indices = free_indices_with_activity(&cauchy.x, lower, upper, bound_activity);
    if free_indices.is_empty() {
        return Some(SubspacePoint {
            x: cauchy.x.clone(),
            free_count: 0,
            clipped: false,
        });
    }

    let displacement = difference(&cauchy.x, x);
    let hessian_displacement = hessian_product(&displacement);
    let matrix = dense_subspace_hessian(x.len(), &free_indices, &mut hessian_product);
    let rhs: Vec<f64> = free_indices
        .iter()
        .map(|&index| -(gradient[index] + hessian_displacement[index]))
        .collect();
    let subspace_step = solve_positive_definite(matrix, rhs)?;

    let mut direction = vec![0.0; x.len()];
    for (&index, &step) in free_indices.iter().zip(subspace_step.iter()) {
        direction[index] = step;
    }

    let alpha = geometric_feasible_step(&cauchy.x, &direction, lower, upper).min(1.0);
    let clipped = alpha < 1.0;
    let point = bounded_step(profile, &cauchy.x, &direction, alpha, lower, upper);

    Some(SubspacePoint {
        x: point,
        free_count: free_indices.len(),
        clipped,
    })
}

pub(in crate::backend::native) fn free_indices_with_activity(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    bound_activity: BoundActivity,
) -> Vec<usize> {
    (0..point.len())
        .filter(|&index| {
            !is_active_at_bound(point[index], lower[index], upper[index], bound_activity)
        })
        .collect()
}

pub(in crate::backend::native) fn dense_subspace_hessian<F>(
    dimension: usize,
    free_indices: &[usize],
    hessian_product: &mut F,
) -> Vec<Vec<f64>>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let mut matrix = vec![vec![0.0; free_indices.len()]; free_indices.len()];

    for (column, &basis_index) in free_indices.iter().enumerate() {
        let mut basis = vec![0.0; dimension];
        basis[basis_index] = 1.0;
        let product = hessian_product(&basis);
        for (row, &free_index) in free_indices.iter().enumerate() {
            matrix[row][column] = product[free_index];
        }
    }

    matrix
}

pub(in crate::backend::native) fn solve_positive_definite(
    matrix: Vec<Vec<f64>>,
    rhs: Vec<f64>,
) -> Option<Vec<f64>> {
    let dimension = rhs.len();
    if dimension == 0 {
        return Some(Vec::new());
    }
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    let mut factor = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for (&row_value, &column_value) in factor[row][..column]
                .iter()
                .zip(factor[column][..column].iter())
            {
                value -= row_value * column_value;
            }

            if row == column {
                if value <= CURVATURE_EPS || !value.is_finite() {
                    return None;
                }
                factor[row][column] = value.sqrt();
            } else {
                factor[row][column] = value / factor[column][column];
            }
        }
    }

    let mut y = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = rhs[row];
        for (column, y_column) in y.iter().take(row).enumerate() {
            value -= factor[row][column] * y_column;
        }
        y[row] = value / factor[row][row];
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = y[row];
        for column in row + 1..dimension {
            value -= factor[column][row] * solution[column];
        }
        solution[row] = value / factor[row][row];
    }

    Some(solution)
}
