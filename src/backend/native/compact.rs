use super::*;

pub(super) fn compact_subspace_minimizer_r23(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
    bound_activity: BoundActivity,
) -> Option<SubspacePoint> {
    let free_indices = free_indices_with_activity(&cauchy.x, lower, upper, bound_activity);
    if free_indices.is_empty() {
        return Some(SubspacePoint {
            x: cauchy.x.clone(),
            free_count: 0,
            clipped: false,
        });
    }

    // Compact L-BFGS matrices reproduce the small finite-box subspace solve
    // without materializing a dense Hessian, keeping this path close to R 2.3.
    let free_mask = index_mask(x.len(), &free_indices);
    let active_indices: Vec<usize> = (0..x.len()).filter(|&index| !free_mask[index]).collect();
    let compact = CompactMatrices::new(history)?;
    let displacement = difference(&cauchy.x, x);
    let cauchy_middle = compact.middle_product(&compact.w_transpose(&displacement))?;
    let mut reduced = Vec::with_capacity(free_indices.len());
    for &index in &free_indices {
        reduced.push(-compact.theta * displacement[index] - gradient[index]);
    }

    for (correction_index, correction) in history.iter().enumerate() {
        let a1 = cauchy_middle[correction_index];
        let a2 = compact.theta * cauchy_middle[history.len() + correction_index];
        for (slot, &index) in free_indices.iter().enumerate() {
            reduced[slot] += correction.y[index] * a1 + correction.s[index] * a2;
        }
    }

    let wn = compact.form_subspace_factor(&free_indices, &active_indices)?;
    let mut wv = vec![0.0; 2 * history.len()];
    for (correction_index, correction) in history.iter().enumerate() {
        let mut y_dot = 0.0;
        let mut s_dot = 0.0;
        for (slot, &index) in free_indices.iter().enumerate() {
            y_dot += correction.y[index] * reduced[slot];
            s_dot += correction.s[index] * reduced[slot];
        }
        wv[correction_index] = y_dot;
        wv[history.len() + correction_index] = compact.theta * s_dot;
    }

    solve_upper_transpose_in_place(&wn, 0, wv.len(), &mut wv)?;
    for value in wv.iter_mut().take(history.len()) {
        *value = -*value;
    }
    solve_upper_in_place(&wn, 0, wv.len(), &mut wv)?;

    let mut reduced_step = reduced;
    for (correction_index, correction) in history.iter().enumerate() {
        let s_slot = history.len() + correction_index;
        for (slot, &index) in free_indices.iter().enumerate() {
            reduced_step[slot] += correction.y[index] * wv[correction_index] / compact.theta
                + correction.s[index] * wv[s_slot];
        }
    }
    for step in &mut reduced_step {
        *step /= compact.theta;
    }

    let mut alpha = 1.0;
    let mut candidate_alpha = alpha;
    let mut boundary_slot = None;
    for (slot, &index) in free_indices.iter().enumerate() {
        let step = reduced_step[slot];
        if lower[index].is_finite() && step < 0.0 {
            let room = lower[index] - cauchy.x[index];
            if room >= 0.0 {
                candidate_alpha = 0.0;
            } else if step * alpha < room {
                candidate_alpha = room / step;
            }
        } else if upper[index].is_finite() && step > 0.0 {
            let room = upper[index] - cauchy.x[index];
            if room <= 0.0 {
                candidate_alpha = 0.0;
            } else if step * alpha > room {
                candidate_alpha = room / step;
            }
        }
        if candidate_alpha < alpha {
            alpha = candidate_alpha;
            boundary_slot = Some(slot);
        }
    }

    let mut point = cauchy.x.clone();
    if alpha < 1.0 {
        if let Some(slot) = boundary_slot {
            let index = free_indices[slot];
            if reduced_step[slot] > 0.0 {
                point[index] = upper[index];
                reduced_step[slot] = 0.0;
            } else if reduced_step[slot] < 0.0 {
                point[index] = lower[index];
                reduced_step[slot] = 0.0;
            }
        }
    }
    for (slot, &index) in free_indices.iter().enumerate() {
        point[index] += alpha * reduced_step[slot];
    }

    Some(SubspacePoint {
        x: point,
        free_count: free_indices.len(),
        clipped: alpha < 1.0,
    })
}

#[derive(Debug)]
struct CompactMatrices {
    theta: f64,
    sy: Vec<Vec<f64>>,
    wt: Vec<Vec<f64>>,
    history: Vec<Correction>,
}

impl CompactMatrices {
    fn new(history: &[Correction]) -> Option<Self> {
        let last = history.last()?;
        let sy_last = dot(&last.s, &last.y);
        let yy_last = dot(&last.y, &last.y);
        if sy_last <= 0.0 || !sy_last.is_finite() || !yy_last.is_finite() {
            return None;
        }
        let theta = (yy_last / sy_last).max(1e-20);
        let col = history.len();
        let mut sy = vec![vec![0.0; col]; col];
        let mut ss = vec![vec![0.0; col]; col];
        for (row, row_correction) in history.iter().enumerate() {
            for (column, column_correction) in history.iter().enumerate() {
                sy[row][column] = dot(&row_correction.s, &column_correction.y);
                ss[row][column] = dot(&row_correction.s, &column_correction.s);
            }
        }
        let wt = form_compact_wt(theta, &sy, &ss)?;
        Some(Self {
            theta,
            sy,
            wt,
            history: history.to_vec(),
        })
    }

    fn w_transpose(&self, vector: &[f64]) -> Vec<f64> {
        let col = self.history.len();
        let mut result = vec![0.0; 2 * col];
        for (slot, correction) in self.history.iter().enumerate() {
            result[slot] = dot(&correction.y, vector);
            result[col + slot] = self.theta * dot(&correction.s, vector);
        }
        result
    }

    fn middle_product(&self, vector: &[f64]) -> Option<Vec<f64>> {
        compact_middle_product(&self.sy, &self.wt, vector)
    }

    fn form_subspace_factor(
        &self,
        free_indices: &[usize],
        active_indices: &[usize],
    ) -> Option<Vec<Vec<f64>>> {
        let col = self.history.len();
        let col2 = 2 * col;
        let mut wn = vec![vec![0.0; col2]; col2];

        for row in 0..col {
            let row_correction = &self.history[row];
            let bottom_row = col + row;
            for column in 0..=row {
                let column_correction = &self.history[column];
                let bottom_column = col + column;
                wn[column][row] =
                    dot_on_indices(&row_correction.y, &column_correction.y, free_indices)
                        / self.theta;
                wn[bottom_column][bottom_row] =
                    dot_on_indices(&row_correction.s, &column_correction.s, active_indices)
                        * self.theta;
            }

            for (column, column_correction) in self.history.iter().enumerate().take(row) {
                wn[column][bottom_row] =
                    -dot_on_indices(&row_correction.s, &column_correction.y, active_indices);
            }
            for (column, column_correction) in self.history.iter().enumerate().take(col).skip(row) {
                wn[column][bottom_row] =
                    dot_on_indices(&row_correction.s, &column_correction.y, free_indices);
            }
            wn[row][row] += self.sy[row][row];
        }

        cholesky_upper_in_place(&mut wn, 0, col)?;
        for column in col..col2 {
            let mut rhs: Vec<f64> = (0..col).map(|row| wn[row][column]).collect();
            solve_upper_transpose_in_place(&wn, 0, col, &mut rhs)?;
            for row in 0..col {
                wn[row][column] = rhs[row];
            }
        }
        for row in col..col2 {
            for column in row..col2 {
                let update = (0..col)
                    .map(|index| wn[index][row] * wn[index][column])
                    .sum::<f64>();
                wn[row][column] += update;
            }
        }
        cholesky_upper_in_place(&mut wn, col, col)?;
        Some(wn)
    }
}

fn form_compact_wt(theta: f64, sy: &[Vec<f64>], ss: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let col = sy.len();
    let mut wt = vec![vec![0.0; col]; col];
    for column in 0..col {
        wt[0][column] = theta * ss[0][column];
    }
    for row in 1..col {
        for column in row..col {
            let update = (0..row)
                .map(|index| sy[row][index] * sy[column][index] / sy[index][index])
                .sum::<f64>();
            wt[row][column] = update + theta * ss[row][column];
        }
    }
    cholesky_upper_in_place(&mut wt, 0, col)?;
    Some(wt)
}

fn compact_middle_product(sy: &[Vec<f64>], wt: &[Vec<f64>], vector: &[f64]) -> Option<Vec<f64>> {
    let col = sy.len();
    if vector.len() != 2 * col {
        return None;
    }
    let mut product = vec![0.0; 2 * col];
    product[col] = vector[col];
    for row in 1..col {
        let update = (0..row)
            .map(|index| sy[row][index] * vector[index] / sy[index][index])
            .sum::<f64>();
        product[col + row] = vector[col + row] + update;
    }
    solve_upper_transpose_in_place(wt, 0, col, &mut product[col..])?;
    for index in 0..col {
        product[index] = vector[index] / sy[index][index].sqrt();
    }
    solve_upper_in_place(wt, 0, col, &mut product[col..])?;
    for index in 0..col {
        product[index] = -product[index] / sy[index][index].sqrt();
    }
    for index in 0..col {
        let update = (index + 1..col)
            .map(|row| sy[row][index] * product[col + row] / sy[index][index])
            .sum::<f64>();
        product[index] += update;
    }
    Some(product)
}

fn cholesky_upper_in_place(matrix: &mut [Vec<f64>], offset: usize, dimension: usize) -> Option<()> {
    for column in 0..dimension {
        for row in 0..=column {
            let mut value = matrix[offset + row][offset + column];
            for index in 0..row {
                value -=
                    matrix[offset + index][offset + row] * matrix[offset + index][offset + column];
            }
            if row == column {
                if value <= CURVATURE_EPS || !value.is_finite() {
                    return None;
                }
                matrix[offset + row][offset + column] = value.sqrt();
            } else {
                matrix[offset + row][offset + column] = value / matrix[offset + row][offset + row];
            }
        }
    }
    Some(())
}

fn solve_upper_transpose_in_place(
    upper: &[Vec<f64>],
    offset: usize,
    dimension: usize,
    rhs: &mut [f64],
) -> Option<()> {
    if rhs.len() != dimension {
        return None;
    }
    for row in 0..dimension {
        let mut value = rhs[row];
        for (column, &solution) in rhs.iter().take(row).enumerate() {
            value -= upper[offset + column][offset + row] * solution;
        }
        let diagonal = upper[offset + row][offset + row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return None;
        }
        rhs[row] = value / diagonal;
    }
    Some(())
}

fn solve_upper_in_place(
    upper: &[Vec<f64>],
    offset: usize,
    dimension: usize,
    rhs: &mut [f64],
) -> Option<()> {
    if rhs.len() != dimension {
        return None;
    }
    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in row + 1..dimension {
            value -= upper[offset + row][offset + column] * rhs[column];
        }
        let diagonal = upper[offset + row][offset + row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return None;
        }
        rhs[row] = value / diagonal;
    }
    Some(())
}

fn index_mask(dimension: usize, indices: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; dimension];
    for &index in indices {
        mask[index] = true;
    }
    mask
}

fn dot_on_indices(left: &[f64], right: &[f64], indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&index| left[index] * right[index])
        .sum()
}
