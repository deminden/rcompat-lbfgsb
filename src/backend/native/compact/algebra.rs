use super::*;

pub(super) fn exact_shared_feasible_step(
    point: &[f64],
    direction: &[f64],
    indices: &[usize],
    nfree: usize,
    lower: &[f64],
    upper: &[f64],
) -> (f64, Option<usize>) {
    let mut alpha = 1.0;
    let mut candidate_alpha = alpha;
    let mut binding_slot = None;
    for (slot, &step) in direction.iter().take(nfree).enumerate() {
        let index = indices[slot];
        let bound_type = coherent_bound_type(lower[index], upper[index]);
        if bound_type != 0 {
            if step < 0.0 && bound_type <= 2 {
                let distance = lower[index] - point[index];
                if distance >= 0.0 {
                    candidate_alpha = 0.0;
                } else if step * alpha < distance {
                    candidate_alpha = distance / step;
                }
            } else if step > 0.0 && bound_type >= 2 {
                let distance = upper[index] - point[index];
                if distance <= 0.0 {
                    candidate_alpha = 0.0;
                } else if step * alpha > distance {
                    candidate_alpha = distance / step;
                }
            }
            if candidate_alpha < alpha {
                alpha = candidate_alpha;
                binding_slot = Some(slot);
            }
        }
    }
    (alpha, binding_slot)
}

pub(super) fn coherent_cholesky_upper_in_place(
    profile: ArithmeticProfile,
    matrix: &mut [Vec<f64>],
    offset: usize,
    dimension: usize,
) -> Option<()> {
    for column in 0..dimension {
        let mut squared_norm = 0.0;
        for row in 0..column {
            let update = profile.indexed_dot(row, |index| {
                (
                    matrix[offset + index][offset + row],
                    matrix[offset + index][offset + column],
                )
            });
            let mut value = matrix[offset + row][offset + column] - update;
            value /= matrix[offset + row][offset + row];
            matrix[offset + row][offset + column] = value;
            squared_norm = profile.mul_add(value, value, squared_norm);
        }
        let pivot = matrix[offset + column][offset + column] - squared_norm;
        if pivot <= 0.0 || !pivot.is_finite() {
            return None;
        }
        matrix[offset + column][offset + column] = pivot.sqrt();
    }
    Some(())
}

pub(super) fn coherent_solve_upper_transpose_in_place(
    profile: ArithmeticProfile,
    upper: &[Vec<f64>],
    offset: usize,
    dimension: usize,
    rhs: &mut [f64],
) -> Option<()> {
    if rhs.len() != dimension {
        return None;
    }
    for row in 0..dimension {
        let update = profile.indexed_dot(row, |column| {
            (upper[offset + column][offset + row], rhs[column])
        });
        let value = rhs[row] - update;
        let diagonal = upper[offset + row][offset + row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return None;
        }
        rhs[row] = value / diagonal;
    }
    Some(())
}

pub(super) fn coherent_solve_upper_in_place(
    upper: &[Vec<f64>],
    offset: usize,
    dimension: usize,
    rhs: &mut [f64],
) -> Option<()> {
    if rhs.len() != dimension {
        return None;
    }
    for row in (0..dimension).rev() {
        let diagonal = upper[offset + row][offset + row];
        if diagonal == 0.0 || !diagonal.is_finite() {
            return None;
        }
        rhs[row] /= diagonal;
        let alpha = -rhs[row];
        for column in 0..row {
            rhs[column] = alpha.mul_add(upper[offset + column][offset + row], rhs[column]);
        }
    }
    Some(())
}

pub(super) fn coherent_bound_type(lower: f64, upper: f64) -> i32 {
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

pub(super) fn coherent_daxpy(target: &mut [f64], source: &[f64], alpha: f64) {
    debug_assert_eq!(target.len(), source.len());
    for (target, &source) in target.iter_mut().zip(source.iter()) {
        *target = alpha.mul_add(source, *target);
    }
}

pub(super) fn coherent_hpsolb(
    count: usize,
    values: &mut [f64],
    indices: &mut [usize],
    heap_state: usize,
) {
    if heap_state == 0 {
        for position in 2..=count {
            let value = values[position - 1];
            let index = indices[position - 1];
            let mut child = position;
            while child > 1 {
                let parent = child / 2;
                if value < values[parent - 1] {
                    values[child - 1] = values[parent - 1];
                    indices[child - 1] = indices[parent - 1];
                    child = parent;
                } else {
                    break;
                }
            }
            values[child - 1] = value;
            indices[child - 1] = index;
        }
    }

    if count > 1 {
        let output_value = values[0];
        let output_index = indices[0];
        let input_value = values[count - 1];
        let input_index = indices[count - 1];
        let mut parent = 1_usize;
        loop {
            let mut child = 2 * parent;
            if child > count - 1 {
                break;
            }
            if values[child] < values[child - 1] {
                child += 1;
            }
            if values[child - 1] < input_value {
                values[parent - 1] = values[child - 1];
                indices[parent - 1] = indices[child - 1];
                parent = child;
            } else {
                break;
            }
        }
        values[parent - 1] = input_value;
        indices[parent - 1] = input_index;
        values[count - 1] = output_value;
        indices[count - 1] = output_index;
    }
}
