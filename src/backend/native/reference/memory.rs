use super::*;

pub(in crate::backend::native) fn lbfgs_inverse_product(
    vector: &[f64],
    history: &[Correction],
) -> Vec<f64> {
    if history.is_empty() {
        return vector.to_vec();
    }

    let mut q = vector.to_vec();
    let mut alphas = vec![0.0; history.len()];

    for (slot, correction) in history.iter().enumerate().rev() {
        let alpha = correction.rho * dot(&correction.s, &q);
        alphas[slot] = alpha;
        axpy(&mut q, &correction.y, -alpha);
    }

    let last = history.last().expect("history is not empty");
    let sy = dot(&last.s, &last.y);
    let yy = dot(&last.y, &last.y);
    let gamma = if yy > 0.0 { sy / yy } else { 1.0 };
    let mut r: Vec<f64> = q.iter().map(|&value| gamma.max(1e-20) * value).collect();

    for (alpha, correction) in alphas.iter().zip(history.iter()) {
        let beta = correction.rho * dot(&correction.y, &r);
        axpy(&mut r, &correction.s, alpha - beta);
    }

    r
}

pub(in crate::backend::native) fn lbfgs_hessian_product(
    vector: &[f64],
    history: &[Correction],
) -> Vec<f64> {
    if history.is_empty() {
        return vector.to_vec();
    }

    let theta = lbfgs_initial_hessian_scale(history);
    let mut updates: Vec<PreparedHessianUpdate> = Vec::with_capacity(history.len());

    for (correction_index, correction) in history.iter().enumerate() {
        let mut bs: Vec<f64> = correction.s.iter().map(|&value| theta * value).collect();
        for update in &updates {
            apply_hessian_update(
                &mut bs,
                &correction.s,
                &history[update.correction_index],
                update,
            );
        }
        let sbs = dot(&correction.s, &bs);
        if sbs > 0.0 && sbs.is_finite() {
            updates.push(PreparedHessianUpdate {
                correction_index,
                bs,
                sbs,
            });
        }
    }

    let mut result: Vec<f64> = vector.iter().map(|&value| theta * value).collect();
    for update in &updates {
        apply_hessian_update(
            &mut result,
            vector,
            &history[update.correction_index],
            update,
        );
    }
    result
}

pub(in crate::backend::native) fn lbfgs_initial_hessian_scale(history: &[Correction]) -> f64 {
    let Some(last) = history.last() else {
        return 1.0;
    };
    let sy = dot(&last.s, &last.y);
    let yy = dot(&last.y, &last.y);
    if sy > 0.0 && yy.is_finite() {
        (yy / sy).max(1e-20)
    } else {
        1.0
    }
}

pub(in crate::backend::native) fn apply_hessian_update(
    target: &mut [f64],
    original: &[f64],
    correction: &Correction,
    update: &PreparedHessianUpdate,
) {
    let s_target = dot(&correction.s, target);
    let y_original = dot(&correction.y, original);
    let sy = 1.0 / correction.rho;
    for ((target_i, &bs_i), &y_i) in target
        .iter_mut()
        .zip(update.bs.iter())
        .zip(correction.y.iter())
    {
        *target_i += -bs_i * s_target / update.sbs + y_i * y_original / sy;
    }
}

pub(in crate::backend::native) fn update_history(
    history: &mut Vec<Correction>,
    limit: usize,
    s: Vec<f64>,
    y: Vec<f64>,
    previous_gradient: &[f64],
) {
    let sy = dot(&s, &y);
    let directional_derivative = dot(previous_gradient, &s);
    let threshold = if directional_derivative < 0.0 && directional_derivative.is_finite() {
        HISTORY_CURVATURE_EPS * -directional_derivative
    } else {
        let ss = dot(&s, &s);
        let yy = dot(&y, &y);
        HISTORY_CURVATURE_EPS * ss.sqrt() * yy.sqrt()
    };
    if sy <= threshold || sy <= 0.0 {
        return;
    }
    if history.len() == limit {
        history.remove(0);
    }
    history.push(Correction {
        s,
        y,
        rho: 1.0 / sy,
    });
}

pub(in crate::backend::native) fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

pub(in crate::backend::native) fn axpy(target: &mut [f64], x: &[f64], alpha: f64) {
    for (target_i, &x_i) in target.iter_mut().zip(x.iter()) {
        *target_i += alpha * x_i;
    }
}

pub(in crate::backend::native) fn lbfgs_direction(
    gradient: &[f64],
    history: &[Correction],
) -> Vec<f64> {
    lbfgs_inverse_product(gradient, history)
        .into_iter()
        .map(|value| -value)
        .collect()
}
