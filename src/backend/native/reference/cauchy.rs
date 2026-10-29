use super::*;

pub(in crate::backend::native) fn generalized_cauchy_point_identity(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(
        x,
        gradient,
        lower,
        upper,
        |vector| vector.to_vec(),
        BoundActivity::Tolerant,
    )
}

pub(in crate::backend::native) fn generalized_cauchy_point_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
) -> CauchyPoint {
    generalized_cauchy_point_limited_memory_with_activity(
        x,
        gradient,
        lower,
        upper,
        history,
        BoundActivity::Tolerant,
    )
}

pub(in crate::backend::native) fn generalized_cauchy_point_limited_memory_with_activity(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    bound_activity: BoundActivity,
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(
        x,
        gradient,
        lower,
        upper,
        |vector| lbfgs_hessian_product(vector, history),
        bound_activity,
    )
}

pub(in crate::backend::native) fn generalized_cauchy_point_with_hessian<F>(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    mut hessian_product: F,
    bound_activity: BoundActivity,
) -> CauchyPoint
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let mut breakpoints = Vec::new();
    let mut point = x.to_vec();
    let mut free = vec![true; x.len()];

    for index in 0..x.len() {
        let g = gradient[index];
        if lower[index] == upper[index] {
            point[index] = lower[index];
            free[index] = false;
            continue;
        }
        if g > 0.0 && lower[index].is_finite() {
            breakpoints.push(((x[index] - lower[index]) / g, index));
        } else if g < 0.0 && upper[index].is_finite() {
            breakpoints.push(((x[index] - upper[index]) / g, index));
        }
    }

    breakpoints.retain(|(time, _)| time.is_finite() && *time >= 0.0);
    breakpoints.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .expect("finite breakpoint times")
    });

    let mut previous_time = 0.0;
    let mut cursor = 0;
    let mut curvature_floor = 0.0;

    loop {
        let direction = cauchy_path_direction(gradient, &free);
        if norm_inf(&direction) <= MIN_STEP {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        let hessian_direction = hessian_product(&direction);
        let displacement = difference(&point, x);
        let derivative = dot(gradient, &direction) + dot(&displacement, &hessian_direction);
        let curvature = dot(&direction, &hessian_direction);
        if cursor == 0 && curvature > 0.0 && curvature.is_finite() {
            curvature_floor = f64::EPSILON * curvature;
        }

        if derivative >= 0.0 {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        let next_time = breakpoints
            .get(cursor)
            .map(|(time, _)| *time)
            .unwrap_or(f64::INFINITY);
        let interval = next_time - previous_time;

        if curvature > 0.0 && curvature.is_finite() {
            let stationary_interval = -derivative / curvature.max(curvature_floor);
            if stationary_interval <= interval {
                advance_cauchy_point(&mut point, &direction, stationary_interval, lower, upper);
                return CauchyPoint {
                    active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                    x: point,
                };
            }
        }

        if !next_time.is_finite() {
            return CauchyPoint {
                active_count: active_count_with_activity(&point, lower, upper, bound_activity),
                x: point,
            };
        }

        advance_cauchy_point(&mut point, &direction, interval.max(0.0), lower, upper);
        previous_time = next_time;

        while cursor < breakpoints.len() && (breakpoints[cursor].0 - next_time).abs() <= BOUND_TOL {
            let index = breakpoints[cursor].1;
            free[index] = false;
            cursor += 1;
        }
    }
}

pub(in crate::backend::native) fn cauchy_path_direction(
    gradient: &[f64],
    free: &[bool],
) -> Vec<f64> {
    gradient
        .iter()
        .zip(free.iter())
        .map(|(&gradient_i, &free_i)| if free_i { -gradient_i } else { 0.0 })
        .collect()
}

pub(in crate::backend::native) fn advance_cauchy_point(
    point: &mut [f64],
    direction: &[f64],
    interval: f64,
    lower: &[f64],
    upper: &[f64],
) {
    for (((point_i, &direction_i), &lower_i), &upper_i) in point
        .iter_mut()
        .zip(direction.iter())
        .zip(lower.iter())
        .zip(upper.iter())
    {
        *point_i = (*point_i + interval * direction_i).clamp(lower_i, upper_i);
    }
}

pub(in crate::backend::native) fn active_count_with_activity(
    point: &[f64],
    lower: &[f64],
    upper: &[f64],
    bound_activity: BoundActivity,
) -> usize {
    let mut count = 0;
    for index in 0..point.len() {
        if is_active_at_bound(point[index], lower[index], upper[index], bound_activity) {
            count += 1;
        }
    }
    count
}

pub(in crate::backend::native) fn is_active_at_bound(
    value: f64,
    lower: f64,
    upper: f64,
    bound_activity: BoundActivity,
) -> bool {
    if lower == upper {
        return true;
    }

    match bound_activity {
        BoundActivity::Tolerant => {
            (value <= lower + BOUND_TOL && lower.is_finite())
                || (value >= upper - BOUND_TOL && upper.is_finite())
        }
        BoundActivity::Exact => {
            (value <= lower && lower.is_finite()) || (value >= upper && upper.is_finite())
        }
    }
}
