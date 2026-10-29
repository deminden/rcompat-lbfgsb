use super::*;

#[derive(Debug, Clone, Copy)]
pub(in crate::backend::native) struct MoreThuentePoint {
    pub(in crate::backend::native) alpha: f64,
    pub(in crate::backend::native) value: f64,
    pub(in crate::backend::native) derivative: f64,
}

#[derive(Debug, Clone, Copy)]
pub(in crate::backend::native) struct MoreThuenteStepState {
    pub(in crate::backend::native) best: MoreThuentePoint,
    pub(in crate::backend::native) other: MoreThuentePoint,
    pub(in crate::backend::native) bracketed: bool,
}

pub(in crate::backend::native) fn more_thuente_safeguard_interval(
    best_alpha: f64,
    other_alpha: f64,
    alpha: f64,
    max_alpha: f64,
    bracketed: bool,
) -> (f64, f64) {
    if bracketed {
        (best_alpha.min(other_alpha), best_alpha.max(other_alpha))
    } else if alpha >= best_alpha {
        (
            0.0,
            (alpha + MORE_THUENTE_XTRAPU * (alpha - best_alpha)).min(max_alpha),
        )
    } else {
        (
            (alpha + MORE_THUENTE_XTRAPU * (alpha - best_alpha)).max(0.0),
            max_alpha,
        )
    }
}

pub(in crate::backend::native) fn more_thuente_step(
    profile: ArithmeticProfile,
    state: &mut MoreThuenteStepState,
    trial: MoreThuentePoint,
    step_min: f64,
    step_max: f64,
) -> f64 {
    let best = state.best;
    let delta = trial.alpha - best.alpha;
    if delta == 0.0 {
        return trial.alpha;
    }
    let signed_derivative = trial.derivative * (best.derivative / best.derivative.abs());
    let mut bounded = false;

    let mut next_alpha = if trial.value > best.value {
        state.bracketed = true;
        bounded = true;
        let cubic = more_thuente_cubic_case_one(profile, best, trial).unwrap_or(trial.alpha);
        let quadratic = more_thuente_quadratic_case_one(best, trial).unwrap_or(cubic);
        if (cubic - best.alpha).abs() < (quadratic - best.alpha).abs() {
            cubic
        } else {
            cubic + 0.5 * (quadratic - cubic)
        }
    } else if signed_derivative < 0.0 {
        state.bracketed = true;
        let cubic = more_thuente_cubic_case_two(profile, best, trial).unwrap_or(trial.alpha);
        let secant = more_thuente_secant(best, trial).unwrap_or(cubic);
        if (cubic - trial.alpha).abs() > (secant - trial.alpha).abs() {
            cubic
        } else {
            secant
        }
    } else if trial.derivative.abs() < best.derivative.abs() {
        bounded = true;
        let cubic = more_thuente_cubic_case_three(profile, best, trial);
        let secant = more_thuente_secant(best, trial);
        match (cubic, secant) {
            (Some(cubic), Some(secant)) if state.bracketed => {
                if (cubic - trial.alpha).abs() < (secant - trial.alpha).abs() {
                    cubic
                } else {
                    secant
                }
            }
            (Some(cubic), Some(secant)) => {
                if (cubic - trial.alpha).abs() > (secant - trial.alpha).abs() {
                    cubic
                } else {
                    secant
                }
            }
            (Some(cubic), None) => cubic,
            (None, Some(secant)) => secant,
            (None, None) if trial.alpha > best.alpha => step_max,
            (None, None) => step_min,
        }
    } else if state.bracketed {
        more_thuente_cubic_case_four(profile, state.other, trial)
            .unwrap_or(0.5 * (trial.alpha + state.other.alpha))
    } else if trial.alpha > best.alpha {
        step_max
    } else {
        step_min
    };

    if trial.value > best.value {
        state.other = trial;
    } else {
        if signed_derivative < 0.0 {
            state.other = best;
        }
        state.best = trial;
    }

    next_alpha = next_alpha.clamp(step_min, step_max);
    if state.bracketed && bounded {
        let bracket_limit =
            state.best.alpha + MORE_THUENTE_BRACKET_SHRINK * (state.other.alpha - state.best.alpha);
        if state.other.alpha > state.best.alpha {
            next_alpha = next_alpha.min(bracket_limit);
        } else {
            next_alpha = next_alpha.max(bracket_limit);
        }
    }

    if next_alpha.is_finite() {
        next_alpha
    } else {
        0.5 * (step_min + step_max)
    }
}

pub(in crate::backend::native) fn more_thuente_cubic_case_one(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    let p = (gamma - best.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - best.derivative) + gamma) + trial.derivative;
    finite_ratio(p, q).map(|ratio| profile.mul_add(ratio, delta, best.alpha))
}

pub(in crate::backend::native) fn more_thuente_quadratic_case_one(
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let denominator = (best.value - trial.value) / delta + best.derivative;
    finite_ratio(0.5 * best.derivative, denominator).map(|ratio| best.alpha + ratio * delta)
}

pub(in crate::backend::native) fn more_thuente_cubic_case_two(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + best.derivative;
    finite_ratio(p, q).map(|ratio| profile.mul_add(ratio, best.alpha - trial.alpha, trial.alpha))
}

pub(in crate::backend::native) fn more_thuente_cubic_case_three(
    profile: ArithmeticProfile,
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(best, trial, delta);
    let q = (gamma + (best.derivative - trial.derivative)) + gamma;
    finite_ratio(p, q)
        .filter(|ratio| *ratio < 0.0 && gamma != 0.0)
        .map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

pub(in crate::backend::native) fn more_thuente_cubic_case_four(
    profile: ArithmeticProfile,
    other: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    let delta = trial.alpha - other.alpha;
    let mut gamma = more_thuente_gamma(
        profile,
        other.value,
        other.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(other, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + other.derivative;
    finite_ratio(p, q).map(|ratio| profile.mul_add(ratio, other.alpha - trial.alpha, trial.alpha))
}

pub(in crate::backend::native) fn more_thuente_secant(
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
) -> Option<f64> {
    finite_ratio(trial.derivative, trial.derivative - best.derivative)
        .map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

pub(in crate::backend::native) fn more_thuente_theta(
    best: MoreThuentePoint,
    trial: MoreThuentePoint,
    delta: f64,
) -> f64 {
    3.0 * (best.value - trial.value) / delta + best.derivative + trial.derivative
}

pub(in crate::backend::native) fn more_thuente_gamma(
    profile: ArithmeticProfile,
    best_value: f64,
    best_derivative: f64,
    trial_value: f64,
    trial_derivative: f64,
    delta: f64,
) -> Option<f64> {
    let theta = 3.0 * (best_value - trial_value) / delta + best_derivative + trial_derivative;
    let scale = theta
        .abs()
        .max(best_derivative.abs())
        .max(trial_derivative.abs());
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }

    let normalized_theta = theta / scale;
    let discriminant = profile.mul_add(
        normalized_theta,
        normalized_theta,
        -((best_derivative / scale) * (trial_derivative / scale)),
    );
    if !discriminant.is_finite() {
        return None;
    }
    let mut gamma = scale * discriminant.max(0.0).sqrt();
    if delta < 0.0 {
        gamma = -gamma;
    }
    Some(gamma)
}

pub(in crate::backend::native) fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let ratio = numerator / denominator;
    ratio.is_finite().then_some(ratio)
}
