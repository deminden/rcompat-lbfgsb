use super::{BackendControl, BackendProblem, BackendResult, LbfgsbBackend};
use crate::error::OptimError;
use crate::result::OptimCounts;

const ARMIJO: f64 = 1e-4;
const BACKTRACK: f64 = 0.4995;
const BOUND_TOL: f64 = 1e-12;
const CURVATURE_EPS: f64 = 1e-12;
const FINITE_DIFF_PROJECTED_GRADIENT_NOISE: f64 = 1e-8;
const HISTORY_CURVATURE_EPS: f64 = f64::EPSILON;
const INTERPOLATION_DAMPING: f64 = 0.999;
const MAX_LINE_SEARCH_TRIALS: usize = 20;
const MAIN_PATH_MIN_STEP: f64 = 1e-16;
const MIN_STEP: f64 = 1e-14;
const MORE_THUENTE_BRACKET_SHRINK: f64 = 0.66;
const MORE_THUENTE_FTOL: f64 = 1e-3;
const MORE_THUENTE_XTRAPU: f64 = 4.0;
#[cfg(test)]
const STRONG_WOLFE_UNBOUNDED_MAX_STEP: f64 = 1.0e20;
const WOLFE_CURVATURE: f64 = 0.9;

/// Native in-tree bounded limited-memory quasi-Newton backend.
#[derive(Debug, Default)]
pub(crate) struct NativeBackend;

#[derive(Debug, Clone)]
struct Correction {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
}

impl LbfgsbBackend for NativeBackend {
    fn minimize<P>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
    ) -> Result<BackendResult, OptimError>
    where
        P: BackendProblem,
    {
        self.minimize_with_modes(
            problem,
            initial,
            lower,
            upper,
            control,
            BackendModes::for_problem(control, initial.len()),
        )
    }
}

impl NativeBackend {
    fn minimize_with_modes<P>(
        &mut self,
        problem: &mut P,
        initial: &[f64],
        lower: &[f64],
        upper: &[f64],
        control: BackendControl,
        modes: BackendModes,
    ) -> Result<BackendResult, OptimError>
    where
        P: BackendProblem,
    {
        let mut x = project(initial, lower, upper);
        let mut counts = OptimCounts::default();
        let Evaluation {
            mut value,
            mut gradient,
        } = evaluate(problem, &x, &mut counts).map_err(initial_evaluation_error)?;
        let mut history = Vec::<Correction>::new();
        let factr_tolerance = if control.factr > 0.0 {
            Some(control.factr * f64::EPSILON)
        } else {
            None
        };
        let mut deferred_exact_zero_pgtol = false;
        let mut used_multidimensional_interpolation = false;
        let has_infinite_bound = lower
            .iter()
            .chain(upper.iter())
            .any(|value| value.is_infinite());
        let has_finite_bound = lower
            .iter()
            .chain(upper.iter())
            .any(|value| value.is_finite());

        let min_step = min_step_for_modes(modes);

        for iteration in 1..=control.maxit.saturating_add(1) {
            let projected_norm = projected_gradient_norm(&x, &gradient, lower, upper);
            if should_stop_for_projected_gradient(
                projected_norm,
                control.pgtol,
                control.has_user_gradient,
                x.len(),
                has_infinite_bound,
                has_finite_bound,
                &mut deferred_exact_zero_pgtol,
            ) {
                if deferred_exact_zero_pgtol
                    && control.pgtol == 0.0
                    && has_infinite_bound
                    && x.len() > 1
                {
                    let evaluation = evaluate(problem, &x, &mut counts)?;
                    value = evaluation.value;
                }
                return Ok(success(
                    x,
                    value,
                    counts,
                    "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                ));
            }

            maybe_trace_cauchy_point(&x, &gradient, lower, upper, &history, control);

            let mut direction = direction_with_mode(
                &x,
                &gradient,
                lower,
                upper,
                &history,
                modes.direction,
                min_step,
            );

            let directional_derivative = dot(&gradient, &direction);
            if directional_derivative >= 0.0 || norm_inf(&direction) <= min_step {
                direction = steepest_projected_direction(&x, &gradient, lower, upper);
            }

            if dot(&gradient, &direction) >= 0.0 || norm_inf(&direction) <= min_step {
                if deferred_exact_zero_pgtol
                    && control.pgtol == 0.0
                    && has_infinite_bound
                    && x.len() > 1
                {
                    let evaluation = evaluate(problem, &x, &mut counts)?;
                    value = evaluation.value;
                }
                return Ok(success(
                    x,
                    value,
                    counts,
                    "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                ));
            }

            let request = LineSearchRequest {
                x: &x,
                value,
                gradient: &gradient,
                direction: &direction,
                lower,
                upper,
                // R's bundled L-BFGS-B caps the first constrained line search at stp = 1.
                max_step_cap: if iteration == 1 && has_finite_bound {
                    Some(1.0)
                } else {
                    None
                },
                cap_initial_unbounded_step: history.is_empty()
                    && lower.iter().all(|value| value.is_infinite())
                    && upper.iter().all(|value| value.is_infinite()),
                initial_step_cap: initial_step_cap_for_modes(modes, history.is_empty()),
                allow_quadratic_interpolation: history.is_empty(),
                min_step,
                quadratic_interpolation_damping: quadratic_interpolation_damping(control, x.len()),
            };
            let Some(step) =
                line_search_with_mode(problem, request, &mut counts, modes.line_search, iteration)?
            else {
                return Ok(BackendResult {
                    x,
                    value,
                    counts,
                    convergence: 52,
                    message: "ERROR: ABNORMAL_TERMINATION_IN_LNSRCH".to_string(),
                });
            };

            let relative_reduction =
                (value - step.value).abs() / value.abs().max(step.value.abs()).max(1.0);
            let next_projected_norm =
                projected_gradient_norm(&step.x, &step.gradient, lower, upper);
            maybe_trace(iteration, step.value, next_projected_norm, &step, control);

            update_history(
                &mut history,
                control.lmm,
                difference(&step.x, &x),
                difference(&step.gradient, &gradient),
                &gradient,
            );

            used_multidimensional_interpolation |= step.used_multidimensional_interpolation;
            x = step.x;
            value = step.value;
            gradient = step.gradient;

            if iteration > control.maxit {
                return Ok(iteration_limit(x, value, counts));
            }

            if should_stop_for_projected_gradient(
                next_projected_norm,
                control.pgtol,
                control.has_user_gradient,
                x.len(),
                has_infinite_bound,
                has_finite_bound,
                &mut deferred_exact_zero_pgtol,
            ) {
                if control.has_user_gradient && used_multidimensional_interpolation {
                    let evaluation = evaluate(problem, &x, &mut counts)?;
                    value = evaluation.value;
                }
                return Ok(success(
                    x,
                    value,
                    counts,
                    "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
                ));
            }

            if factr_tolerance.is_some_and(|tolerance| relative_reduction <= tolerance) {
                return Ok(success(
                    x,
                    value,
                    counts,
                    "CONVERGENCE: REL_REDUCTION_OF_F <= FACTR*EPSMCH",
                ));
            }
        }

        Ok(iteration_limit(x, value, counts))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineSearchMode {
    BacktrackingArmijo,
    MoreThuente,
    #[cfg(test)]
    StrongWolfe,
    #[cfg(test)]
    MoreThuenteFirstThenBacktracking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BackendModes {
    line_search: LineSearchMode,
    direction: DirectionMode,
}

impl Default for BackendModes {
    fn default() -> Self {
        Self {
            line_search: LineSearchMode::MoreThuente,
            direction: DirectionMode::CauchySubspace,
        }
    }
}

impl BackendModes {
    fn for_problem(control: BackendControl, dimension: usize) -> Self {
        if control.has_user_gradient && dimension > 1 {
            Self::default()
        } else {
            Self {
                line_search: LineSearchMode::BacktrackingArmijo,
                direction: DirectionMode::ProjectedLbfgs,
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirectionMode {
    ProjectedLbfgs,
    CauchySubspace,
    #[cfg(test)]
    CauchyFirstThenProjected,
    #[cfg(test)]
    CauchySubspaceCappedFirstStep,
    #[cfg(test)]
    CauchyFirstThenProjectedCapped,
}

fn initial_step_cap_for_modes(modes: BackendModes, history_is_empty: bool) -> Option<f64> {
    if !history_is_empty {
        return None;
    }

    match modes.direction {
        DirectionMode::ProjectedLbfgs => None,
        DirectionMode::CauchySubspace => None,
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjected => None,
        #[cfg(test)]
        DirectionMode::CauchySubspaceCappedFirstStep
        | DirectionMode::CauchyFirstThenProjectedCapped => Some(0.1),
    }
}

fn min_step_for_modes(modes: BackendModes) -> f64 {
    match modes.line_search {
        LineSearchMode::MoreThuente => MAIN_PATH_MIN_STEP,
        LineSearchMode::BacktrackingArmijo => MIN_STEP,
        #[cfg(test)]
        LineSearchMode::StrongWolfe | LineSearchMode::MoreThuenteFirstThenBacktracking => MIN_STEP,
    }
}

fn quadratic_interpolation_damping(control: BackendControl, dimension: usize) -> f64 {
    if !control.has_user_gradient && dimension > 1 && (control.pgtol > 0.0 || control.maxit == 0) {
        1.0
    } else {
        INTERPOLATION_DAMPING
    }
}

#[derive(Debug, Clone)]
struct Evaluation {
    value: f64,
    gradient: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Step {
    x: Vec<f64>,
    value: f64,
    gradient: Vec<f64>,
    line_search_trials: usize,
    alpha: f64,
    max_alpha: f64,
    step_norm: f64,
    curvature_ratio: f64,
    wolfe_curvature_satisfied: bool,
    used_multidimensional_interpolation: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct CauchyPoint {
    x: Vec<f64>,
    active_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SubspacePoint {
    x: Vec<f64>,
    free_count: usize,
    clipped: bool,
}

struct LineSearchRequest<'a> {
    x: &'a [f64],
    value: f64,
    gradient: &'a [f64],
    direction: &'a [f64],
    lower: &'a [f64],
    upper: &'a [f64],
    max_step_cap: Option<f64>,
    cap_initial_unbounded_step: bool,
    initial_step_cap: Option<f64>,
    allow_quadratic_interpolation: bool,
    min_step: f64,
    quadratic_interpolation_damping: f64,
}

fn evaluate<P>(
    problem: &mut P,
    x: &[f64],
    counts: &mut OptimCounts,
) -> Result<Evaluation, OptimError>
where
    P: BackendProblem,
{
    let (value, gradient) = problem.value_and_gradient(x)?;
    counts.function += 1;
    counts.gradient += 1;
    Ok(Evaluation { value, gradient })
}

fn initial_evaluation_error(error: OptimError) -> OptimError {
    match error {
        OptimError::NonFiniteObjective { value } => OptimError::NonFiniteInitialValue { value },
        other => other,
    }
}

fn should_stop_for_projected_gradient(
    projected_norm: f64,
    pgtol: f64,
    has_user_gradient: bool,
    dimension: usize,
    has_infinite_bound: bool,
    has_finite_bound: bool,
    deferred_exact_zero_pgtol: &mut bool,
) -> bool {
    let zero_pgtol_noise_floor = if pgtol == 0.0 && has_infinite_bound {
        if has_user_gradient {
            Some(1e-12)
        } else if has_finite_bound {
            Some(FINITE_DIFF_PROJECTED_GRADIENT_NOISE)
        } else {
            None
        }
    } else {
        None
    };
    let compatible_norm =
        if zero_pgtol_noise_floor.is_some_and(|noise_floor| projected_norm <= noise_floor) {
            0.0
        } else {
            projected_norm
        };

    if compatible_norm > pgtol {
        return false;
    }

    if pgtol == 0.0
        && compatible_norm == 0.0
        && dimension > 1
        && has_infinite_bound
        && has_user_gradient
        && !*deferred_exact_zero_pgtol
    {
        *deferred_exact_zero_pgtol = true;
        return false;
    }

    true
}

fn success(x: Vec<f64>, value: f64, counts: OptimCounts, message: &str) -> BackendResult {
    BackendResult {
        x,
        value,
        counts,
        convergence: 0,
        message: message.to_string(),
    }
}

fn iteration_limit(x: Vec<f64>, value: f64, counts: OptimCounts) -> BackendResult {
    BackendResult {
        x,
        value,
        counts,
        convergence: 1,
        message: "NEW_X".to_string(),
    }
}

fn project(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|((&value, &lower), &upper)| value.clamp(lower, upper))
        .collect()
}

fn line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let mut max_alpha =
        max_feasible_step(request.x, request.direction, request.lower, request.upper);
    if let Some(cap) = request.max_step_cap {
        max_alpha = max_alpha.min(cap);
    }
    let mut alpha = max_alpha.min(1.0);
    if request.cap_initial_unbounded_step {
        alpha = alpha.min(1.0 / norm_inf(request.direction).max(1.0));
    }
    if let Some(cap) = request.initial_step_cap {
        alpha = alpha.min(cap);
    }
    if alpha <= 0.0 || !alpha.is_finite() {
        return Ok(None);
    }
    let initial_alpha = alpha;
    let initial_directional_derivative = dot(request.gradient, request.direction);
    let mut tried_quadratic_step = false;
    let mut used_multidimensional_interpolation = false;
    let mut evaluated_candidates = 0_usize;

    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        let candidate = bounded_step(
            request.x,
            request.direction,
            alpha,
            request.lower,
            request.upper,
        );
        let step = difference(&candidate, request.x);

        if norm_inf(&step) <= request.min_step {
            alpha *= BACKTRACK;
            continue;
        }

        let directional_decrease = dot(request.gradient, &step);
        if directional_decrease < 0.0 {
            let evaluation = evaluate(problem, &candidate, counts)?;
            evaluated_candidates += 1;
            if evaluation.value <= request.value + ARMIJO * directional_decrease {
                let directional_derivative = dot(&evaluation.gradient, request.direction);
                let curvature_ratio = directional_derivative.abs()
                    / (-initial_directional_derivative).max(f64::MIN_POSITIVE);
                return Ok(Some(Step {
                    x: candidate,
                    value: evaluation.value,
                    gradient: evaluation.gradient,
                    line_search_trials: evaluated_candidates.saturating_sub(1),
                    alpha,
                    max_alpha,
                    step_norm: norm2(&step),
                    curvature_ratio,
                    wolfe_curvature_satisfied: curvature_ratio <= WOLFE_CURVATURE,
                    used_multidimensional_interpolation,
                }));
            }
            if !tried_quadratic_step
                && (request.x.len() == 1
                    || (request.allow_quadratic_interpolation && initial_alpha >= 0.1))
            {
                tried_quadratic_step = true;
                if let Some(interpolated_alpha) = quadratic_trial_alpha(
                    request.value,
                    initial_directional_derivative,
                    initial_alpha,
                    evaluation.value,
                    request.quadratic_interpolation_damping,
                ) {
                    used_multidimensional_interpolation = request.x.len() > 1;
                    alpha = interpolated_alpha;
                    continue;
                }
            }
        }

        alpha *= BACKTRACK;
    }

    Ok(None)
}

fn line_search_with_mode<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
    mode: LineSearchMode,
    iteration: usize,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let _ = iteration;
    match mode {
        LineSearchMode::BacktrackingArmijo => line_search(problem, request, counts),
        LineSearchMode::MoreThuente => more_thuente_line_search(problem, request, counts),
        #[cfg(test)]
        LineSearchMode::StrongWolfe => strong_wolfe_line_search(problem, request, counts),
        #[cfg(test)]
        LineSearchMode::MoreThuenteFirstThenBacktracking => {
            if iteration == 1 {
                more_thuente_line_search(problem, request, counts)
            } else {
                line_search(problem, request, counts)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MoreThuentePoint {
    alpha: f64,
    value: f64,
    derivative: f64,
}

fn more_thuente_line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let mut max_alpha =
        max_feasible_step(request.x, request.direction, request.lower, request.upper);
    if let Some(cap) = request.max_step_cap {
        max_alpha = max_alpha.min(cap);
    }
    let mut alpha = max_alpha.min(1.0);
    if request.cap_initial_unbounded_step {
        alpha = alpha.min(1.0 / norm_inf(request.direction).max(1.0));
    }
    if let Some(cap) = request.initial_step_cap {
        alpha = alpha.min(cap);
    }

    let initial_derivative = dot(request.gradient, request.direction);
    if alpha <= 0.0 || !alpha.is_finite() || initial_derivative >= 0.0 {
        return Ok(None);
    }

    let mut best = MoreThuentePoint {
        alpha: 0.0,
        value: request.value,
        derivative: initial_derivative,
    };
    let mut other = best;
    let mut bracketed = false;
    let mut stage_one = true;
    let mut width = max_alpha;
    let mut previous_width = 2.0 * width;
    let mut evaluated_candidates = 0_usize;
    let decrease_test = MORE_THUENTE_FTOL * initial_derivative;

    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        let candidate = bounded_step(
            request.x,
            request.direction,
            alpha,
            request.lower,
            request.upper,
        );
        let step = difference(&candidate, request.x);
        if norm_inf(&step) <= request.min_step {
            alpha = if bracketed {
                0.5 * (best.alpha + other.alpha)
            } else {
                (alpha * BACKTRACK).max(request.min_step)
            };
            continue;
        }

        let evaluation = evaluate(problem, &candidate, counts)?;
        evaluated_candidates += 1;
        let derivative = dot(&evaluation.gradient, request.direction);
        let point = MoreThuentePoint {
            alpha,
            value: evaluation.value,
            derivative,
        };
        let curvature_ratio = derivative.abs() / (-initial_derivative).max(f64::MIN_POSITIVE);
        if evaluation.value <= more_thuente_sufficient_decrease(request.value, decrease_test, alpha)
            && curvature_ratio <= WOLFE_CURVATURE
        {
            return Ok(Some(Step {
                x: candidate,
                value: evaluation.value,
                gradient: evaluation.gradient,
                line_search_trials: evaluated_candidates.saturating_sub(1),
                alpha,
                max_alpha,
                step_norm: norm2(&step),
                curvature_ratio,
                wolfe_curvature_satisfied: true,
                used_multidimensional_interpolation: false,
            }));
        }

        let sufficient_decrease =
            more_thuente_sufficient_decrease(request.value, decrease_test, alpha);
        if alpha == max_alpha
            && evaluation.value <= sufficient_decrease
            && derivative <= decrease_test
        {
            return Ok(Some(Step {
                x: candidate,
                value: evaluation.value,
                gradient: evaluation.gradient,
                line_search_trials: evaluated_candidates.saturating_sub(1),
                alpha,
                max_alpha,
                step_norm: norm2(&step),
                curvature_ratio,
                wolfe_curvature_satisfied: false,
                used_multidimensional_interpolation: false,
            }));
        }

        if more_thuente_enters_stage_two(
            stage_one,
            evaluation.value,
            sufficient_decrease,
            derivative,
        ) {
            stage_one = false;
        }

        let (step_min, step_max) =
            more_thuente_safeguard_interval(best.alpha, other.alpha, alpha, max_alpha, bracketed);

        let next_alpha = if stage_one
            && evaluation.value <= best.value
            && evaluation.value > sufficient_decrease
        {
            let modified_best = MoreThuentePoint {
                alpha: best.alpha,
                value: best.value - best.alpha * decrease_test,
                derivative: best.derivative - decrease_test,
            };
            let modified_other = MoreThuentePoint {
                alpha: other.alpha,
                value: other.value - other.alpha * decrease_test,
                derivative: other.derivative - decrease_test,
            };
            let modified_trial = MoreThuentePoint {
                alpha,
                value: evaluation.value - alpha * decrease_test,
                derivative: derivative - decrease_test,
            };
            let mut step_state = MoreThuenteStepState {
                best: modified_best,
                other: modified_other,
                bracketed,
            };
            let next_alpha = more_thuente_step(&mut step_state, modified_trial, step_min, step_max);
            best = MoreThuentePoint {
                alpha: step_state.best.alpha,
                value: step_state.best.value + step_state.best.alpha * decrease_test,
                derivative: step_state.best.derivative + decrease_test,
            };
            other = MoreThuentePoint {
                alpha: step_state.other.alpha,
                value: step_state.other.value + step_state.other.alpha * decrease_test,
                derivative: step_state.other.derivative + decrease_test,
            };
            bracketed = step_state.bracketed;
            next_alpha
        } else {
            let mut step_state = MoreThuenteStepState {
                best,
                other,
                bracketed,
            };
            let next_alpha = more_thuente_step(&mut step_state, point, step_min, step_max);
            best = step_state.best;
            other = step_state.other;
            bracketed = step_state.bracketed;
            next_alpha
        };

        alpha = next_alpha.clamp(0.0, max_alpha);
        if bracketed {
            if (other.alpha - best.alpha).abs() >= MORE_THUENTE_BRACKET_SHRINK * previous_width {
                alpha = best.alpha + 0.5 * (other.alpha - best.alpha);
            }
            previous_width = width;
            width = (other.alpha - best.alpha).abs();
        }

        if alpha <= 0.0 || !alpha.is_finite() {
            return Ok(None);
        }
    }

    Ok(None)
}

fn more_thuente_sufficient_decrease(value_at_zero: f64, decrease_test: f64, alpha: f64) -> f64 {
    value_at_zero + alpha * decrease_test
}

fn more_thuente_enters_stage_two(
    stage_one: bool,
    value: f64,
    sufficient_decrease: f64,
    derivative: f64,
) -> bool {
    stage_one && value <= sufficient_decrease && derivative >= 0.0
}

#[derive(Debug, Clone, Copy)]
struct MoreThuenteStepState {
    best: MoreThuentePoint,
    other: MoreThuentePoint,
    bracketed: bool,
}

fn more_thuente_safeguard_interval(
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

fn more_thuente_step(
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
        let cubic = more_thuente_cubic_case_one(best, trial).unwrap_or(trial.alpha);
        let quadratic = more_thuente_quadratic_case_one(best, trial).unwrap_or(cubic);
        if (cubic - best.alpha).abs() < (quadratic - best.alpha).abs() {
            cubic
        } else {
            cubic + 0.5 * (quadratic - cubic)
        }
    } else if signed_derivative < 0.0 {
        state.bracketed = true;
        let cubic = more_thuente_cubic_case_two(best, trial).unwrap_or(trial.alpha);
        let secant = more_thuente_secant(best, trial).unwrap_or(cubic);
        if (cubic - trial.alpha).abs() > (secant - trial.alpha).abs() {
            cubic
        } else {
            secant
        }
    } else if trial.derivative.abs() < best.derivative.abs() {
        bounded = true;
        let cubic = more_thuente_cubic_case_three(best, trial);
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
        more_thuente_cubic_case_four(state.other, trial)
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

fn more_thuente_cubic_case_one(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let gamma = more_thuente_gamma(
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    let p = (gamma - best.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - best.derivative) + gamma) + trial.derivative;
    finite_ratio(p, q).map(|ratio| best.alpha + ratio * delta)
}

fn more_thuente_quadratic_case_one(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let denominator = (best.value - trial.value) / delta + best.derivative;
    finite_ratio(0.5 * best.derivative, denominator).map(|ratio| best.alpha + ratio * delta)
}

fn more_thuente_cubic_case_two(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
        best.value,
        best.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(best, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + best.derivative;
    finite_ratio(p, q).map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

fn more_thuente_cubic_case_three(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    let delta = trial.alpha - best.alpha;
    let mut gamma = more_thuente_gamma(
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

fn more_thuente_cubic_case_four(other: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    let delta = trial.alpha - other.alpha;
    let mut gamma = more_thuente_gamma(
        other.value,
        other.derivative,
        trial.value,
        trial.derivative,
        delta,
    )?;
    gamma = -gamma;
    let p = (gamma - trial.derivative) + more_thuente_theta(other, trial, delta);
    let q = ((gamma - trial.derivative) + gamma) + other.derivative;
    finite_ratio(p, q).map(|ratio| trial.alpha + ratio * (other.alpha - trial.alpha))
}

fn more_thuente_secant(best: MoreThuentePoint, trial: MoreThuentePoint) -> Option<f64> {
    finite_ratio(trial.derivative, trial.derivative - best.derivative)
        .map(|ratio| trial.alpha + ratio * (best.alpha - trial.alpha))
}

fn more_thuente_theta(best: MoreThuentePoint, trial: MoreThuentePoint, delta: f64) -> f64 {
    3.0 * (best.value - trial.value) / delta + best.derivative + trial.derivative
}

fn more_thuente_gamma(
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

    let discriminant =
        (theta / scale).powi(2) - (best_derivative / scale) * (trial_derivative / scale);
    if !discriminant.is_finite() {
        return None;
    }
    let mut gamma = scale * discriminant.max(0.0).sqrt();
    if delta < 0.0 {
        gamma = -gamma;
    }
    Some(gamma)
}

fn finite_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let ratio = numerator / denominator;
    ratio.is_finite().then_some(ratio)
}

#[cfg(test)]
#[derive(Debug, Clone)]
struct StrongWolfeCandidate {
    point: LineSearchPoint1D,
    x: Vec<f64>,
    step: Vec<f64>,
    evaluation: Evaluation,
}

#[cfg(test)]
fn strong_wolfe_line_search<P>(
    problem: &mut P,
    request: LineSearchRequest<'_>,
    counts: &mut OptimCounts,
) -> Result<Option<Step>, OptimError>
where
    P: BackendProblem,
{
    let max_alpha = max_feasible_step_allowing_unbounded(
        request.x,
        request.direction,
        request.lower,
        request.upper,
    );
    let mut initial_alpha = max_alpha.min(1.0);
    if request.cap_initial_unbounded_step {
        initial_alpha = initial_alpha.min(1.0 / norm_inf(request.direction).max(1.0));
    }
    if let Some(cap) = request.initial_step_cap {
        initial_alpha = initial_alpha.min(cap);
    }

    let initial_directional_derivative = dot(request.gradient, request.direction);
    if initial_alpha <= 0.0 || initial_directional_derivative >= 0.0 {
        return Ok(None);
    }

    let mut last_candidate = None;
    let mut evaluations = 0_usize;
    let result = strong_wolfe_search(
        initial_alpha,
        max_alpha,
        request.value,
        initial_directional_derivative,
        |alpha| {
            let candidate = bounded_step(
                request.x,
                request.direction,
                alpha,
                request.lower,
                request.upper,
            );
            let step = difference(&candidate, request.x);
            let evaluation = evaluate(problem, &candidate, counts)?;
            evaluations += 1;
            let derivative = dot(&evaluation.gradient, request.direction);
            let point = LineSearchPoint1D {
                alpha,
                value: evaluation.value,
                derivative,
            };
            last_candidate = Some(StrongWolfeCandidate {
                point,
                x: candidate,
                step,
                evaluation,
            });
            Ok(point)
        },
    )?;

    let Some(result) = result else {
        let Some(candidate) = last_candidate else {
            return Ok(None);
        };
        if candidate.point.alpha == max_alpha
            && candidate.point.value
                <= armijo_value(
                    request.value,
                    initial_directional_derivative,
                    candidate.point.alpha,
                )
        {
            return Ok(Some(strong_wolfe_step_from_candidate(
                candidate,
                max_alpha,
                evaluations.saturating_sub(1),
                initial_directional_derivative,
            )));
        }
        return Ok(None);
    };

    let Some(candidate) = last_candidate else {
        return Ok(None);
    };
    debug_assert_eq!(candidate.point, result.point);

    Ok(Some(strong_wolfe_step_from_candidate(
        candidate,
        max_alpha,
        result.evaluations.saturating_sub(1),
        initial_directional_derivative,
    )))
}

#[cfg(test)]
fn strong_wolfe_step_from_candidate(
    candidate: StrongWolfeCandidate,
    max_alpha: f64,
    line_search_trials: usize,
    initial_directional_derivative: f64,
) -> Step {
    let curvature_ratio =
        candidate.point.derivative.abs() / (-initial_directional_derivative).max(f64::MIN_POSITIVE);

    Step {
        x: candidate.x,
        value: candidate.evaluation.value,
        gradient: candidate.evaluation.gradient,
        line_search_trials,
        alpha: candidate.point.alpha,
        max_alpha,
        step_norm: norm2(&candidate.step),
        curvature_ratio,
        wolfe_curvature_satisfied: curvature_ratio <= WOLFE_CURVATURE,
        used_multidimensional_interpolation: false,
    }
}

fn quadratic_trial_alpha(
    value_at_zero: f64,
    derivative_at_zero: f64,
    trial_alpha: f64,
    trial_value: f64,
    damping: f64,
) -> Option<f64> {
    let denominator = 2.0 * (trial_value - value_at_zero - derivative_at_zero * trial_alpha);
    if denominator <= 0.0 || !denominator.is_finite() {
        return None;
    }

    let alpha = -derivative_at_zero * trial_alpha * trial_alpha / denominator;
    if alpha > 0.0 && alpha < trial_alpha && alpha.is_finite() {
        if trial_alpha >= 1.0 {
            Some(alpha * damping)
        } else {
            Some(alpha)
        }
    } else {
        None
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct LineSearchPoint1D {
    alpha: f64,
    value: f64,
    derivative: f64,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct StrongWolfeResult {
    point: LineSearchPoint1D,
    evaluations: usize,
}

#[cfg(test)]
fn strong_wolfe_search<F>(
    initial_alpha: f64,
    max_alpha: f64,
    value_at_zero: f64,
    derivative_at_zero: f64,
    mut evaluate: F,
) -> Result<Option<StrongWolfeResult>, OptimError>
where
    F: FnMut(f64) -> Result<LineSearchPoint1D, OptimError>,
{
    if initial_alpha <= 0.0
        || max_alpha <= 0.0
        || derivative_at_zero >= 0.0
        || !initial_alpha.is_finite()
        || !max_alpha.is_finite()
    {
        return Ok(None);
    }

    let mut evaluations = 0;
    let mut previous_alpha = 0.0;
    let mut previous_value = value_at_zero;
    let mut alpha = initial_alpha.min(max_alpha);

    for iteration in 0..MAX_LINE_SEARCH_TRIALS {
        let point = evaluate(alpha)?;
        evaluations += 1;

        if !point.value.is_finite() || !point.derivative.is_finite() {
            return Ok(None);
        }

        if point.value > armijo_value(value_at_zero, derivative_at_zero, alpha)
            || (iteration > 0 && point.value >= previous_value)
        {
            return strong_wolfe_zoom(
                previous_alpha,
                alpha,
                value_at_zero,
                derivative_at_zero,
                previous_value,
                evaluate,
                evaluations,
            );
        }

        if point.derivative.abs() <= -WOLFE_CURVATURE * derivative_at_zero {
            return Ok(Some(StrongWolfeResult { point, evaluations }));
        }

        if point.derivative >= 0.0 {
            return strong_wolfe_zoom(
                alpha,
                previous_alpha,
                value_at_zero,
                derivative_at_zero,
                point.value,
                evaluate,
                evaluations,
            );
        }

        if alpha >= max_alpha {
            return Ok(None);
        }

        previous_alpha = alpha;
        previous_value = point.value;
        alpha = (2.0 * alpha).min(max_alpha);
    }

    Ok(None)
}

#[cfg(test)]
fn strong_wolfe_zoom<F>(
    mut low_alpha: f64,
    mut high_alpha: f64,
    value_at_zero: f64,
    derivative_at_zero: f64,
    mut low_value: f64,
    mut evaluate: F,
    mut evaluations: usize,
) -> Result<Option<StrongWolfeResult>, OptimError>
where
    F: FnMut(f64) -> Result<LineSearchPoint1D, OptimError>,
{
    for _ in 0..MAX_LINE_SEARCH_TRIALS {
        let alpha = 0.5 * (low_alpha + high_alpha);
        if (high_alpha - low_alpha).abs() <= MIN_STEP.max(MIN_STEP * alpha.abs()) {
            return Ok(None);
        }

        let point = evaluate(alpha)?;
        evaluations += 1;

        if !point.value.is_finite() || !point.derivative.is_finite() {
            return Ok(None);
        }

        if point.value > armijo_value(value_at_zero, derivative_at_zero, alpha)
            || point.value >= low_value
        {
            high_alpha = alpha;
            continue;
        }

        if point.derivative.abs() <= -WOLFE_CURVATURE * derivative_at_zero {
            return Ok(Some(StrongWolfeResult { point, evaluations }));
        }

        if point.derivative * (high_alpha - low_alpha) >= 0.0 {
            high_alpha = low_alpha;
        }
        low_alpha = alpha;
        low_value = point.value;
    }

    Ok(None)
}

#[cfg(test)]
fn armijo_value(value_at_zero: f64, derivative_at_zero: f64, alpha: f64) -> f64 {
    value_at_zero + ARMIJO * alpha * derivative_at_zero
}

fn max_feasible_step(x: &[f64], direction: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut alpha = f64::INFINITY;
    for index in 0..x.len() {
        let d = direction[index];
        if d > 0.0 && upper[index].is_finite() {
            alpha = alpha.min((upper[index] - x[index]) / d);
        } else if d < 0.0 && lower[index].is_finite() {
            alpha = alpha.min((lower[index] - x[index]) / d);
        }
    }
    if alpha.is_infinite() {
        1.0
    } else {
        alpha.max(0.0)
    }
}

#[cfg(test)]
fn max_feasible_step_allowing_unbounded(
    x: &[f64],
    direction: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> f64 {
    let mut alpha = f64::INFINITY;
    for index in 0..x.len() {
        let d = direction[index];
        if d > 0.0 && upper[index].is_finite() {
            alpha = alpha.min((upper[index] - x[index]) / d);
        } else if d < 0.0 && lower[index].is_finite() {
            alpha = alpha.min((lower[index] - x[index]) / d);
        }
    }
    if alpha.is_infinite() {
        STRONG_WOLFE_UNBOUNDED_MAX_STEP
    } else {
        alpha.max(0.0)
    }
}

fn bounded_step(
    x: &[f64],
    direction: &[f64],
    alpha: f64,
    lower: &[f64],
    upper: &[f64],
) -> Vec<f64> {
    x.iter()
        .zip(direction.iter())
        .zip(lower.iter())
        .zip(upper.iter())
        .map(|(((&x_i, &d_i), &lower_i), &upper_i)| (x_i + alpha * d_i).clamp(lower_i, upper_i))
        .collect()
}

fn lbfgs_direction(gradient: &[f64], history: &[Correction]) -> Vec<f64> {
    lbfgs_inverse_product(gradient, history)
        .into_iter()
        .map(|value| -value)
        .collect()
}

fn projected_lbfgs_direction(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
) -> Vec<f64> {
    let mut direction = lbfgs_direction(gradient, history);
    project_direction(x, lower, upper, &mut direction);
    direction
}

fn direction_with_mode(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    mode: DirectionMode,
    min_step: f64,
) -> Vec<f64> {
    match mode {
        DirectionMode::ProjectedLbfgs => {
            projected_lbfgs_direction(x, gradient, lower, upper, history)
        }
        DirectionMode::CauchySubspace => {
            cauchy_subspace_direction(x, gradient, lower, upper, history, min_step)
                .unwrap_or_else(|| projected_lbfgs_direction(x, gradient, lower, upper, history))
        }
        #[cfg(test)]
        DirectionMode::CauchySubspaceCappedFirstStep => {
            cauchy_subspace_direction(x, gradient, lower, upper, history, min_step)
                .unwrap_or_else(|| projected_lbfgs_direction(x, gradient, lower, upper, history))
        }
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjected => {
            if history.is_empty() {
                cauchy_subspace_direction(x, gradient, lower, upper, history, min_step)
                    .unwrap_or_else(|| {
                        projected_lbfgs_direction(x, gradient, lower, upper, history)
                    })
            } else {
                projected_lbfgs_direction(x, gradient, lower, upper, history)
            }
        }
        #[cfg(test)]
        DirectionMode::CauchyFirstThenProjectedCapped => {
            if history.is_empty() {
                cauchy_subspace_direction(x, gradient, lower, upper, history, min_step)
                    .unwrap_or_else(|| {
                        projected_lbfgs_direction(x, gradient, lower, upper, history)
                    })
            } else {
                projected_lbfgs_direction(x, gradient, lower, upper, history)
            }
        }
    }
}

fn cauchy_subspace_direction(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    min_step: f64,
) -> Option<Vec<f64>> {
    let cauchy = generalized_cauchy_point_limited_memory(x, gradient, lower, upper, history);
    let target = subspace_minimizer_limited_memory(x, gradient, lower, upper, history, &cauchy)
        .map(|point| point.x)
        .unwrap_or(cauchy.x);
    let direction = difference(&target, x);
    if norm_inf(&direction) <= min_step {
        None
    } else {
        Some(direction)
    }
}

fn lbfgs_inverse_product(vector: &[f64], history: &[Correction]) -> Vec<f64> {
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

#[derive(Debug, Clone)]
struct PreparedHessianUpdate {
    correction_index: usize,
    bs: Vec<f64>,
    sbs: f64,
}

fn lbfgs_hessian_product(vector: &[f64], history: &[Correction]) -> Vec<f64> {
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

fn lbfgs_initial_hessian_scale(history: &[Correction]) -> f64 {
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

fn apply_hessian_update(
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

fn project_direction(x: &[f64], lower: &[f64], upper: &[f64], direction: &mut [f64]) {
    for index in 0..x.len() {
        let fixed = lower[index] == upper[index];
        let below_lower = x[index] <= lower[index] + BOUND_TOL && direction[index] < 0.0;
        let above_upper = x[index] >= upper[index] - BOUND_TOL && direction[index] > 0.0;
        if fixed || below_lower || above_upper {
            direction[index] = 0.0;
        }
    }
}

fn steepest_projected_direction(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Vec<f64> {
    let mut direction: Vec<f64> = gradient.iter().map(|&value| -value).collect();
    project_direction(x, lower, upper, &mut direction);
    direction
}

#[cfg(test)]
fn generalized_cauchy_point_identity(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(x, gradient, lower, upper, |vector| vector.to_vec())
}

fn generalized_cauchy_point_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
) -> CauchyPoint {
    generalized_cauchy_point_with_hessian(x, gradient, lower, upper, |vector| {
        lbfgs_hessian_product(vector, history)
    })
}

fn generalized_cauchy_point_with_hessian<F>(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    mut hessian_product: F,
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
                active_count: active_count(&point, lower, upper),
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
                active_count: active_count(&point, lower, upper),
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
                    active_count: active_count(&point, lower, upper),
                    x: point,
                };
            }
        }

        if !next_time.is_finite() {
            return CauchyPoint {
                active_count: active_count(&point, lower, upper),
                x: point,
            };
        }

        advance_cauchy_point(&mut point, &direction, interval.max(0.0), lower, upper);
        previous_time = next_time;

        while cursor < breakpoints.len() && (breakpoints[cursor].0 - next_time).abs() <= BOUND_TOL {
            free[breakpoints[cursor].1] = false;
            cursor += 1;
        }
    }
}

fn cauchy_path_direction(gradient: &[f64], free: &[bool]) -> Vec<f64> {
    gradient
        .iter()
        .zip(free.iter())
        .map(|(&gradient_i, &free_i)| if free_i { -gradient_i } else { 0.0 })
        .collect()
}

fn advance_cauchy_point(
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

fn active_count(point: &[f64], lower: &[f64], upper: &[f64]) -> usize {
    let mut count = 0;
    for index in 0..point.len() {
        if is_active_at_bound(point[index], lower[index], upper[index]) {
            count += 1;
        }
    }
    count
}

fn is_active_at_bound(value: f64, lower: f64, upper: f64) -> bool {
    lower == upper
        || (value <= lower + BOUND_TOL && lower.is_finite())
        || (value >= upper - BOUND_TOL && upper.is_finite())
}

fn subspace_minimizer_limited_memory(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    cauchy: &CauchyPoint,
) -> Option<SubspacePoint> {
    subspace_minimizer_with_hessian(x, gradient, lower, upper, cauchy, |vector| {
        lbfgs_hessian_product(vector, history)
    })
}

fn subspace_minimizer_with_hessian<F>(
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
    let free_indices = free_indices(&cauchy.x, lower, upper);
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

    let alpha = max_feasible_step(&cauchy.x, &direction, lower, upper).min(1.0);
    let clipped = alpha < 1.0;
    let point = bounded_step(&cauchy.x, &direction, alpha, lower, upper);

    Some(SubspacePoint {
        x: point,
        free_count: free_indices.len(),
        clipped,
    })
}

fn free_indices(point: &[f64], lower: &[f64], upper: &[f64]) -> Vec<usize> {
    (0..point.len())
        .filter(|&index| !is_active_at_bound(point[index], lower[index], upper[index]))
        .collect()
}

fn dense_subspace_hessian<F>(
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

fn solve_positive_definite(matrix: Vec<Vec<f64>>, rhs: Vec<f64>) -> Option<Vec<f64>> {
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

fn projected_gradient_norm(x: &[f64], gradient: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut max_norm: f64 = 0.0;
    for index in 0..x.len() {
        let fixed = lower[index] == upper[index];
        let lower_is_optimal = x[index] <= lower[index] + BOUND_TOL && gradient[index] >= 0.0;
        let upper_is_optimal = x[index] >= upper[index] - BOUND_TOL && gradient[index] <= 0.0;
        let component = if fixed || lower_is_optimal || upper_is_optimal {
            0.0
        } else {
            gradient[index]
        };
        max_norm = max_norm.max(component.abs());
    }
    max_norm
}

fn update_history(
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

fn difference(left: &[f64], right: &[f64]) -> Vec<f64> {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left - right)
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(&left, &right)| left * right)
        .sum()
}

fn axpy(target: &mut [f64], x: &[f64], alpha: f64) {
    for (target_i, &x_i) in target.iter_mut().zip(x.iter()) {
        *target_i += alpha * x_i;
    }
}

fn norm_inf(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn norm2(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn maybe_trace(
    iteration: usize,
    value: f64,
    projected_norm: f64,
    step: &Step,
    control: BackendControl,
) {
    if control.trace > 0 && control.report > 0 && iteration.is_multiple_of(control.report) {
        eprintln!(
            "iter={iteration} f={value:.6e} ||proj_grad||_inf={projected_norm:.3e} line_search={} alpha={:.6e} max_alpha={:.6e} step_norm={:.6e} wolfe_curv={} curvature_ratio={:.3e}",
            step.line_search_trials,
            step.alpha,
            step.max_alpha,
            step.step_norm,
            step.wolfe_curvature_satisfied,
            step.curvature_ratio
        );
        if control.trace > 1 {
            eprintln!("x={:?}", step.x);
            eprintln!("gradient={:?}", step.gradient);
        }
    }
}

fn maybe_trace_cauchy_point(
    x: &[f64],
    gradient: &[f64],
    lower: &[f64],
    upper: &[f64],
    history: &[Correction],
    control: BackendControl,
) {
    if control.trace > 1 {
        let cauchy = generalized_cauchy_point_limited_memory(x, gradient, lower, upper, history);
        let step_norm = norm_inf(&difference(&cauchy.x, x));
        let subspace =
            subspace_minimizer_limited_memory(x, gradient, lower, upper, history, &cauchy);
        let (free_count, subspace_step_norm, clipped) = subspace
            .as_ref()
            .map(|point| {
                (
                    point.free_count,
                    norm_inf(&difference(&point.x, x)),
                    point.clipped,
                )
            })
            .unwrap_or((0, f64::NAN, false));
        eprintln!(
            "gcp_active={} gcp_step_inf={step_norm:.3e} subspace_free={free_count} subspace_step_inf={subspace_step_norm:.3e} subspace_clipped={clipped}",
            cauchy.active_count,
        );
    }
}

#[cfg(test)]
fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
        );
    }
}

#[cfg(test)]
fn assert_vec_not_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let max_delta = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_delta > tolerance,
        "vectors were unexpectedly close: actual={actual:?}, expected={expected:?}, tolerance={tolerance:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TraceFixture {
        fixture: String,
        source_fixture: String,
        trace_kind: String,
        first_points: Vec<Vec<f64>>,
    }

    fn assert_mirrored_steps(right: Option<f64>, left: Option<f64>) {
        let right = right.expect("right-hand step");
        let left = left.expect("left-hand mirrored step");
        assert!(
            (right + left).abs() <= 1e-14,
            "right={right:?}, left={left:?}"
        );
    }

    #[test]
    fn project_direction_keeps_free_quasi_newton_components() {
        let mut direction = vec![1.0, -2.0];
        project_direction(
            &[0.0, 0.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
            &mut direction,
        );
        assert_eq!(direction, vec![1.0, -2.0]);
    }

    #[test]
    fn project_direction_blocks_outward_bound_components() {
        let mut direction = vec![-1.0, 2.0, 1.0];
        project_direction(
            &[0.0, 1.0, 3.0],
            &[0.0, 0.0, 3.0],
            &[5.0, 1.0, 3.0],
            &mut direction,
        );
        assert_eq!(direction, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn history_update_accepts_machine_epsilon_scale_positive_curvature() {
        let mut history = Vec::new();

        update_history(
            &mut history,
            5,
            vec![1.0, 0.0],
            vec![1e-13, 1.0],
            &[-1.0, 0.0],
        );

        assert_eq!(history.len(), 1);
    }

    #[test]
    fn history_update_uses_directional_algorithm_778_skip_threshold() {
        let mut history = Vec::new();

        update_history(
            &mut history,
            5,
            vec![1.0, 0.0],
            vec![1e-18, 0.0],
            &[-1.0, 0.0],
        );

        assert!(history.is_empty());
    }

    #[test]
    fn more_thuente_stage_two_waits_for_nonnegative_directional_derivative() {
        assert!(!more_thuente_enters_stage_two(true, 0.0, 0.0, -1e-12));
        assert!(more_thuente_enters_stage_two(true, 0.0, 0.0, 0.0));
        assert!(!more_thuente_enters_stage_two(false, 0.0, 0.0, 1.0));
    }

    #[test]
    fn more_thuente_cubic_steps_are_mirror_symmetric() {
        let right_best = MoreThuentePoint {
            alpha: 0.0,
            value: 0.0,
            derivative: -1.0,
        };
        let left_best = MoreThuentePoint {
            alpha: 0.0,
            value: 0.0,
            derivative: 1.0,
        };

        assert_mirrored_steps(
            more_thuente_cubic_case_two(
                right_best,
                MoreThuentePoint {
                    alpha: 1.0,
                    value: -0.25,
                    derivative: 0.2,
                },
            ),
            more_thuente_cubic_case_two(
                left_best,
                MoreThuentePoint {
                    alpha: -1.0,
                    value: -0.25,
                    derivative: -0.2,
                },
            ),
        );

        assert_mirrored_steps(
            more_thuente_cubic_case_three(
                right_best,
                MoreThuentePoint {
                    alpha: 1.0,
                    value: -0.5,
                    derivative: -0.1,
                },
            ),
            more_thuente_cubic_case_three(
                left_best,
                MoreThuentePoint {
                    alpha: -1.0,
                    value: -0.5,
                    derivative: 0.1,
                },
            ),
        );

        assert_mirrored_steps(
            more_thuente_cubic_case_four(
                MoreThuentePoint {
                    alpha: 2.0,
                    value: 0.5,
                    derivative: 0.7,
                },
                MoreThuentePoint {
                    alpha: 1.0,
                    value: -0.25,
                    derivative: -1.2,
                },
            ),
            more_thuente_cubic_case_four(
                MoreThuentePoint {
                    alpha: -2.0,
                    value: 0.5,
                    derivative: -0.7,
                },
                MoreThuentePoint {
                    alpha: -1.0,
                    value: -0.25,
                    derivative: 1.2,
                },
            ),
        );
    }

    #[test]
    fn identity_cauchy_point_is_unconstrained_newton_step() {
        let cauchy = generalized_cauchy_point_identity(
            &[1.0, -2.0],
            &[0.25, -0.5],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
        );
        assert_eq!(cauchy.active_count, 0);
        assert_vec_close(&cauchy.x, &[0.75, -1.5], 1e-15);
    }

    #[test]
    fn identity_cauchy_point_stops_at_box_when_stationary_point_is_beyond_bounds() {
        let cauchy = generalized_cauchy_point_identity(
            &[-1.2, 1.0],
            &[-215.6, -88.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert_eq!(cauchy.active_count, 2);
        assert_vec_close(&cauchy.x, &[5.0, 5.0], 1e-15);
    }

    #[test]
    fn identity_cauchy_point_handles_partial_activation() {
        let cauchy =
            generalized_cauchy_point_identity(&[0.0, 0.0], &[1.0, 1.0], &[-0.9, -5.0], &[5.0, 5.0]);
        assert_eq!(cauchy.active_count, 1);
        assert_vec_close(&cauchy.x, &[-0.9, -1.0], 1e-15);
    }

    #[test]
    fn direct_hessian_product_inverts_two_loop_inverse_product() {
        let history = vec![
            Correction {
                s: vec![1.0, 0.0],
                y: vec![2.0, 0.0],
                rho: 0.5,
            },
            Correction {
                s: vec![0.0, 1.0],
                y: vec![0.0, 3.0],
                rho: 1.0 / 3.0,
            },
        ];

        let vector = vec![4.0, 9.0];
        let hessian_vector = lbfgs_hessian_product(&vector, &history);
        assert_vec_close(&hessian_vector, &[8.0, 27.0], 1e-14);

        let inverse_hessian_vector = lbfgs_inverse_product(&hessian_vector, &history);
        assert_vec_close(&inverse_hessian_vector, &vector, 1e-14);
    }

    #[test]
    fn direct_hessian_product_satisfies_latest_secant_condition() {
        let history = vec![
            Correction {
                s: vec![1.0, 0.0],
                y: vec![2.0, 1.0],
                rho: 0.5,
            },
            Correction {
                s: vec![0.0, 1.0],
                y: vec![1.0, 3.0],
                rho: 1.0 / 3.0,
            },
        ];

        let latest = history.last().expect("history fixture");
        let hessian_s = lbfgs_hessian_product(&latest.s, &history);
        assert_vec_close(&hessian_s, &latest.y, 1e-14);
    }

    #[test]
    fn direct_hessian_product_inverts_two_loop_for_nonorthogonal_history() {
        let history = vec![
            Correction {
                s: vec![1.0, 0.0],
                y: vec![2.0, 0.5],
                rho: 0.5,
            },
            Correction {
                s: vec![0.25, 1.0],
                y: vec![1.0, 3.0],
                rho: 1.0 / 3.25,
            },
        ];

        for vector in [vec![4.0, 9.0], vec![-1.25, 0.75], vec![0.0, 2.5]] {
            let inverse_hessian_vector = lbfgs_inverse_product(&vector, &history);
            let hessian_inverse_hessian_vector =
                lbfgs_hessian_product(&inverse_hessian_vector, &history);
            assert_vec_close(&hessian_inverse_hessian_vector, &vector, 1e-12);
        }
    }

    #[test]
    fn limited_memory_cauchy_point_uses_model_curvature() {
        let history = vec![Correction {
            s: vec![1.0, 0.0],
            y: vec![2.0, 0.0],
            rho: 0.5,
        }];

        let cauchy = generalized_cauchy_point_limited_memory(
            &[0.0, 0.0],
            &[2.0, 0.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
            &history,
        );

        assert_eq!(cauchy.active_count, 0);
        assert_vec_close(&cauchy.x, &[-1.0, 0.0], 1e-14);
    }

    #[test]
    fn unbounded_cauchy_subspace_step_matches_two_loop_direction() {
        let history = vec![
            Correction {
                s: vec![1.0, 0.0],
                y: vec![2.0, 0.5],
                rho: 0.5,
            },
            Correction {
                s: vec![0.25, 1.0],
                y: vec![1.0, 3.0],
                rho: 1.0 / 3.25,
            },
        ];
        let x = vec![0.2, -0.4];
        let gradient = vec![3.0, -2.0];
        let lower = vec![f64::NEG_INFINITY; 2];
        let upper = vec![f64::INFINITY; 2];

        let cauchy =
            generalized_cauchy_point_limited_memory(&x, &gradient, &lower, &upper, &history);
        let subspace =
            subspace_minimizer_limited_memory(&x, &gradient, &lower, &upper, &history, &cauchy)
                .expect("positive definite full-space model");
        let model_direction = difference(&subspace.x, &x);
        let two_loop_direction = lbfgs_direction(&gradient, &history);

        assert_eq!(cauchy.active_count, 0);
        assert_eq!(subspace.free_count, 2);
        assert!(!subspace.clipped);
        assert_vec_close(&model_direction, &two_loop_direction, 1e-12);
    }

    #[test]
    fn subspace_minimizer_solves_free_block_after_cauchy_point() {
        let cauchy = CauchyPoint {
            x: vec![-0.5, 1.0],
            active_count: 1,
        };

        let subspace = subspace_minimizer_with_hessian(
            &[0.0, 0.0],
            &[5.0, -4.0],
            &[-0.5, -10.0],
            &[10.0, 10.0],
            &cauchy,
            dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
        )
        .expect("positive definite subspace");

        assert_eq!(subspace.free_count, 1);
        assert!(!subspace.clipped);
        assert_vec_close(&subspace.x, &[-0.5, 2.25], 1e-14);
    }

    #[test]
    fn subspace_minimizer_clips_to_remaining_bounds() {
        let cauchy = CauchyPoint {
            x: vec![-0.5, 1.0],
            active_count: 1,
        };

        let subspace = subspace_minimizer_with_hessian(
            &[0.0, 0.0],
            &[5.0, -4.0],
            &[-0.5, -10.0],
            &[10.0, 2.0],
            &cauchy,
            dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
        )
        .expect("positive definite subspace");

        assert_eq!(subspace.free_count, 1);
        assert!(subspace.clipped);
        assert_vec_close(&subspace.x, &[-0.5, 2.0], 1e-14);
    }

    #[test]
    fn subspace_minimizer_returns_cauchy_point_when_all_variables_are_active() {
        let cauchy = CauchyPoint {
            x: vec![-0.5, 2.0],
            active_count: 2,
        };

        let subspace = subspace_minimizer_with_hessian(
            &[0.0, 0.0],
            &[5.0, -4.0],
            &[-0.5, -10.0],
            &[10.0, 2.0],
            &cauchy,
            dense_two_by_two_product([[4.0, 1.0], [1.0, 2.0]]),
        )
        .expect("no free variables still yields a subspace point");

        assert_eq!(subspace.free_count, 0);
        assert!(!subspace.clipped);
        assert_vec_close(&subspace.x, &cauchy.x, 1e-14);
    }

    #[test]
    fn line_search_reports_zero_trials_for_accepted_full_step() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[1.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[f64::INFINITY],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("line search succeeds")
        .expect("descent step");

        assert_eq!(step.line_search_trials, 0);
        assert_eq!(counts.function, 1);
        assert_eq!(counts.gradient, 1);
        assert_vec_close(&step.x, &[1.0], 1e-15);
        assert!((step.alpha - 1.0).abs() <= 1e-15);
        assert!((step.max_alpha - 1.0).abs() <= 1e-15);
        assert!((step.step_norm - 1.0).abs() <= 1e-15);
        assert!((step.curvature_ratio - 0.5).abs() <= 1e-15);
        assert!(step.wolfe_curvature_satisfied);
    }

    #[test]
    fn line_search_reports_interpolated_trial_after_rejected_full_step() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[10.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[f64::INFINITY],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: true,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("line search succeeds")
        .expect("descent step");

        assert_eq!(step.line_search_trials, 1);
        assert_eq!(counts.function, 2);
        assert_eq!(counts.gradient, 2);
        assert!((step.alpha - 0.1998).abs() <= 1e-15);
        assert!(step.value < 1e-4, "{step:?}");
        assert!((step.step_norm - step.x[0].abs()).abs() <= 1e-15);
        assert!((step.curvature_ratio - 1e-3).abs() <= 1e-15, "{step:?}");
        assert!(step.wolfe_curvature_satisfied);
    }

    #[test]
    fn line_search_diagnoses_armijo_step_that_is_too_short_for_wolfe_curvature() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[0.1],
                lower: &[f64::NEG_INFINITY],
                upper: &[f64::INFINITY],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("line search succeeds")
        .expect("descent step");

        assert_eq!(step.line_search_trials, 0);
        assert_eq!(counts.function, 1);
        assert_eq!(counts.gradient, 1);
        assert_vec_close(&step.x, &[0.1], 1e-15);
        assert!((step.alpha - 1.0).abs() <= 1e-15);
        assert!((step.curvature_ratio - 0.95).abs() <= 1e-15, "{step:?}");
        assert!(!step.wolfe_curvature_satisfied);
    }

    #[test]
    fn more_thuente_accepts_r_max_step_warning_case() {
        let mut problem = LinearDescent;
        let mut counts = OptimCounts::default();
        let step = more_thuente_line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 0.0,
                gradient: &[-1.0],
                direction: &[1.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[10.0],
                max_step_cap: Some(1.0),
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MAIN_PATH_MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("More-Thuente should not error")
        .expect("R accepts STPMAX warning as a new iterate");

        assert_eq!(counts.function, 1);
        assert_eq!(counts.gradient, 1);
        assert_eq!(step.alpha, 1.0);
        assert_eq!(step.max_alpha, 1.0);
        assert!(!step.wolfe_curvature_satisfied);
        assert_vec_close(&step.x, &[1.0], 1e-15);
    }

    #[test]
    fn strong_wolfe_line_search_expands_beyond_unit_alpha_when_direction_is_short() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = strong_wolfe_line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[0.1],
                lower: &[f64::NEG_INFINITY],
                upper: &[1.0],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("strong-Wolfe line search should not error")
        .expect("strong-Wolfe line search should expand to satisfy curvature");

        assert_eq!(step.line_search_trials, 1);
        assert_eq!(counts.function, 2);
        assert_eq!(counts.gradient, 2);
        assert!((step.alpha - 2.0).abs() <= 1e-15, "{step:?}");
        assert!((step.max_alpha - 10.0).abs() <= 1e-15, "{step:?}");
        assert_vec_close(&step.x, &[0.2], 1e-15);
        assert!(step.wolfe_curvature_satisfied);
    }

    #[test]
    fn strong_wolfe_line_search_zooms_after_overlarge_unit_step() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = strong_wolfe_line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[10.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[10.0],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("strong-Wolfe line search should not error")
        .expect("strong-Wolfe line search should zoom into the acceptable interval");

        assert_eq!(step.line_search_trials, 2);
        assert_eq!(counts.function, 3);
        assert_eq!(counts.gradient, 3);
        assert!((step.alpha - 0.25).abs() <= 1e-15, "{step:?}");
        assert_vec_close(&step.x, &[2.5], 1e-15);
        assert!(step.wolfe_curvature_satisfied);
    }

    #[test]
    fn strong_wolfe_line_search_accepts_armijo_step_at_feasible_cap() {
        let mut problem = OneDimensionalQuadratic;
        let mut counts = OptimCounts::default();
        let step = strong_wolfe_line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[1.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[0.1],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("strong-Wolfe line search should not error")
        .expect("bounded line search should accept sufficient decrease at the cap");

        assert_eq!(step.line_search_trials, 0);
        assert_eq!(counts.function, 1);
        assert_eq!(counts.gradient, 1);
        assert!((step.alpha - 0.1).abs() <= 1e-15, "{step:?}");
        assert!((step.max_alpha - 0.1).abs() <= 1e-15, "{step:?}");
        assert_vec_close(&step.x, &[0.1], 1e-15);
        assert!((step.curvature_ratio - 0.95).abs() <= 1e-15, "{step:?}");
        assert!(!step.wolfe_curvature_satisfied);
    }

    #[test]
    fn strong_wolfe_line_search_propagates_evaluation_errors() {
        let mut problem = AlwaysNonFiniteObjective;
        let mut counts = OptimCounts::default();
        let error = strong_wolfe_line_search(
            &mut problem,
            LineSearchRequest {
                x: &[0.0],
                value: 4.0,
                gradient: &[-4.0],
                direction: &[1.0],
                lower: &[f64::NEG_INFINITY],
                upper: &[f64::INFINITY],
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect_err("line search should return the objective error");

        assert!(matches!(
            error,
            OptimError::NonFiniteObjective { value } if value.is_nan()
        ));
        assert_eq!(counts.function, 0);
        assert_eq!(counts.gradient, 0);
    }

    #[test]
    fn strong_wolfe_mode_records_rosenbrock_probe_drift() {
        let armijo = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::ProjectedLbfgs,
            5,
            1e7,
        );
        let strong_wolfe = run_rosenbrock_backend(
            LineSearchMode::StrongWolfe,
            DirectionMode::ProjectedLbfgs,
            5,
            1e7,
        );
        let strong_wolfe_loose = run_rosenbrock_backend(
            LineSearchMode::StrongWolfe,
            DirectionMode::ProjectedLbfgs,
            5,
            1e12,
        );
        let strong_wolfe_lmm_one = run_rosenbrock_backend(
            LineSearchMode::StrongWolfe,
            DirectionMode::ProjectedLbfgs,
            1,
            1e7,
        );

        assert_eq!(armijo.convergence, 0, "{armijo:?}");
        assert_eq!(armijo.counts.function, 48, "{armijo:?}");
        assert_eq!(armijo.counts.gradient, 48, "{armijo:?}");

        assert_eq!(strong_wolfe.convergence, 0, "{strong_wolfe:?}");
        assert_eq!(strong_wolfe.counts.function, 55, "{strong_wolfe:?}");
        assert_eq!(strong_wolfe.counts.gradient, 55, "{strong_wolfe:?}");

        assert_eq!(strong_wolfe_loose.convergence, 0, "{strong_wolfe_loose:?}");
        assert_eq!(
            strong_wolfe_loose.counts.function, 50,
            "{strong_wolfe_loose:?}"
        );
        assert_eq!(
            strong_wolfe_loose.counts.gradient, 50,
            "{strong_wolfe_loose:?}"
        );

        assert_eq!(
            strong_wolfe_lmm_one.convergence, 0,
            "{strong_wolfe_lmm_one:?}"
        );
        assert_eq!(
            strong_wolfe_lmm_one.counts.function, 92,
            "{strong_wolfe_lmm_one:?}"
        );
        assert_eq!(
            strong_wolfe_lmm_one.counts.gradient, 92,
            "{strong_wolfe_lmm_one:?}"
        );
    }

    #[test]
    fn cauchy_subspace_direction_mode_records_rosenbrock_probe_drift() {
        let cauchy_armijo = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchySubspace,
            5,
            1e7,
        );
        let cauchy_capped_first_step = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchySubspaceCappedFirstStep,
            5,
            1e7,
        );
        let cauchy_first_then_projected = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchyFirstThenProjectedCapped,
            5,
            1e7,
        );
        let cauchy_strong_wolfe = run_rosenbrock_backend(
            LineSearchMode::StrongWolfe,
            DirectionMode::CauchySubspace,
            5,
            1e7,
        );
        let cauchy_more_thuente = run_rosenbrock_backend(
            LineSearchMode::MoreThuente,
            DirectionMode::CauchySubspace,
            5,
            1e7,
        );
        let cauchy_loose = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchySubspace,
            5,
            1e12,
        );
        let cauchy_lmm_one = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchySubspace,
            1,
            1e7,
        );
        let cauchy_more_thuente_loose = run_rosenbrock_backend(
            LineSearchMode::MoreThuente,
            DirectionMode::CauchySubspace,
            5,
            1e12,
        );
        let cauchy_more_thuente_lmm_one = run_rosenbrock_backend(
            LineSearchMode::MoreThuente,
            DirectionMode::CauchySubspace,
            1,
            1e7,
        );
        let cauchy_first_then_projected_loose = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchyFirstThenProjectedCapped,
            5,
            1e12,
        );
        let cauchy_first_then_projected_lmm_one = run_rosenbrock_backend(
            LineSearchMode::BacktrackingArmijo,
            DirectionMode::CauchyFirstThenProjectedCapped,
            1,
            1e7,
        );
        assert_eq!(cauchy_armijo.convergence, 0, "{cauchy_armijo:?}");
        assert_eq!(cauchy_armijo.counts.function, 47, "{cauchy_armijo:?}");
        assert_eq!(
            cauchy_armijo.counts.function, cauchy_armijo.counts.gradient,
            "{cauchy_armijo:?}"
        );
        assert_eq!(
            cauchy_capped_first_step.convergence, 0,
            "{cauchy_capped_first_step:?}"
        );
        assert_eq!(
            cauchy_capped_first_step.counts.function, 53,
            "{cauchy_capped_first_step:?}"
        );
        assert_eq!(
            cauchy_capped_first_step.counts.function, cauchy_capped_first_step.counts.gradient,
            "{cauchy_capped_first_step:?}"
        );
        assert_eq!(
            cauchy_first_then_projected.convergence, 0,
            "{cauchy_first_then_projected:?}"
        );
        assert_eq!(
            cauchy_first_then_projected.counts.function, 53,
            "{cauchy_first_then_projected:?}"
        );
        assert_eq!(
            cauchy_first_then_projected.counts.function,
            cauchy_first_then_projected.counts.gradient,
            "{cauchy_first_then_projected:?}"
        );
        assert_eq!(
            cauchy_strong_wolfe.convergence, 0,
            "{cauchy_strong_wolfe:?}"
        );
        assert_eq!(
            cauchy_strong_wolfe.counts.function, 56,
            "{cauchy_strong_wolfe:?}"
        );
        assert_eq!(
            cauchy_strong_wolfe.counts.function, cauchy_strong_wolfe.counts.gradient,
            "{cauchy_strong_wolfe:?}"
        );
        assert_eq!(
            cauchy_more_thuente.convergence, 0,
            "{cauchy_more_thuente:?}"
        );
        assert_eq!(
            cauchy_more_thuente.counts.function, 48,
            "{cauchy_more_thuente:?}"
        );
        assert_eq!(
            cauchy_more_thuente.counts.function, cauchy_more_thuente.counts.gradient,
            "{cauchy_more_thuente:?}"
        );
        assert_eq!(cauchy_loose.convergence, 0, "{cauchy_loose:?}");
        assert_eq!(cauchy_loose.counts.function, 43, "{cauchy_loose:?}");
        assert_eq!(
            cauchy_loose.counts.function, cauchy_loose.counts.gradient,
            "{cauchy_loose:?}"
        );
        assert_eq!(cauchy_lmm_one.convergence, 0, "{cauchy_lmm_one:?}");
        assert_eq!(cauchy_lmm_one.counts.function, 81, "{cauchy_lmm_one:?}");
        assert_eq!(
            cauchy_lmm_one.counts.function, cauchy_lmm_one.counts.gradient,
            "{cauchy_lmm_one:?}"
        );
        assert_eq!(
            cauchy_more_thuente_loose.convergence, 0,
            "{cauchy_more_thuente_loose:?}"
        );
        assert_eq!(
            cauchy_more_thuente_loose.counts.function, 45,
            "{cauchy_more_thuente_loose:?}"
        );
        assert_eq!(
            cauchy_more_thuente_loose.counts.function, cauchy_more_thuente_loose.counts.gradient,
            "{cauchy_more_thuente_loose:?}"
        );
        assert_eq!(
            cauchy_more_thuente_lmm_one.convergence, 0,
            "{cauchy_more_thuente_lmm_one:?}"
        );
        assert_eq!(
            cauchy_more_thuente_lmm_one.counts.function, 69,
            "{cauchy_more_thuente_lmm_one:?}"
        );
        assert_eq!(
            cauchy_more_thuente_lmm_one.counts.function,
            cauchy_more_thuente_lmm_one.counts.gradient,
            "{cauchy_more_thuente_lmm_one:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_loose.convergence, 0,
            "{cauchy_first_then_projected_loose:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_loose.counts.function, 48,
            "{cauchy_first_then_projected_loose:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_loose.counts.function,
            cauchy_first_then_projected_loose.counts.gradient,
            "{cauchy_first_then_projected_loose:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_lmm_one.convergence, 0,
            "{cauchy_first_then_projected_lmm_one:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_lmm_one.counts.function, 69,
            "{cauchy_first_then_projected_lmm_one:?}"
        );
        assert_eq!(
            cauchy_first_then_projected_lmm_one.counts.function,
            cauchy_first_then_projected_lmm_one.counts.gradient,
            "{cauchy_first_then_projected_lmm_one:?}"
        );
    }

    #[test]
    fn more_thuente_first_step_then_projected_records_rosenbrock_probe_drift() {
        let r_shaped_first = run_rosenbrock_backend(
            LineSearchMode::MoreThuenteFirstThenBacktracking,
            DirectionMode::CauchyFirstThenProjected,
            5,
            1e7,
        );
        let r_shaped_first_loose = run_rosenbrock_backend(
            LineSearchMode::MoreThuenteFirstThenBacktracking,
            DirectionMode::CauchyFirstThenProjected,
            5,
            1e12,
        );
        let r_shaped_first_lmm_one = run_rosenbrock_backend(
            LineSearchMode::MoreThuenteFirstThenBacktracking,
            DirectionMode::CauchyFirstThenProjected,
            1,
            1e7,
        );

        assert_eq!(r_shaped_first.convergence, 0, "{r_shaped_first:?}");
        assert_eq!(r_shaped_first.counts.function, 46, "{r_shaped_first:?}");
        assert_eq!(
            r_shaped_first.counts.function, r_shaped_first.counts.gradient,
            "{r_shaped_first:?}"
        );
        assert_eq!(
            r_shaped_first_loose.convergence, 0,
            "{r_shaped_first_loose:?}"
        );
        assert_eq!(
            r_shaped_first_loose.counts.function, 43,
            "{r_shaped_first_loose:?}"
        );
        assert_eq!(
            r_shaped_first_loose.counts.function, r_shaped_first_loose.counts.gradient,
            "{r_shaped_first_loose:?}"
        );
        assert_eq!(
            r_shaped_first_lmm_one.convergence, 0,
            "{r_shaped_first_lmm_one:?}"
        );
        assert_eq!(
            r_shaped_first_lmm_one.counts.function, 62,
            "{r_shaped_first_lmm_one:?}"
        );
        assert_eq!(
            r_shaped_first_lmm_one.counts.function, r_shaped_first_lmm_one.counts.gradient,
            "{r_shaped_first_lmm_one:?}"
        );
    }

    #[test]
    fn more_thuente_first_step_full_probe_matches_r_trace_prefix() {
        let fixture: TraceFixture = serde_json::from_str(include_str!(
            "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
        ))
        .expect("R trace fixture JSON");
        let mut backend = NativeBackend;
        let mut problem = RecordingRosenbrockProblem::default();
        let result = backend
            .minimize_with_modes(
                &mut problem,
                &[-1.2, 1.0],
                &[-5.0, -5.0],
                &[5.0, 5.0],
                BackendControl {
                    maxit: 100,
                    factr: 1e7,
                    pgtol: 0.0,
                    lmm: 5,
                    trace: 0,
                    report: 10,
                    has_user_gradient: true,
                },
                BackendModes {
                    line_search: LineSearchMode::MoreThuenteFirstThenBacktracking,
                    direction: DirectionMode::CauchyFirstThenProjected,
                },
            )
            .expect("Rosenbrock backend optimization");

        assert_eq!(result.convergence, 0, "{result:?}");
        assert_eq!(result.counts.function, 46, "{result:?}");
        assert_eq!(result.counts.function, result.counts.gradient, "{result:?}");
        assert!(
            problem.points.len() >= 8,
            "expected at least the first accepted R-shaped step: {:?}",
            problem.points
        );

        for (actual, expected) in problem
            .points
            .iter()
            .zip(fixture.first_points.iter())
            .take(7)
        {
            assert_vec_close(actual, expected, 1e-12);
        }
        assert_vec_close(
            &problem.points[7],
            &[-0.020278688163727798, -0.9287651476463392],
            1e-12,
        );
        assert_vec_not_close(&problem.points[7], &fixture.first_points[7], 1e-3);
    }

    #[test]
    fn cauchy_more_thuente_full_probe_matches_r_trace_prefix_and_counts() {
        let fixture: TraceFixture = serde_json::from_str(include_str!(
            "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
        ))
        .expect("R trace fixture JSON");
        let mut backend = NativeBackend;
        let mut problem = RecordingRosenbrockProblem::default();
        let result = backend
            .minimize_with_modes(
                &mut problem,
                &[-1.2, 1.0],
                &[-5.0, -5.0],
                &[5.0, 5.0],
                BackendControl {
                    maxit: 100,
                    factr: 1e7,
                    pgtol: 0.0,
                    lmm: 5,
                    trace: 0,
                    report: 10,
                    has_user_gradient: true,
                },
                BackendModes {
                    line_search: LineSearchMode::MoreThuente,
                    direction: DirectionMode::CauchySubspace,
                },
            )
            .expect("Rosenbrock backend optimization");

        assert_eq!(result.convergence, 0, "{result:?}");
        assert_eq!(result.counts.function, 48, "{result:?}");
        assert_eq!(result.counts.function, result.counts.gradient, "{result:?}");
        assert!(
            problem.points.len() >= 8,
            "expected at least the first third-iteration trial: {:?}",
            problem.points
        );
        for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
            assert_vec_close(actual, expected, 1e-10);
        }
    }

    #[test]
    fn fourth_iteration_more_thuente_extrapolates_to_r_trace_step() {
        let fixture: TraceFixture = serde_json::from_str(include_str!(
            "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
        ))
        .expect("R trace fixture JSON");
        let lower = vec![-5.0, -5.0];
        let upper = vec![5.0, 5.0];
        let mut problem = RosenbrockProblem;
        let x0 = fixture.first_points[0].clone();
        let x1 = fixture.first_points[3].clone();
        let x2 = fixture.first_points[4].clone();
        let x3 = fixture.first_points[5].clone();
        let g0 = problem.gradient(&x0).expect("Rosenbrock gradient");
        let g1 = problem.gradient(&x1).expect("Rosenbrock gradient");
        let g2 = problem.gradient(&x2).expect("Rosenbrock gradient");
        let g3 = problem.gradient(&x3).expect("Rosenbrock gradient");
        let mut history = Vec::new();

        update_history(
            &mut history,
            5,
            difference(&x1, &x0),
            difference(&g1, &g0),
            &g0,
        );
        update_history(
            &mut history,
            5,
            difference(&x2, &x1),
            difference(&g2, &g1),
            &g1,
        );
        update_history(
            &mut history,
            5,
            difference(&x3, &x2),
            difference(&g3, &g2),
            &g2,
        );

        let direction =
            cauchy_subspace_direction(&x3, &g3, &lower, &upper, &history, MAIN_PATH_MIN_STEP)
                .expect("fourth iteration model should produce a step");
        let value = problem.value(&x3).expect("Rosenbrock value");
        let mut recording = RecordingRosenbrockProblem::default();
        let mut counts = OptimCounts::default();
        let step = line_search_with_mode(
            &mut recording,
            LineSearchRequest {
                x: &x3,
                value,
                gradient: &g3,
                direction: &direction,
                lower: &lower,
                upper: &upper,
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MAIN_PATH_MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
            LineSearchMode::MoreThuente,
            4,
        )
        .expect("line search should not error")
        .expect("line search should accept the R trace step");
        assert_eq!(history.len(), 3);
        assert_eq!(counts.function, 5);
        assert_eq!(counts.gradient, 5);
        assert_eq!(step.line_search_trials, 4);
        for (actual, expected) in recording
            .points
            .iter()
            .zip(fixture.first_points.iter().skip(6))
        {
            assert_vec_close(actual, expected, 1e-10);
        }
        assert_vec_close(&step.x, &fixture.first_points[10], 1e-10);
    }

    #[test]
    fn capped_cauchy_subspace_first_step_matches_r_trace_shape() {
        let x = vec![-1.2, 1.0];
        let lower = vec![-5.0, -5.0];
        let upper = vec![5.0, 5.0];
        let history = Vec::new();
        let modes = BackendModes {
            line_search: LineSearchMode::BacktrackingArmijo,
            direction: DirectionMode::CauchySubspaceCappedFirstStep,
        };
        let mut problem = RosenbrockProblem;
        let value = problem.value(&x).expect("Rosenbrock value");
        let gradient = problem.gradient(&x).expect("Rosenbrock gradient");
        let direction = direction_with_mode(
            &x,
            &gradient,
            &lower,
            &upper,
            &history,
            modes.direction,
            MIN_STEP,
        );
        let mut counts = OptimCounts::default();

        let step = line_search(
            &mut problem,
            LineSearchRequest {
                x: &x,
                value,
                gradient: &gradient,
                direction: &direction,
                lower: &lower,
                upper: &upper,
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: initial_step_cap_for_modes(modes, history.is_empty()),
                allow_quadratic_interpolation: true,
                min_step: MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
        )
        .expect("line search should not error")
        .expect("capped Cauchy step should be accepted");

        assert_eq!(step.line_search_trials, 1);
        assert_eq!(counts.function, 2);
        assert_eq!(counts.gradient, 2);
        assert!(
            (step.alpha - 0.032438841511832885).abs() <= 1e-14,
            "{step:?}"
        );
        assert!((step.x[0] + 0.9988791826266361).abs() <= 1e-14, "{step:?}");
        assert!((step.x[1] - 1.1297553660473316).abs() <= 1e-14, "{step:?}");
        assert!(
            (step.step_norm - 0.2393450191648178).abs() <= 1e-14,
            "{step:?}"
        );
    }

    #[test]
    fn more_thuente_cauchy_first_step_matches_r_trace_prefix() {
        let fixture: TraceFixture = serde_json::from_str(include_str!(
            "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
        ))
        .expect("R trace fixture JSON");
        let x = vec![-1.2, 1.0];
        let lower = vec![-5.0, -5.0];
        let upper = vec![5.0, 5.0];
        let history = Vec::new();
        let mut problem = RecordingRosenbrockProblem::default();
        let mut counts = OptimCounts::default();
        let Evaluation { value, gradient } =
            evaluate(&mut problem, &x, &mut counts).expect("initial Rosenbrock evaluation");
        let direction = direction_with_mode(
            &x,
            &gradient,
            &lower,
            &upper,
            &history,
            DirectionMode::CauchySubspace,
            MAIN_PATH_MIN_STEP,
        );

        let step = line_search_with_mode(
            &mut problem,
            LineSearchRequest {
                x: &x,
                value,
                gradient: &gradient,
                direction: &direction,
                lower: &lower,
                upper: &upper,
                max_step_cap: None,
                cap_initial_unbounded_step: false,
                initial_step_cap: None,
                allow_quadratic_interpolation: false,
                min_step: MAIN_PATH_MIN_STEP,
                quadratic_interpolation_damping: INTERPOLATION_DAMPING,
            },
            &mut counts,
            LineSearchMode::MoreThuente,
            1,
        )
        .expect("More-Thuente probe should not error")
        .expect("More-Thuente probe should accept the first R-like step");

        assert_eq!(step.line_search_trials, 2);
        assert_eq!(counts.function, 4);
        assert_eq!(counts.gradient, 4);
        assert!((step.alpha - 0.03238226005834323).abs() <= 1e-14);
        assert_vec_close(&step.x, &fixture.first_points[3], 1e-14);
        for (actual, expected) in problem
            .points
            .iter()
            .zip(fixture.first_points.iter())
            .take(4)
        {
            assert_vec_close(actual, expected, 1e-12);
        }
    }

    #[test]
    fn rosenbrock_default_evaluation_trace_matches_r_prefix() {
        let fixture: TraceFixture = serde_json::from_str(include_str!(
            "../../fixtures/r_optim_lbfgsb/rosenbrock_default_trace.json"
        ))
        .expect("R trace fixture JSON");
        assert_eq!(fixture.fixture, "rosenbrock_default_trace");
        assert_eq!(fixture.source_fixture, "rosenbrock");
        assert_eq!(fixture.trace_kind, "function_evaluation_points");

        let mut backend = NativeBackend;
        let mut problem = RecordingRosenbrockProblem::default();
        let result = backend
            .minimize_with_modes(
                &mut problem,
                &[-1.2, 1.0],
                &[-5.0, -5.0],
                &[5.0, 5.0],
                BackendControl {
                    maxit: 100,
                    factr: 1e7,
                    pgtol: 0.0,
                    lmm: 5,
                    trace: 0,
                    report: 10,
                    has_user_gradient: true,
                },
                BackendModes::default(),
            )
            .expect("Rosenbrock backend optimization");

        assert_eq!(result.counts.function, 48, "{result:?}");
        assert_eq!(result.counts.gradient, 48, "{result:?}");
        assert!(
            problem.points.len() >= fixture.first_points.len(),
            "{:?}",
            problem.points
        );

        for (actual, expected) in problem.points.iter().zip(fixture.first_points.iter()) {
            assert_vec_close(actual, expected, 1e-10);
        }
    }

    #[test]
    fn strong_wolfe_scaffold_expands_past_too_short_armijo_step() {
        let result =
            strong_wolfe_search(0.1, 4.0, 4.0, -4.0, |alpha| Ok(one_dimensional_phi(alpha)))
                .expect("strong Wolfe search should not error")
                .expect("strong Wolfe search should expand to a curvature-satisfying step");

        assert!(result.point.alpha > 0.1, "{result:?}");
        assert!(result.point.value <= armijo_value(4.0, -4.0, result.point.alpha));
        assert!(result.point.derivative.abs() <= -WOLFE_CURVATURE * -4.0);
        assert_eq!(result.point.alpha, 0.2);
    }

    #[test]
    fn strong_wolfe_scaffold_zooms_from_overlarge_step() {
        let result = strong_wolfe_search(10.0, 10.0, 4.0, -4.0, |alpha| {
            Ok(one_dimensional_phi(alpha))
        })
        .expect("strong Wolfe search should not error")
        .expect("strong Wolfe search should zoom into the acceptable interval");

        assert!(result.point.alpha > 0.0, "{result:?}");
        assert!(result.point.alpha < 10.0, "{result:?}");
        assert!(result.point.value <= armijo_value(4.0, -4.0, result.point.alpha));
        assert!(result.point.derivative.abs() <= -WOLFE_CURVATURE * -4.0);
        assert_eq!(result.point.alpha, 2.5);
    }

    fn dense_two_by_two_product(matrix: [[f64; 2]; 2]) -> impl FnMut(&[f64]) -> Vec<f64> {
        move |vector| {
            vec![
                matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
                matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
            ]
        }
    }

    fn one_dimensional_phi(alpha: f64) -> LineSearchPoint1D {
        LineSearchPoint1D {
            alpha,
            value: (alpha - 2.0).powi(2),
            derivative: 2.0 * (alpha - 2.0),
        }
    }

    fn run_rosenbrock_backend(
        line_search_mode: LineSearchMode,
        direction_mode: DirectionMode,
        lmm: usize,
        factr: f64,
    ) -> BackendResult {
        let mut backend = NativeBackend;
        let mut problem = RosenbrockProblem;
        backend
            .minimize_with_modes(
                &mut problem,
                &[-1.2, 1.0],
                &[-5.0, -5.0],
                &[5.0, 5.0],
                BackendControl {
                    maxit: 100,
                    factr,
                    pgtol: 0.0,
                    lmm,
                    trace: 0,
                    report: 10,
                    has_user_gradient: true,
                },
                BackendModes {
                    line_search: line_search_mode,
                    direction: direction_mode,
                },
            )
            .expect("Rosenbrock backend optimization")
    }

    struct RosenbrockProblem;

    impl BackendProblem for RosenbrockProblem {
        fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
            Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
        }

        fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
            Ok(vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ])
        }
    }

    #[derive(Default)]
    struct RecordingRosenbrockProblem {
        points: Vec<Vec<f64>>,
    }

    impl BackendProblem for RecordingRosenbrockProblem {
        fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
            Ok(100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2))
        }

        fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
            Ok(vec![
                -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0] * x[0]),
            ])
        }

        fn value_and_gradient(&mut self, x: &[f64]) -> Result<(f64, Vec<f64>), OptimError> {
            self.points.push(x.to_vec());
            Ok((self.value(x)?, self.gradient(x)?))
        }
    }

    struct OneDimensionalQuadratic;

    impl BackendProblem for OneDimensionalQuadratic {
        fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
            Ok((x[0] - 2.0).powi(2))
        }

        fn gradient(&mut self, x: &[f64]) -> Result<Vec<f64>, OptimError> {
            Ok(vec![2.0 * (x[0] - 2.0)])
        }
    }

    struct LinearDescent;

    impl BackendProblem for LinearDescent {
        fn value(&mut self, x: &[f64]) -> Result<f64, OptimError> {
            Ok(-x[0])
        }

        fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
            Ok(vec![-1.0])
        }
    }

    struct AlwaysNonFiniteObjective;

    impl BackendProblem for AlwaysNonFiniteObjective {
        fn value(&mut self, _x: &[f64]) -> Result<f64, OptimError> {
            Err(OptimError::NonFiniteObjective { value: f64::NAN })
        }

        fn gradient(&mut self, _x: &[f64]) -> Result<Vec<f64>, OptimError> {
            Ok(vec![0.0])
        }
    }
}
