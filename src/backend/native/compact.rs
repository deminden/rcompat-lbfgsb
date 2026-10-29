use super::*;

#[cfg(test)]
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
#[cfg(test)]
struct CompactMatrices {
    theta: f64,
    sy: Vec<Vec<f64>>,
    wt: Vec<Vec<f64>>,
    history: Vec<Correction>,
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn index_mask(dimension: usize, indices: &[usize]) -> Vec<bool> {
    let mut mask = vec![false; dimension];
    for &index in indices {
        mask[index] = true;
    }
    mask
}

#[cfg(test)]
fn dot_on_indices(left: &[f64], right: &[f64], indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&index| left[index] * right[index])
        .sum()
}

/// Stateful compact L-BFGS-B model for all production bound and gradient routes.
///
/// Unlike `CompactMatrices`, this retains the cyclic S/Y columns and the cached
/// SY/SS/WT/WN/WN1 workspaces across outer iterations.  Its operation order is
/// the one documented by the permissive language-neutral subroutine specs.
#[derive(Debug)]
pub(super) struct ExactCompactState {
    profile: ArithmeticProfile,
    n: usize,
    memory: usize,
    ws: Vec<Vec<f64>>,
    wy: Vec<Vec<f64>>,
    sy: Vec<Vec<f64>>,
    ss: Vec<Vec<f64>>,
    wt: Vec<Vec<f64>>,
    wn: Vec<Vec<f64>>,
    wn1: Vec<Vec<f64>>,
    head: usize,
    itail: usize,
    col: usize,
    iupdat: usize,
    theta: f64,
    updatd: bool,
    iwhere: Vec<i32>,
    index: Vec<usize>,
    indx2: Vec<usize>,
    nfree_previous: usize,
    iteration: usize,
}

#[derive(Debug)]
struct ExactCauchyPoint {
    x: Vec<f64>,
    c: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct FreePartition {
    nfree: usize,
    nenter: usize,
    ileave: usize,
    work: bool,
}

impl ExactCompactState {
    pub(super) fn new(
        profile: ArithmeticProfile,
        dimension: usize,
        memory: usize,
        lower: &[f64],
        upper: &[f64],
    ) -> Self {
        let memory = memory.max(1);
        let mut iwhere = vec![0; dimension];
        for index in 0..dimension {
            iwhere[index] = match coherent_bound_type(lower[index], upper[index]) {
                0 => -1,
                2 if upper[index] - lower[index] <= 0.0 => 3,
                _ => 0,
            };
        }
        Self {
            profile,
            n: dimension,
            memory,
            ws: vec![vec![0.0; dimension]; memory],
            wy: vec![vec![0.0; dimension]; memory],
            sy: vec![vec![0.0; memory]; memory],
            ss: vec![vec![0.0; memory]; memory],
            wt: vec![vec![0.0; memory]; memory],
            wn: vec![vec![0.0; 2 * memory]; 2 * memory],
            wn1: vec![vec![0.0; 2 * memory]; 2 * memory],
            head: 0,
            itail: 0,
            col: 0,
            iupdat: 0,
            theta: 1.0,
            updatd: false,
            iwhere,
            index: (0..dimension).collect(),
            indx2: vec![0; dimension],
            nfree_previous: 0,
            iteration: 0,
        }
    }

    pub(super) fn clear_history(&mut self) {
        self.head = 0;
        self.itail = 0;
        self.col = 0;
        self.iupdat = 0;
        self.theta = 1.0;
        self.updatd = false;
        for matrix in [
            &mut self.sy,
            &mut self.ss,
            &mut self.wt,
            &mut self.wn,
            &mut self.wn1,
        ] {
            for row in matrix {
                row.fill(0.0);
            }
        }
    }

    pub(super) fn snapshot(&self) -> crate::ModelSnapshot {
        crate::ModelSnapshot {
            active: self.n - self.nfree_previous,
            free: self.nfree_previous,
            history: self.col,
        }
    }

    pub(super) fn has_history(&self) -> bool {
        self.col > 0
    }

    pub(super) fn direction(
        &mut self,
        x: &[f64],
        gradient: &[f64],
        lower: &[f64],
        upper: &[f64],
        projected_gradient_norm: f64,
        _min_step: f64,
    ) -> Result<DirectionChoice, RecoveryReason> {
        debug_assert_eq!(x.len(), self.n);
        // With no constraints and a retained model, solve directly in the
        // full subspace. Computing and cancelling an unnecessary Cauchy step
        // changes rounding and can amplify drift on ill-conditioned models.
        let cauchy = if self.col > 0 && lower.iter().chain(upper).all(|bound| bound.is_infinite()) {
            ExactCauchyPoint {
                x: x.to_vec(),
                c: vec![0.0; 2 * self.col],
            }
        } else {
            self.cauchy(x, gradient, lower, upper, projected_gradient_norm)
                .ok_or(RecoveryReason::CauchyModel)?
        };
        let partition = self.partition_free_variables();
        let reduced = self
            .reduced_gradient(x, gradient, &cauchy.x, &cauchy.c, partition.nfree)
            .ok_or(RecoveryReason::ReducedGradient)?;

        let mut target = cauchy.x;
        if partition.nfree > 0 && self.col > 0 {
            if partition.work && !self.formk(partition) {
                self.clear_history();
                return Err(RecoveryReason::SubspaceFactorization);
            }
            target = self
                .subspace_minimizer(lower, upper, target, reduced, partition.nfree)
                .ok_or(RecoveryReason::SubspaceSolve)?;
        }

        self.nfree_previous = partition.nfree;
        self.iteration += 1;
        let direction = difference(&target, x);
        Ok(DirectionChoice {
            direction,
            unit_step_target: Some(target),
            refresh_history: false,
        })
    }

    /// Apply the public `mainlb`/`matupd` state handoff.  `direction` is the
    /// unscaled line-search direction; on non-unit steps the runtime scales a
    /// copy by `alpha` before storing it in WS.
    pub(super) fn update(
        &mut self,
        direction: &[f64],
        alpha: f64,
        gradient_difference: &[f64],
        initial_derivative: f64,
        final_derivative: f64,
    ) -> HistoryUpdate {
        debug_assert_eq!(direction.len(), self.n);
        let rr = self
            .profile
            .short_dot(gradient_difference, gradient_difference);
        let derivative_delta = final_derivative - initial_derivative;
        let dr = if alpha == 1.0 {
            derivative_delta
        } else {
            derivative_delta * alpha
        };
        let ddum = if alpha == 1.0 {
            -initial_derivative
        } else {
            -initial_derivative * alpha
        };
        if dr <= f64::EPSILON * ddum {
            self.updatd = false;
            return HistoryUpdate::SkippedCurvature;
        }

        let dtd = self.profile.short_dot(direction, direction);
        self.iupdat += 1;
        if self.iupdat <= self.memory {
            self.col = self.iupdat;
            self.itail = (self.head + self.iupdat - 1) % self.memory;
        } else {
            self.itail = (self.itail + 1) % self.memory;
            self.head = (self.head + 1) % self.memory;
        }

        self.ws[self.itail].copy_from_slice(direction);
        if alpha != 1.0 {
            for value in &mut self.ws[self.itail] {
                *value *= alpha;
            }
        }
        self.wy[self.itail].copy_from_slice(gradient_difference);
        self.theta = rr / dr;

        if self.iupdat > self.memory {
            for column in 0..self.col - 1 {
                for row in 0..=column {
                    self.ss[row][column] = self.ss[row + 1][column + 1];
                }
                for row in column..self.col - 1 {
                    self.sy[row][column] = self.sy[row + 1][column + 1];
                }
            }
        }

        let mut pointer = self.head;
        for previous in 0..self.col - 1 {
            self.sy[self.col - 1][previous] = self
                .profile
                .short_dot(&self.ws[self.itail], &self.wy[pointer]);
            self.ss[previous][self.col - 1] = self
                .profile
                .short_dot(&self.ws[pointer], &self.ws[self.itail]);
            pointer = (pointer + 1) % self.memory;
        }
        self.ss[self.col - 1][self.col - 1] = if alpha == 1.0 {
            dtd
        } else {
            alpha * alpha * dtd
        };
        self.sy[self.col - 1][self.col - 1] = dr;

        if !self.formt() {
            self.clear_history();
            return HistoryUpdate::ResetFactorization;
        }
        self.updatd = true;
        HistoryUpdate::Stored
    }

    fn formt(&mut self) -> bool {
        if self.col == 0 {
            return true;
        }
        for column in 0..self.col {
            self.wt[0][column] = self.theta * self.ss[0][column];
        }
        for row in 1..self.col {
            for column in row..self.col {
                let mut sum = 0.0;
                for index in 0..row {
                    sum += self.sy[row][index] * self.sy[column][index] / self.sy[index][index];
                }
                self.wt[row][column] = self.profile.mul_add(self.theta, self.ss[row][column], sum);
            }
        }
        coherent_cholesky_upper_in_place(self.profile, &mut self.wt, 0, self.col).is_some()
    }

    fn bmv(&self, vector: &[f64]) -> Option<Vec<f64>> {
        let col = self.col;
        if vector.len() != 2 * col {
            return None;
        }
        if col == 0 {
            return Some(Vec::new());
        }
        let mut product = vec![0.0; 2 * col];
        product[col] = vector[col];
        for row in 1..col {
            let mut sum = 0.0;
            for (index, &value) in vector.iter().take(row).enumerate() {
                sum += self.sy[row][index] * value / self.sy[index][index];
            }
            product[col + row] = vector[col + row] + sum;
        }
        coherent_solve_upper_transpose_in_place(
            self.profile,
            &self.wt,
            0,
            col,
            &mut product[col..],
        )?;
        for (index, (target, &value)) in product[..col]
            .iter_mut()
            .zip(vector[..col].iter())
            .enumerate()
        {
            *target = value / self.sy[index][index].sqrt();
        }
        coherent_solve_upper_in_place(&self.wt, 0, col, &mut product[col..])?;
        let (product_front, product_back) = product.split_at_mut(col);
        for (index, value) in product_front.iter_mut().enumerate() {
            *value = -*value / self.sy[index][index].sqrt();
        }
        for (index, value) in product_front.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (row, &back_value) in product_back.iter().enumerate().skip(index + 1) {
                sum += self.sy[row][index] * back_value / self.sy[index][index];
            }
            *value += sum;
        }
        Some(product)
    }

    fn cauchy(
        &mut self,
        x: &[f64],
        gradient: &[f64],
        lower: &[f64],
        upper: &[f64],
        projected_gradient_norm: f64,
    ) -> Option<ExactCauchyPoint> {
        if projected_gradient_norm <= 0.0 {
            return Some(ExactCauchyPoint {
                x: x.to_vec(),
                c: vec![0.0; 2 * self.col],
            });
        }

        let col2 = 2 * self.col;
        let mut p = vec![0.0; col2];
        let mut c = vec![0.0; col2];
        let mut direction = vec![0.0; self.n];
        let mut breakpoints = vec![0.0; self.n];
        let mut order = vec![0_usize; self.n];
        let mut bounded = true;
        let mut nfree_marker = self.n + 1;
        let mut nbreak = 0_usize;
        let mut minimum_slot = 0_usize;
        let mut minimum_breakpoint = 0.0;
        let mut f1 = 0.0;

        for index in 0..self.n {
            let neggi = -gradient[index];
            let bound_type = coherent_bound_type(lower[index], upper[index]);
            if self.iwhere[index] != 3 && self.iwhere[index] != -1 {
                let mut lower_distance = 0.0;
                let mut upper_distance = 0.0;
                if bound_type <= 2 {
                    lower_distance = x[index] - lower[index];
                }
                if bound_type >= 2 {
                    upper_distance = upper[index] - x[index];
                }
                let at_lower = bound_type <= 2 && lower_distance <= 0.0;
                let at_upper = bound_type >= 2 && upper_distance <= 0.0;
                self.iwhere[index] = 0;
                if at_lower {
                    if neggi <= 0.0 {
                        self.iwhere[index] = 1;
                    }
                } else if at_upper {
                    if neggi >= 0.0 {
                        self.iwhere[index] = 2;
                    }
                } else if neggi.abs() <= 0.0 {
                    self.iwhere[index] = -3;
                }
            }

            if self.iwhere[index] != 0 && self.iwhere[index] != -1 {
                direction[index] = 0.0;
                continue;
            }

            direction[index] = neggi;
            f1 = self.profile.mul_add(-neggi, neggi, f1);
            let mut pointer = self.head;
            for history_index in 0..self.col {
                p[history_index] =
                    self.profile
                        .mul_add(self.wy[pointer][index], neggi, p[history_index]);
                p[self.col + history_index] = self.profile.mul_add(
                    self.ws[pointer][index],
                    neggi,
                    p[self.col + history_index],
                );
                pointer = (pointer + 1) % self.memory;
            }

            if bound_type <= 2 && bound_type != 0 && neggi < 0.0 {
                let distance = x[index] - lower[index];
                order[nbreak] = index;
                breakpoints[nbreak] = distance / -neggi;
                if nbreak == 0 || breakpoints[nbreak] < minimum_breakpoint {
                    minimum_breakpoint = breakpoints[nbreak];
                    minimum_slot = nbreak;
                }
                nbreak += 1;
            } else if bound_type >= 2 && neggi > 0.0 {
                let distance = upper[index] - x[index];
                order[nbreak] = index;
                breakpoints[nbreak] = distance / neggi;
                if nbreak == 0 || breakpoints[nbreak] < minimum_breakpoint {
                    minimum_breakpoint = breakpoints[nbreak];
                    minimum_slot = nbreak;
                }
                nbreak += 1;
            } else {
                nfree_marker -= 1;
                order[nfree_marker - 1] = index;
                if neggi.abs() > 0.0 {
                    bounded = false;
                }
            }
        }

        if self.theta != 1.0 {
            for index in 0..self.col {
                p[self.col + index] *= self.theta;
            }
        }

        let mut point = x.to_vec();
        if nbreak == 0 && nfree_marker == self.n + 1 {
            return Some(ExactCauchyPoint { x: point, c });
        }

        let mut f2 = -self.theta * f1;
        let f2_original = f2;
        if self.col > 0 {
            let inverse_p = self.bmv(&p)?;
            f2 -= self.profile.short_dot(&inverse_p, &p);
        }
        let mut dtm = if f2 != 0.0 { -f1 / f2 } else { 0.0 };
        let mut total = 0.0;

        if nbreak > 0 {
            let mut nleft = nbreak;
            let mut walk_iteration = 1_usize;
            let mut breakpoint = 0.0;
            loop {
                let previous_breakpoint = breakpoint;
                let bound_index;
                if walk_iteration == 1 {
                    breakpoint = minimum_breakpoint;
                    bound_index = order[minimum_slot];
                } else {
                    if walk_iteration == 2 && minimum_slot != nbreak - 1 {
                        breakpoints[minimum_slot] = breakpoints[nbreak - 1];
                        order[minimum_slot] = order[nbreak - 1];
                    }
                    coherent_hpsolb(nleft, &mut breakpoints, &mut order, walk_iteration - 2);
                    breakpoint = breakpoints[nleft - 1];
                    bound_index = order[nleft - 1];
                }

                let interval = breakpoint - previous_breakpoint;
                if dtm < interval {
                    break;
                }

                total += interval;
                nleft -= 1;
                walk_iteration += 1;
                let bound_direction = direction[bound_index];
                direction[bound_index] = 0.0;
                let bound_displacement;
                if bound_direction > 0.0 {
                    bound_displacement = upper[bound_index] - x[bound_index];
                    point[bound_index] = upper[bound_index];
                    self.iwhere[bound_index] = 2;
                } else {
                    bound_displacement = lower[bound_index] - x[bound_index];
                    point[bound_index] = lower[bound_index];
                    self.iwhere[bound_index] = 1;
                }

                if nleft == 0 && nbreak == self.n {
                    dtm = interval;
                    break;
                }

                let bound_direction_squared = bound_direction * bound_direction;
                let partial = self.profile.mul_add(interval, f2, bound_direction_squared);
                let correction = self.profile.mul_add(
                    -self.theta * bound_direction,
                    bound_displacement,
                    partial,
                );
                f1 += correction;
                f2 = self
                    .profile
                    .mul_add(-self.theta, bound_direction_squared, f2);

                if self.col > 0 {
                    coherent_daxpy(&mut c, &p, interval);
                    let mut wbp = vec![0.0; col2];
                    let mut pointer = self.head;
                    for history_index in 0..self.col {
                        wbp[history_index] = self.wy[pointer][bound_index];
                        wbp[self.col + history_index] = self.theta * self.ws[pointer][bound_index];
                        pointer = (pointer + 1) % self.memory;
                    }
                    let inverse_wbp = self.bmv(&wbp)?;
                    let wmc = self.profile.short_dot(&c, &inverse_wbp);
                    let wmp = self.profile.short_dot(&p, &inverse_wbp);
                    let wmw = self.profile.short_dot(&wbp, &inverse_wbp);
                    coherent_daxpy(&mut p, &wbp, -bound_direction);
                    f1 = self.profile.mul_add(bound_direction, wmc, f1);
                    f2 += self.profile.mul_add(
                        2.0 * bound_direction,
                        wmp,
                        -(bound_direction_squared * wmw),
                    );
                }

                f2 = f2.max(f64::EPSILON * f2_original);
                if nleft > 0 {
                    dtm = -f1 / f2;
                    continue;
                }
                if bounded {
                    dtm = 0.0;
                } else {
                    dtm = -f1 / f2;
                }
                break;
            }
        }

        if dtm <= 0.0 {
            dtm = 0.0;
        }
        total += dtm;
        coherent_daxpy(&mut point, &direction, total);
        if self.col > 0 {
            coherent_daxpy(&mut c, &p, dtm);
        }
        Some(ExactCauchyPoint { x: point, c })
    }

    fn partition_free_variables(&mut self) -> FreePartition {
        let mut nenter = 0_usize;
        let mut ileave = self.n;
        if self.iteration > 0 {
            for slot in 0..self.nfree_previous {
                let index = self.index[slot];
                if self.iwhere[index] > 0 {
                    ileave -= 1;
                    self.indx2[ileave] = index;
                }
            }
            for slot in self.nfree_previous..self.n {
                let index = self.index[slot];
                if self.iwhere[index] <= 0 {
                    self.indx2[nenter] = index;
                    nenter += 1;
                }
            }
        }

        let work = ileave < self.n || nenter > 0 || self.updatd;
        let mut nfree = 0_usize;
        let mut active_slot = self.n;
        for index in 0..self.n {
            if self.iwhere[index] <= 0 {
                self.index[nfree] = index;
                nfree += 1;
            } else {
                active_slot -= 1;
                self.index[active_slot] = index;
            }
        }
        FreePartition {
            nfree,
            nenter,
            ileave,
            work,
        }
    }

    fn reduced_gradient(
        &self,
        x: &[f64],
        gradient: &[f64],
        cauchy: &[f64],
        c: &[f64],
        nfree: usize,
    ) -> Option<Vec<f64>> {
        let mut reduced = vec![0.0; nfree];
        for (slot, value) in reduced.iter_mut().enumerate() {
            let index = self.index[slot];
            *value = self
                .profile
                .mul_add(-self.theta, cauchy[index] - x[index], -gradient[index]);
        }
        if self.col == 0 {
            return Some(reduced);
        }

        let middle = self.bmv(c)?;
        let mut pointer = self.head;
        for history_index in 0..self.col {
            let a1 = middle[history_index];
            let a2 = self.theta * middle[self.col + history_index];
            for (slot, value) in reduced.iter_mut().enumerate() {
                let index = self.index[slot];
                *value +=
                    self.profile
                        .mul_add(self.wy[pointer][index], a1, self.ws[pointer][index] * a2);
            }
            pointer = (pointer + 1) % self.memory;
        }
        Some(reduced)
    }

    fn formk(&mut self, partition: FreePartition) -> bool {
        let m = self.memory;
        if self.updatd {
            if self.iupdat > m {
                for column in 0..m - 1 {
                    let length = m - column - 1;
                    for offset in 0..length {
                        self.wn1[column + offset][column] =
                            self.wn1[column + offset + 1][column + 1];
                        self.wn1[m + column + offset][m + column] =
                            self.wn1[m + column + offset + 1][m + column + 1];
                    }
                    for offset in 0..m - 1 {
                        self.wn1[m + offset][column] = self.wn1[m + offset + 1][column + 1];
                    }
                }
            }

            let newest_logical = self.col - 1;
            let newest_storage = (self.head + self.col - 1) % m;
            let mut previous_storage = self.head;
            for previous_logical in 0..self.col {
                let mut yy_free = 0.0;
                let mut ss_active = 0.0;
                let mut sy_active = 0.0;
                for slot in 0..partition.nfree {
                    let index = self.index[slot];
                    yy_free = self.profile.mul_add(
                        self.wy[newest_storage][index],
                        self.wy[previous_storage][index],
                        yy_free,
                    );
                }
                for slot in partition.nfree..self.n {
                    let index = self.index[slot];
                    ss_active = self.profile.mul_add(
                        self.ws[newest_storage][index],
                        self.ws[previous_storage][index],
                        ss_active,
                    );
                    sy_active = self.profile.mul_add(
                        self.ws[newest_storage][index],
                        self.wy[previous_storage][index],
                        sy_active,
                    );
                }
                self.wn1[newest_logical][previous_logical] = yy_free;
                self.wn1[m + newest_logical][m + previous_logical] = ss_active;
                self.wn1[m + newest_logical][previous_logical] = sy_active;
                previous_storage = (previous_storage + 1) % m;
            }

            let mut row_storage = self.head;
            for row_logical in 0..self.col {
                let mut sy_free = 0.0;
                for slot in 0..partition.nfree {
                    let index = self.index[slot];
                    sy_free = self.profile.mul_add(
                        self.ws[row_storage][index],
                        self.wy[newest_storage][index],
                        sy_free,
                    );
                }
                self.wn1[m + row_logical][newest_logical] = sy_free;
                row_storage = (row_storage + 1) % m;
            }
        }

        let update_columns = if self.updatd { self.col - 1 } else { self.col };
        let mut row_storage = self.head;
        for row_logical in 0..update_columns {
            let mut column_storage = self.head;
            for column_logical in 0..=row_logical {
                let mut yy_enter = 0.0;
                let mut ss_enter = 0.0;
                let mut yy_leave = 0.0;
                let mut ss_leave = 0.0;
                for slot in 0..partition.nenter {
                    let index = self.indx2[slot];
                    yy_enter = self.profile.mul_add(
                        self.wy[row_storage][index],
                        self.wy[column_storage][index],
                        yy_enter,
                    );
                    ss_enter = self.profile.mul_add(
                        self.ws[row_storage][index],
                        self.ws[column_storage][index],
                        ss_enter,
                    );
                }
                for slot in partition.ileave..self.n {
                    let index = self.indx2[slot];
                    yy_leave = self.profile.mul_add(
                        self.wy[row_storage][index],
                        self.wy[column_storage][index],
                        yy_leave,
                    );
                    ss_leave = self.profile.mul_add(
                        self.ws[row_storage][index],
                        self.ws[column_storage][index],
                        ss_leave,
                    );
                }
                // Add entrants to the cached YY product before subtracting
                // departures. Pre-combining both sums changes reference rounding.
                self.wn1[row_logical][column_logical] =
                    (self.wn1[row_logical][column_logical] + yy_enter) - yy_leave;
                self.wn1[m + row_logical][m + column_logical] += -ss_enter + ss_leave;
                column_storage = (column_storage + 1) % m;
            }
            row_storage = (row_storage + 1) % m;
        }

        row_storage = self.head;
        for row_logical in 0..update_columns {
            let mut column_storage = self.head;
            for column_logical in 0..update_columns {
                let mut sy_enter = 0.0;
                let mut sy_leave = 0.0;
                for slot in 0..partition.nenter {
                    let index = self.indx2[slot];
                    sy_enter = self.profile.mul_add(
                        self.ws[row_storage][index],
                        self.wy[column_storage][index],
                        sy_enter,
                    );
                }
                for slot in partition.ileave..self.n {
                    let index = self.indx2[slot];
                    sy_leave = self.profile.mul_add(
                        self.ws[row_storage][index],
                        self.wy[column_storage][index],
                        sy_leave,
                    );
                }
                if row_logical <= column_logical {
                    self.wn1[m + row_logical][column_logical] += sy_enter - sy_leave;
                } else {
                    self.wn1[m + row_logical][column_logical] += -sy_enter + sy_leave;
                }
                column_storage = (column_storage + 1) % m;
            }
            row_storage = (row_storage + 1) % m;
        }

        let col = self.col;
        for row in 0..col {
            for column in 0..=row {
                self.wn[column][row] = self.wn1[row][column] / self.theta;
                self.wn[col + column][col + row] = self.wn1[m + row][m + column] * self.theta;
            }
            for column in 0..row {
                self.wn[column][col + row] = -self.wn1[m + row][column];
            }
            for column in row..col {
                self.wn[column][col + row] = self.wn1[m + row][column];
            }
            self.wn[row][row] += self.sy[row][row];
        }

        if coherent_cholesky_upper_in_place(self.profile, &mut self.wn, 0, col).is_none() {
            return false;
        }
        let mut rhs = vec![0.0; col];
        for column in col..2 * col {
            for (row, value) in rhs.iter_mut().enumerate() {
                *value = self.wn[row][column];
            }
            if coherent_solve_upper_transpose_in_place(self.profile, &self.wn, 0, col, &mut rhs)
                .is_none()
            {
                return false;
            }
            for (row, &value) in rhs.iter().enumerate() {
                self.wn[row][column] = value;
            }
        }
        for row in col..2 * col {
            for column in row..2 * col {
                let update = self
                    .profile
                    .indexed_dot(col, |index| (self.wn[index][row], self.wn[index][column]));
                self.wn[row][column] += update;
            }
        }
        coherent_cholesky_upper_in_place(self.profile, &mut self.wn, col, col).is_some()
    }

    fn subspace_minimizer(
        &self,
        lower: &[f64],
        upper: &[f64],
        mut point: Vec<f64>,
        mut direction: Vec<f64>,
        nfree: usize,
    ) -> Option<Vec<f64>> {
        let col = self.col;
        let mut wv = vec![0.0; 2 * col];
        let mut pointer = self.head;
        for history_index in 0..col {
            let mut y_dot = 0.0;
            let mut s_dot = 0.0;
            for (slot, &direction_value) in direction.iter().take(nfree).enumerate() {
                let index = self.index[slot];
                y_dot = self
                    .profile
                    .mul_add(self.wy[pointer][index], direction_value, y_dot);
                s_dot = self
                    .profile
                    .mul_add(self.ws[pointer][index], direction_value, s_dot);
            }
            wv[history_index] = y_dot;
            wv[col + history_index] = self.theta * s_dot;
            pointer = (pointer + 1) % self.memory;
        }
        coherent_solve_upper_transpose_in_place(self.profile, &self.wn, 0, 2 * col, &mut wv)?;
        for value in wv.iter_mut().take(col) {
            *value = -*value;
        }
        coherent_solve_upper_in_place(&self.wn, 0, 2 * col, &mut wv)?;

        pointer = self.head;
        for history_index in 0..col {
            for (slot, value) in direction.iter_mut().take(nfree).enumerate() {
                let index = self.index[slot];
                *value += self.profile.mul_add(
                    self.ws[pointer][index],
                    wv[col + history_index],
                    self.wy[pointer][index] * wv[history_index] / self.theta,
                );
            }
            pointer = (pointer + 1) % self.memory;
        }
        for value in direction.iter_mut().take(nfree) {
            *value /= self.theta;
        }

        // R 4.6 reports the bundled algorithm as L-BFGS-B 2.3.  Its observed
        // bounded subspace endpoint is the original single feasible step along
        // d (all coordinates share alpha), rather than the later independent
        // projection/Morales safeguard described by the v3 public spec.
        let (alpha, binding_slot) =
            exact_shared_feasible_step(&point, &direction, &self.index, nfree, lower, upper);
        if alpha < 1.0 {
            if let Some(binding_slot) = binding_slot {
                let index = self.index[binding_slot];
                if direction[binding_slot] > 0.0 {
                    point[index] = upper[index];
                    direction[binding_slot] = 0.0;
                } else if direction[binding_slot] < 0.0 {
                    point[index] = lower[index];
                    direction[binding_slot] = 0.0;
                }
            }
        }
        for (slot, &step) in direction.iter().take(nfree).enumerate() {
            let index = self.index[slot];
            point[index] = self.profile.mul_add(alpha, step, point[index]);
        }
        Some(point)
    }
}

fn exact_shared_feasible_step(
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

fn coherent_cholesky_upper_in_place(
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

fn coherent_solve_upper_transpose_in_place(
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

fn coherent_solve_upper_in_place(
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

fn coherent_bound_type(lower: f64, upper: f64) -> i32 {
    match (lower.is_finite(), upper.is_finite()) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

fn coherent_daxpy(target: &mut [f64], source: &[f64], alpha: f64) {
    debug_assert_eq!(target.len(), source.len());
    for (target, &source) in target.iter_mut().zip(source.iter()) {
        *target = alpha.mul_add(source, *target);
    }
}

fn coherent_hpsolb(count: usize, values: &mut [f64], indices: &mut [usize], heap_state: usize) {
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

#[cfg(test)]
mod exact_tests {
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
}

#[cfg(test)]
mod factorization_tests {
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
            coherent_cholesky_upper_in_place(ArithmeticProfile::legacy(), &mut matrix, 0, n)
                .unwrap();
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
}

#[cfg(test)]
mod model_tests;
