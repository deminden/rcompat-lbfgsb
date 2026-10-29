use super::*;

impl ExactCompactState {
    /// Apply the public `mainlb`/`matupd` state handoff.  `direction` is the
    /// unscaled line-search direction; on non-unit steps the runtime scales a
    /// copy by `alpha` before storing it in WS.
    pub(in crate::backend::native) fn update(
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

    pub(super) fn formt(&mut self) -> bool {
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

    pub(super) fn bmv(&self, vector: &[f64]) -> Option<Vec<f64>> {
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
}
