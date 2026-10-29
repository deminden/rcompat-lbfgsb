use super::*;

impl ExactCompactState {
    pub(super) fn reduced_gradient(
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

    pub(super) fn formk(&mut self, partition: FreePartition) -> bool {
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

    pub(super) fn subspace_minimizer(
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
