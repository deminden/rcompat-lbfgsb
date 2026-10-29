use super::*;

impl ExactCompactState {
    pub(super) fn cauchy(
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

    pub(super) fn partition_free_variables(&mut self) -> FreePartition {
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
}
