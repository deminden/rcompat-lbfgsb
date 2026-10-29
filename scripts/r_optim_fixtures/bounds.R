write_fixture(
  "quadratic",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
)

write_fixture(
  "first_constrained_linear_2d",
  initial_par = c(0.0, 0.0),
  lower = c(-100.0, -100.0),
  upper = c(100.0, 100.0),
  control = list(),
  fn = function(p) -p[[1]] - 0.5 * p[[2]],
  gr = function(p) c(-1.0, -0.5)
)

write_fixture(
  "zero_dim",
  initial_par = numeric(0),
  lower = numeric(0),
  upper = numeric(0),
  control = list(),
  fn = function(p) 123.0
)

write_fixture(
  "zero_dim_with_gradient",
  initial_par = numeric(0),
  lower = numeric(0),
  upper = numeric(0),
  control = list(),
  fn = function(p) 123.0,
  gr = function(p) stop("zero-dimensional gradient should not be called")
)

write_fixture(
  "lower_bound_finite_difference_trace_probe",
  initial_par = c(0.0, 1.0),
  lower = c(0.0, -1.0),
  upper = c(2.0, 1.0),
  control = list(ndeps = c(0.1, 0.1), maxit = 0),
  fn = function(p) p[[1]] + 2.0 * p[[2]]
)

write_trace_fixture(
  "lower_bound_finite_difference_trace",
  source_fixture = "lower_bound_finite_difference_trace_probe",
  initial_par = c(0.0, 1.0),
  lower = c(0.0, -1.0),
  upper = c(2.0, 1.0),
  control = list(ndeps = c(0.1, 0.1), factr = 1e7, pgtol = 0.0, lmm = 5, maxit = 0),
  fn = function(p) p[[1]] + 2.0 * p[[2]],
  gr = NULL,
  prefix_len = 5
)

write_fixture(
  "active_bounds",
  initial_par = c(3.0),
  lower = c(0.0),
  upper = c(10.0),
  control = list(),
  fn = function(p) (p[[1]] + 2.0)^2
)

write_fixture(
  "finite_difference",
  initial_par = c(6.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(ndeps = c(1e-4)),
  fn = function(p) (p[[1]] + 1.0)^2
)

write_fixture(
  "active_upper_bound",
  initial_par = c(-3.0),
  lower = c(-10.0),
  upper = c(0.0),
  control = list(),
  fn = function(p) (p[[1]] - 2.0)^2
)

write_fixture(
  "fixed_parameter",
  initial_par = c(1.5),
  lower = c(1.5),
  upper = c(1.5),
  control = list(),
  fn = function(p) (p[[1]] - 10.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 10.0))
)

write_fixture(
  "fixed_free_2d",
  initial_par = c(1.5, 0.0),
  lower = c(1.5, -10.0),
  upper = c(1.5, 10.0),
  control = list(),
  fn = function(p) (p[[1]] - 10.0)^2 + (p[[2]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 10.0), 2.0 * (p[[2]] - 2.0))
)

write_fixture(
  "all_fixed_2d",
  initial_par = c(1.5, -2.0),
  lower = c(1.5, -2.0),
  upper = c(1.5, -2.0),
  control = list(),
  fn = function(p) (p[[1]] - 10.0)^2 + (p[[2]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 10.0), 2.0 * (p[[2]] - 2.0))
)

write_fixture(
  "initial_projected_gradient",
  initial_par = c(0.0),
  lower = c(0.0),
  upper = c(10.0),
  control = list(),
  fn = function(p) (p[[1]] + 2.0)^2
)

write_fixture(
  "initial_projected_gradient_2d",
  initial_par = c(0.0, -1.0),
  lower = c(0.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(),
  fn = function(p) (p[[1]] + 2.0)^2 + (p[[2]] + 1.0)^2,
  gr = function(p) c(2.0 * (p[[1]] + 2.0), 2.0 * (p[[2]] + 1.0))
)

write_fixture(
  "unbounded_quadratic",
  initial_par = c(-5.0),
  lower = c(-Inf),
  upper = c(Inf),
  control = list(),
  fn = function(p) (p[[1]] - 1.25)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.25))
)

write_fixture(
  "mixed_bounds_quadratic",
  initial_par = c(3.0, -4.0),
  lower = c(0.0, -Inf),
  upper = c(Inf, 0.0),
  control = list(),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "mixed_bounds_finite_difference_2d",
  initial_par = c(3.0, -4.0),
  lower = c(0.0, -Inf),
  upper = c(Inf, 0.0),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "lower_bounded_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(Inf, Inf),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "upper_bounded_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-Inf, -Inf),
  upper = c(10.0, 10.0),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "all_unbounded_2d_quadratic",
  initial_par = c(4.0, -5.0),
  lower = c(-Inf, -Inf),
  upper = c(Inf, Inf),
  control = list(),
  fn = function(p) (p[[1]] - 1.0)^2 + 2.0 * (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 4.0 * (p[[2]] + 2.0))
)

write_fixture(
  "all_unbounded_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-Inf, -Inf),
  upper = c(Inf, Inf),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "three_dim_quadratic",
  initial_par = c(3.0, -4.0, 0.5),
  lower = c(-10.0, -10.0, -10.0),
  upper = c(10.0, 10.0, 10.0),
  control = list(),
  fn = function(p) (p[[1]] - 1.0)^2 + 3.0 * (p[[2]] + 2.0)^2 + 0.5 * (p[[3]] - 0.25)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 6.0 * (p[[2]] + 2.0), p[[3]] - 0.25)
)

write_fixture(
  "three_dim_box_active",
  initial_par = c(3.0, -4.0, 0.5),
  lower = c(0.0, -3.0, -1.0),
  upper = c(2.0, 0.0, 1.0),
  control = list(),
  fn = function(p) (p[[1]] - 5.0)^2 + (p[[2]] + 2.0)^2 + (p[[3]] - 0.25)^2,
  gr = function(p) c(2.0 * (p[[1]] - 5.0), 2.0 * (p[[2]] + 2.0), 2.0 * (p[[3]] - 0.25))
)

write_fixture(
  "near_box_finite_difference_2d",
  initial_par = c(0.99995, -0.99995),
  lower = c(-1.0, -1.0),
  upper = c(1.0, 1.0),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 0.25)^2 + 2.0 * (p[[2]] + 0.25)^2
)

write_fixture(
  "near_upper_finite_difference",
  initial_par = c(0.99995),
  lower = c(-1.0),
  upper = c(1.0),
  control = list(ndeps = c(1e-4)),
  fn = function(p) (p[[1]] - 0.25)^2
)

write_fixture(
  "near_lower_finite_difference",
  initial_par = c(-0.99995),
  lower = c(-1.0),
  upper = c(1.0),
  control = list(ndeps = c(1e-4)),
  fn = function(p) (p[[1]] + 0.25)^2
)

write_fixture(
  "initial_outside_bounds",
  initial_par = c(2.0),
  lower = c(0.0),
  upper = c(1.0),
  control = list(),
  fn = function(p) p[[1]]^2
)

write_fixture(
  "initial_pos_inf_projected",
  initial_par = c(Inf),
  lower = c(-1.0),
  upper = c(1.0),
  control = list(),
  fn = function(p) p[[1]]^2,
  gr = function(p) c(2.0 * p[[1]])
)

write_fixture(
  "initial_neg_inf_projected",
  initial_par = c(-Inf),
  lower = c(-1.0),
  upper = c(1.0),
  control = list(),
  fn = function(p) p[[1]]^2,
  gr = function(p) c(2.0 * p[[1]])
)

write_error_fixture(
  "fixed_no_gradient_error",
  initial_par = c(1.5),
  lower = c(1.5),
  upper = c(1.5),
  control = list(),
  fn = function(p) (p[[1]] - 10.0)^2
)
