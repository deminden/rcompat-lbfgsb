write_fixture(
  "factr_zero_quadratic",
  initial_par = c(10.0),
  lower = c(-Inf),
  upper = c(Inf),
  control = list(factr = 0.0),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
)

write_fixture(
  "fnscale",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(fnscale = -1.0),
  fn = function(p) -(p[[1]] - 3.0)^2
)

write_fixture(
  "parscale",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(parscale = c(2.0)),
  fn = function(p) (p[[1]] - 4.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 4.0))
)

write_fixture(
  "maxit",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(maxit = 1),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
)

write_fixture(
  "parscale_bounds_gradient_2d",
  initial_par = c(0.0, 0.0),
  lower = c(-4.0, -1.0),
  upper = c(8.0, 1.0),
  control = list(parscale = c(2.0, 0.5)),
  fn = function(p) (p[[1]] - 6.0)^2 + 2.0 * (p[[2]] - 0.75)^2,
  gr = function(p) c(2.0 * (p[[1]] - 6.0), 4.0 * (p[[2]] - 0.75))
)

write_fixture(
  "parscale_finite_difference",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(parscale = c(2.0), ndeps = c(1e-4)),
  fn = function(p) (p[[1]] - 4.0)^2
)

write_fixture(
  "two_dim_parscale_finite_difference",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(parscale = c(2.0, 0.5), ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + 3.0 * (p[[2]] + 2.0)^2
)

write_fixture(
  "maxit_zero",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(maxit = 0),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
)

write_fixture(
  "maxit_zero_2d_quadratic",
  initial_par = c(3.0, -4.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(maxit = 0),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "maxit_zero_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(maxit = 0, ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "maxit_one_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(maxit = 1, ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "pgtol_initial",
  initial_par = c(1.95),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(pgtol = 0.2),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
)

write_fixture(
  "pgtol_near_upper_uses_bound_distance",
  initial_par = c(0.99),
  lower = c(-10.0),
  upper = c(1.0),
  control = list(pgtol = 0.02),
  fn = function(p) -10.0 * p[[1]],
  gr = function(p) c(-10.0)
)

write_fixture(
  "pgtol_initial_finite_difference_2d",
  initial_par = c(1.01, -2.02),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(pgtol = 0.05, ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "pgtol_finite_difference_2d",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(pgtol = 0.25, ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)

write_fixture(
  "pgtol_after_step_2d",
  initial_par = c(6.0, -6.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(pgtol = 0.5),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "maxit_one_2d_quadratic",
  initial_par = c(3.0, -4.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(maxit = 1),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "factr_zero_2d_quadratic",
  initial_par = c(3.0, -4.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(factr = 0.0),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "factr_loose_2d_quadratic",
  initial_par = c(3.0, -4.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(factr = 1e12),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 1.0), 2.0 * (p[[2]] + 2.0))
)

write_fixture(
  "negative_fnscale_gradient",
  initial_par = c(0.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(fnscale = -1.0),
  fn = function(p) -(p[[1]] - 3.0)^2,
  gr = function(p) c(-2.0 * (p[[1]] - 3.0))
)

write_fixture(
  "fnscale_parscale_gradient_2d",
  initial_par = c(0.0, 0.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(fnscale = -1.0, parscale = c(2.0, 0.5)),
  fn = function(p) -((p[[1]] - 3.0)^2 + 2.0 * (p[[2]] + 1.0)^2),
  gr = function(p) c(-2.0 * (p[[1]] - 3.0), -4.0 * (p[[2]] + 1.0))
)

write_fixture(
  "fnscale_parscale_finite_difference_2d",
  initial_par = c(0.0, 0.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(fnscale = -1.0, parscale = c(2.0, 0.5), ndeps = c(1e-4, 2e-4)),
  fn = function(p) -((p[[1]] - 3.0)^2 + 2.0 * (p[[2]] + 1.0)^2)
)

write_fixture(
  "two_dim_parscale_gradient",
  initial_par = c(0.0, 0.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(parscale = c(2.0, 0.5)),
  fn = function(p) (p[[1]] - 4.0)^2 + (p[[2]] + 1.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 4.0), 2.0 * (p[[2]] + 1.0))
)

write_fixture(
  "ndeps_vector",
  initial_par = c(4.0, -5.0),
  lower = c(-10.0, -10.0),
  upper = c(10.0, 10.0),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) (p[[1]] - 1.0)^2 + (p[[2]] + 2.0)^2
)
