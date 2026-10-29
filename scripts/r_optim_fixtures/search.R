write_fixture(
  "rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  )
)

write_fixture(
  "rosenbrock_finite_difference_box",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(ndeps = c(1e-4, 2e-4)),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2
)

write_fixture(
  "line_search_refresh_sin_quad",
  initial_par = c(-0.60003378661349416, -1.04788708873093128),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(maxit = 80),
  fn = function(p) sum(0.02 * p^2 + sin(15.0 * p) + 0.3 * cos(7.0 * sum(p)))
)

write_trace_fixture(
  "rosenbrock_finite_difference_box_trace",
  source_fixture = "rosenbrock_finite_difference_box",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(ndeps = c(1e-4, 2e-4), factr = 1e7, pgtol = 0.0, lmm = 5, maxit = 100),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = NULL,
  prefix_len = 24
)

write_trace_fixture(
  "rosenbrock_default_trace",
  source_fixture = "rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(factr = 1e7, pgtol = 0.0, lmm = 5, maxit = 100),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  ),
  prefix_len = 16
)

write_fixture(
  "factr_loose_rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(factr = 1e12),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  )
)

write_fixture(
  "lmm_one_rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(lmm = 1),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  )
)

write_fixture(
  "lmm_two_rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(lmm = 2),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  )
)

write_fixture(
  "lmm_ten_rosenbrock",
  initial_par = c(-1.2, 1.0),
  lower = c(-5.0, -5.0),
  upper = c(5.0, 5.0),
  control = list(lmm = 10),
  fn = function(p) 100.0 * (p[[2]] - p[[1]] * p[[1]])^2 + (1.0 - p[[1]])^2,
  gr = function(p) c(
    -400.0 * p[[1]] * (p[[2]] - p[[1]] * p[[1]]) - 2.0 * (1.0 - p[[1]]),
    200.0 * (p[[2]] - p[[1]] * p[[1]])
  )
)
