#!/usr/bin/env Rscript

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required to generate fixtures")
}

`%||%` <- function(left, right) {
  if (is.null(left)) right else left
}

args <- commandArgs(trailingOnly = FALSE)
file_arg <- "--file="
script_path <- sub(file_arg, "", args[grep(file_arg, args)][[1]])
root <- normalizePath(file.path(dirname(script_path), ".."), mustWork = TRUE)
out_dir <- file.path(root, "fixtures", "r_optim_lbfgsb")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_fixture <- function(name, initial_par, lower, upper, control, fn, gr = NULL) {
  result <- optim(
    par = initial_par,
    fn = fn,
    gr = gr,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = control
  )

  effective_control <- modifyList(
    list(
      maxit = 100,
      fnscale = 1.0,
      parscale = rep(1.0, length(initial_par)),
      ndeps = rep(1e-3, length(initial_par)),
      factr = 1e7,
      pgtol = 0.0,
      lmm = 5
    ),
    control
  )
  effective_control$parscale <- I(unname(effective_control$parscale))
  effective_control$ndeps <- I(unname(effective_control$ndeps))

  fixture <- list(
    fixture = name,
    initial_par = I(unname(initial_par)),
    lower = I(unname(lower)),
    upper = I(unname(upper)),
    control = effective_control,
    gradient_supplied = !is.null(gr),
    result = list(
      par = I(unname(result$par)),
      value = unname(result$value),
      counts = as.list(unname(result$counts)),
      convergence = unname(result$convergence),
      message = result$message %||% ""
    ),
    r_version = R.version.string,
    platform = R.version$platform
  )

  names(fixture$result$counts) <- c("function", "gradient")
  json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 17)
  writeLines(json, file.path(out_dir, paste0(name, ".json")))
}

write_error_fixture <- function(name, initial_par, lower, upper, control, fn, gr = NULL) {
  message <- tryCatch(
    {
      optim(
        par = initial_par,
        fn = fn,
        gr = gr,
        method = "L-BFGS-B",
        lower = lower,
        upper = upper,
        control = control
      )
      NULL
    },
    error = function(e) e$message
  )

  if (is.null(message)) {
    stop(paste("expected R error for fixture", name))
  }

  fixture <- list(
    fixture = name,
    initial_par = I(unname(initial_par)),
    lower = I(unname(lower)),
    upper = I(unname(upper)),
    control = control,
    gradient_supplied = !is.null(gr),
    error = list(message = message),
    r_version = R.version.string,
    platform = R.version$platform
  )

  json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 17)
  writeLines(json, file.path(out_dir, paste0(name, ".json")))
}

write_trace_fixture <- function(name, source_fixture, initial_par, lower, upper, control, fn, gr, prefix_len) {
  fn_calls <- list()
  traced_fn <- function(p) {
    fn_calls[[length(fn_calls) + 1]] <<- p
    fn(p)
  }

  result <- optim(
    par = initial_par,
    fn = traced_fn,
    gr = gr,
    method = "L-BFGS-B",
    lower = lower,
    upper = upper,
    control = control
  )

  fixture <- list(
    fixture = name,
    source_fixture = source_fixture,
    trace_kind = "function_evaluation_points",
    first_points = I(lapply(seq_len(min(prefix_len, length(fn_calls))), function(i) I(unname(fn_calls[[i]])))),
    result = list(
      counts = as.list(unname(result$counts)),
      convergence = unname(result$convergence),
      message = result$message %||% ""
    ),
    r_version = R.version.string,
    platform = R.version$platform
  )

  names(fixture$result$counts) <- c("function", "gradient")
  json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 17)
  writeLines(json, file.path(out_dir, paste0(name, ".json")))
}

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
  "factr_zero_quadratic",
  initial_par = c(10.0),
  lower = c(-Inf),
  upper = c(Inf),
  control = list(factr = 0.0),
  fn = function(p) (p[[1]] - 2.0)^2,
  gr = function(p) c(2.0 * (p[[1]] - 2.0))
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
  "active_bounds",
  initial_par = c(3.0),
  lower = c(0.0),
  upper = c(10.0),
  control = list(),
  fn = function(p) (p[[1]] + 2.0)^2
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
  "finite_difference",
  initial_par = c(6.0),
  lower = c(-10.0),
  upper = c(10.0),
  control = list(ndeps = c(1e-4)),
  fn = function(p) (p[[1]] + 1.0)^2
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

write_error_fixture(
  "fixed_no_gradient_error",
  initial_par = c(1.5),
  lower = c(1.5),
  upper = c(1.5),
  control = list(),
  fn = function(p) (p[[1]] - 10.0)^2
)

write_deseq_real_subset <- function() {
  data_dir <- file.path(root, "data")
  required <- file.path(
    data_dir,
    c(
      "selected_gene_cases.tsv",
      "selected_counts.tsv",
      "selected_coefficients_long.tsv",
      "selected_reference_results.tsv",
      "design_matrix.tsv",
      "size_factors.tsv",
      "sample_metadata.tsv"
    )
  )

  if (!all(file.exists(required))) {
    message("Skipping DESeq-derived fixtures because data/ is unavailable")
    return(invisible(NULL))
  }

  subset_dir <- file.path(root, "fixtures", "deseq_real_subset")
  dir.create(subset_dir, recursive = TRUE, showWarnings = FALSE)

  genes <- c(
    "MTND1P23",
    "IGKV1-27",
    "LINC02172",
    "CYP11B1",
    "LINC00370",
    "PPIAP86",
    "ENSG00000249901",
    "ENSG00000301330",
    "ENSG00000305966"
  )

  read_tsv <- function(name) {
    read.delim(file.path(data_dir, name), check.names = FALSE, stringsAsFactors = FALSE)
  }
  write_tsv <- function(data, name) {
    write.table(
      data,
      file.path(subset_dir, name),
      sep = "\t",
      quote = FALSE,
      row.names = FALSE
    )
  }

  gene_cases <- read_tsv("selected_gene_cases.tsv")
  counts <- read_tsv("selected_counts.tsv")
  coefficients <- read_tsv("selected_coefficients_long.tsv")
  reference_results <- read_tsv("selected_reference_results.tsv")
  design <- read_tsv("design_matrix.tsv")
  size_factors <- read_tsv("size_factors.tsv")
  sample_metadata <- read_tsv("sample_metadata.tsv")

  genes <- genes[genes %in% gene_cases$gene]
  if (length(genes) == 0) {
    stop("no DESeq-derived fixture genes were found in data/selected_gene_cases.tsv")
  }

  write_tsv(gene_cases[gene_cases$gene %in% genes, ], "gene_cases.tsv")
  write_tsv(counts[counts$gene %in% genes, ], "counts.tsv")
  write_tsv(coefficients[coefficients$gene %in% genes, ], "coefficients_long.tsv")
  write_tsv(reference_results[reference_results$gene %in% genes, ], "reference_results.tsv")
  write_tsv(design, "design_matrix.tsv")
  write_tsv(size_factors, "size_factors.tsv")
  write_tsv(sample_metadata, "sample_metadata.tsv")

  if (!identical(colnames(counts)[-1], size_factors$sample)) {
    stop("count columns and size factor samples are not aligned")
  }
  if (nrow(design) != nrow(size_factors)) {
    stop("design rows and size factors have different lengths")
  }

  x <- as.matrix(design[, -1, drop = FALSE])
  storage.mode(x) <- "double"
  size_factor_values <- as.numeric(size_factors$sizeFactor)
  dimension <- ncol(x)

  nb_nll_without_constants <- function(beta, y, dispersion) {
    size <- 1.0 / dispersion
    eta <- as.vector(x %*% beta) + log(size_factor_values)
    mu <- exp(eta)
    sum((y + size) * log(size + mu) - size * log(size) - y * log(mu))
  }

  nb_nll_gradient <- function(beta, y, dispersion) {
    size <- 1.0 / dispersion
    eta <- as.vector(x %*% beta) + log(size_factor_values)
    mu <- exp(eta)
    residual <- (y + size) * mu / (size + mu) - y
    as.vector(crossprod(x, residual))
  }

  case_control <- list(
    maxit = 100,
    fnscale = 1.0,
    parscale = I(rep(1.0, dimension)),
    ndeps = I(rep(1e-3, dimension)),
    factr = 1e7,
    pgtol = 0.0,
    lmm = 5
  )

  cases <- lapply(genes, function(gene) {
    gene_case <- gene_cases[gene_cases$gene == gene, , drop = FALSE][1, ]
    gene_counts <- counts[counts$gene == gene, , drop = FALSE][1, ]
    gene_coefficients <- coefficients[coefficients$gene == gene, , drop = FALSE]
    gene_coefficients <- gene_coefficients[order(gene_coefficients$coefficient_index_1based), ]

    y <- as.numeric(gene_counts[, -1])
    dispersion <- as.numeric(gene_case$dispersion)
    initial_par <- as.numeric(gene_coefficients$betaNoOptim)
    lower <- rep(-30.0, dimension)
    upper <- rep(30.0, dimension)

    result <- optim(
      par = initial_par,
      fn = nb_nll_without_constants,
      gr = nb_nll_gradient,
      method = "L-BFGS-B",
      lower = lower,
      upper = upper,
      control = list(maxit = 100),
      y = y,
      dispersion = dispersion
    )

    counts_list <- as.list(unname(result$counts))
    names(counts_list) <- c("function", "gradient")

    list(
      fixture = paste0("deseq_nb_", tolower(gsub("[^A-Za-z0-9]+", "_", gene))),
      gene = gene,
      reason = as.character(gene_case$caseKind),
      dispersion = dispersion,
      initial_par = I(unname(initial_par)),
      lower = I(unname(lower)),
      upper = I(unname(upper)),
      control = case_control,
      gradient_supplied = TRUE,
      result = list(
        par = I(unname(result$par)),
        value = unname(result$value),
        counts = counts_list,
        convergence = unname(result$convergence),
        message = result$message %||% ""
      )
    )
  })

  fixture <- list(
    fixture = "deseq_nb_real_subset",
    source = list(
      source_directory = "data/",
      committed_directory = "fixtures/deseq_real_subset/",
      note = "Small DESeq-derived real-data subset; optimizer fixture uses an independently written NB GLM objective without DESeq2 code."
    ),
    objective = "negative binomial GLM negative log likelihood without beta-independent constants",
    samples = nrow(design),
    coefficients = dimension,
    cases = cases,
    r_version = R.version.string,
    platform = R.version$platform
  )

  json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 17)
  writeLines(json, file.path(subset_dir, "optim_cases.json"))
}

write_deseq_real_subset()
