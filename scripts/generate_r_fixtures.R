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
output_args <- commandArgs(trailingOnly = TRUE)
out_dir <- if(length(output_args)) output_args[[1]] else file.path(root, "fixtures", "r_optim_lbfgsb")
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

for (case_file in c("bounds.R", "controls.R", "search.R")) {
  source(file.path(root, "scripts", "r_optim_fixtures", case_file), local = TRUE)
}

# Keep the self-contained recipe and every generated payload independently pinned.
file_sha256 <- function(path) strsplit(system2("shasum", c("-a", "256", shQuote(path)), stdout=TRUE), "[[:space:]]+")[[1]][[1]]
payloads <- list.files(out_dir, pattern="\\.json$", full.names=TRUE)
payloads <- payloads[basename(payloads)!="manifest.json"]
stopifnot(length(payloads)==63L)
sources <- c("scripts/generate_r_fixtures.R", file.path("scripts", "r_optim_fixtures", c("bounds.R", "controls.R", "search.R")))
manifest <- list(r_version=R.version.string, platform=R.version$platform,
  sources=setNames(lapply(file.path(root,sources),file_sha256),sources),
  payloads=setNames(lapply(payloads,file_sha256),basename(payloads)))
jsonlite::write_json(manifest,file.path(out_dir,"manifest.json"),auto_unbox=TRUE,pretty=TRUE)
