#!/usr/bin/env Rscript

`%||%` <- function(left, right) {
  if (is.null(left)) right else left
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L) {
  stop("usage: trace_hard_real_case.R <contrast> <gene> [max_printed_calls]")
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
script_path <- if (is.na(script_arg)) {
  file.path("scripts", "trace_hard_real_case.R")
} else {
  sub("^--file=", "", script_arg)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
data_root <- file.path(repo_root, "data", "lbfgsb_hard_real_2026-06-01")

contrast <- args[[1]]
gene <- args[[2]]
max_printed_calls <- if (length(args) >= 3L) {
  as.integer(args[[3]])
} else {
  as.integer(Sys.getenv("HARD_REAL_TRACE_CALLS", unset = "120"))
}
optim_trace <- as.integer(Sys.getenv("HARD_REAL_OPTIM_TRACE", unset = "0"))
optim_report <- as.integer(Sys.getenv("HARD_REAL_OPTIM_REPORT", unset = "1"))
if (is.na(max_printed_calls) || max_printed_calls < 0L) {
  stop("max_printed_calls must be a nonnegative integer")
}
if (is.na(optim_trace) || optim_trace < 0L) {
  stop("HARD_REAL_OPTIM_TRACE must be a nonnegative integer")
}
if (is.na(optim_report) || optim_report < 1L) {
  stop("HARD_REAL_OPTIM_REPORT must be a positive integer")
}

base <- file.path(data_root, "contrasts", contrast, "lbfgsb")
if (!dir.exists(base)) {
  stop("missing hard-real contrast directory: ", base)
}

read_tsv <- function(name) {
  read.delim(file.path(base, name), check.names = FALSE, stringsAsFactors = FALSE)
}

design <- read_tsv("design_matrix.tsv")
counts <- read_tsv("selected_counts.tsv")
size_factors <- read_tsv("size_factors.tsv")
dispersions <- read_tsv("selected_dispersions.tsv")
coefficients <- read_tsv("selected_coefficients_long.tsv")

x <- as.matrix(design[, -1, drop = FALSE])
storage.mode(x) <- "double"
y <- as.numeric(counts[counts$gene == gene, -1])
size_factor_values <- as.numeric(size_factors$sizeFactor)
dispersion <- as.numeric(dispersions$dispersion[dispersions$gene == gene])
gene_coefficients <- coefficients[coefficients$gene == gene, , drop = FALSE]
gene_coefficients <- gene_coefficients[order(gene_coefficients$coefficient_index_1based), ]

if (length(y) == 0L || length(dispersion) == 0L || nrow(gene_coefficients) == 0L) {
  stop("missing hard-real inputs for ", contrast, " / ", gene)
}

initial <- as.numeric(gene_coefficients$betaNoOptim)
lower <- rep(-30.0, length(initial))
upper <- rep(30.0, length(initial))
call_count <- 0L

nb_nll_without_constants <- function(beta, y, dispersion, x, size_factors) {
  call_count <<- call_count + 1L
  if (max_printed_calls == 0L || call_count <= max_printed_calls) {
    cat(
      "R_HARD_TRACE",
      contrast,
      gene,
      call_count,
      paste(formatC(beta, digits = 17, format = "e"), collapse = "\t"),
      sep = "\t"
    )
    cat("\n")
  }

  size <- 1.0 / dispersion
  beta_scale <- log(2)
  total <- 0.0
  for (i in seq_along(y)) {
    linear <- 0.0
    for (j in seq_along(beta)) {
      linear <- linear + x[i, j] * beta[j]
    }
    eta <- beta_scale * linear + log(size_factors[i])
    mu <- exp(eta)
    total <- total + (y[i] + size) * log(size + mu) - size * log(size) - y[i] * log(mu)
  }
  total
}

result <- optim(
  par = initial,
  fn = nb_nll_without_constants,
  method = "L-BFGS-B",
  lower = lower,
  upper = upper,
  control = list(maxit = 100, trace = optim_trace, REPORT = optim_report),
  y = y,
  dispersion = dispersion,
  x = x,
  size_factors = size_factor_values
)

cat("R_RESULT", paste(formatC(result$par, digits = 17, format = "e"), collapse = "\t"), sep = "\t")
cat("\n")
cat(
  "R_SUMMARY",
  contrast,
  gene,
  paste(result$counts, collapse = "/"),
  result$convergence,
  result$message %||% "",
  formatC(result$value, digits = 17, format = "e"),
  call_count,
  sep = "\t"
)
cat("\n")
