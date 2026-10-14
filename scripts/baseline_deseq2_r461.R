#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
script_path <- if (is.na(script_arg)) {
  file.path("scripts", "baseline_deseq2_r461.R")
} else {
  sub("^--file=", "", script_arg)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
fixture_path <- if (length(args) >= 1L) args[[1L]] else {
  file.path(repo_root, "fixtures", "deseq_hard_real_subset", "optim_cases.json")
}
out_path <- if (length(args) >= 2L) args[[2L]] else {
  file.path(repo_root, "target", "deseq2_r461_nbinomWaldTest.json")
}

fixture <- read_json(fixture_path, simplifyVector = FALSE)
maxit <- as.integer(Sys.getenv("DESEQ2_BASELINE_MAXIT", unset = "100"))
stopifnot(length(maxit) == 1L, !is.na(maxit), maxit >= 1L)
.capture_optim <- list()
.capture_optim_result <- NULL
trace(
  "optim",
  where = asNamespace("stats"),
  tracer = quote({
    .capture_optim <<- c(
      .capture_optim,
      list(list(
        par = par,
        fn = fn,
        dots = list(...),
        has_gradient = !missing(gr),
        method = method,
        lower = lower,
        upper = upper,
        control = control
      ))
    )
  }),
  exit = quote(.capture_optim_result <<- returnValue()),
  print = FALSE
)

fit_case <- function(contrast, case) {
  x <- do.call(rbind, contrast$design)
  storage.mode(x) <- "double"
  y <- matrix(as.integer(unlist(case$counts)), nrow = 1L)
  sample_names <- paste0("sample", seq_len(ncol(y)))
  colnames(y) <- sample_names
  rownames(y) <- case$gene
  rownames(x) <- sample_names
  colnames(x) <- unlist(contrast$coefficient_names)

  dds <- DESeqDataSetFromMatrix(
    countData = y,
    colData = S4Vectors::DataFrame(row.names = sample_names),
    design = ~1
  )
  sizeFactors(dds) <- unlist(contrast$size_factors)
  dispersions(dds) <- case$dispersion
  .capture_optim <<- list()
  .capture_optim_result <<- NULL
  fitted <- nbinomWaldTest(
    dds,
    modelMatrix = x,
    modelMatrixType = "standard",
    maxit = maxit,
    useOptim = TRUE,
    quiet = TRUE
  )
  if (length(.capture_optim) > 1L) {
    stop("expected at most one stats::optim call for ", case$fixture)
  }
  optimizer_call <- if (length(.capture_optim) == 1L) .capture_optim[[1L]] else NULL
  optimizer_result <- .capture_optim_result
  objective_offset_start <- NULL
  objective_offset_endpoint <- NULL
  if (!is.null(optimizer_call)) {
    y <- as.numeric(unlist(case$counts))
    dispersion <- as.numeric(case$dispersion)
    size_factors <- unlist(contrast$size_factors)
    standalone_nll <- function(beta) {
      mu <- exp(as.vector(x %*% (log(2) * beta)) + log(size_factors))
      -sum(stats::dnbinom(y, size = 1.0 / dispersion, mu = mu, log = TRUE))
    }
    callback_start <- do.call(
      optimizer_call$fn,
      c(list(optimizer_call$par), optimizer_call$dots)
    )
    callback_endpoint <- do.call(
      optimizer_call$fn,
      c(list(optimizer_result$par), optimizer_call$dots)
    )
    objective_offset_start <- callback_start - standalone_nll(optimizer_call$par)
    objective_offset_endpoint <- callback_endpoint - standalone_nll(optimizer_result$par)
  }
  beta_names <- sub("^\\(Intercept\\)$", "Intercept", unlist(contrast$coefficient_names))

  list(
    fixture = case$fixture,
    contrast = contrast$contrast,
    gene = case$gene,
    beta = as.numeric(unlist(
      mcols(fitted)[1L, beta_names],
      use.names = FALSE
    )),
    beta_converged = isTRUE(mcols(fitted)$betaConv[[1L]]),
    beta_iterations = as.integer(mcols(fitted)$betaIter[[1L]]),
    deviance = as.numeric(mcols(fitted)$deviance[[1L]]),
    optimizer_start = if (is.null(optimizer_call)) NULL else as.numeric(optimizer_call$par),
    optimizer_has_gradient = if (is.null(optimizer_call)) NULL else optimizer_call$has_gradient,
    optimizer_method = if (is.null(optimizer_call)) NULL else optimizer_call$method,
    optimizer_lower = if (is.null(optimizer_call)) NULL else as.numeric(optimizer_call$lower),
    optimizer_upper = if (is.null(optimizer_call)) NULL else as.numeric(optimizer_call$upper),
    optimizer_control = if (is.null(optimizer_call)) NULL else optimizer_call$control,
    objective_offset_start = objective_offset_start,
    objective_offset_endpoint = objective_offset_endpoint,
    optimizer_result = if (is.null(optimizer_result)) NULL else list(
      par = as.numeric(optimizer_result$par),
      value = optimizer_result$value,
      counts = optimizer_result$counts,
      convergence = optimizer_result$convergence,
      message = if (is.null(optimizer_result$message)) "" else optimizer_result$message
    )
  )
}

results <- unlist(lapply(fixture$contrasts, function(contrast) {
  lapply(contrast$cases, function(case) fit_case(contrast, case))
}), recursive = FALSE)
untrace("optim", where = asNamespace("stats"))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(
  list(
    fixture = fixture$fixture,
    r_version = R.version.string,
    platform = R.version$platform,
    bioconductor_version = as.character(BiocManager::version()),
    deseq2_version = as.character(packageVersion("DESeq2")),
    interface = "DESeq2::nbinomWaldTest(modelMatrix=..., useOptim=TRUE)",
    maxit = maxit,
    results = results
  ),
  path = out_path,
  pretty = TRUE,
  auto_unbox = TRUE,
  digits = 17L,
  null = "null",
  na = "null"
)
cat("Wrote ", length(results), " real DESeq2 fits to ", out_path, "\n", sep = "")
