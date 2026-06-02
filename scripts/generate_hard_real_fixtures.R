#!/usr/bin/env Rscript

`%||%` <- function(left, right) {
  if (is.null(left)) right else left
}

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
script_path <- if (is.na(script_arg)) {
  file.path("scripts", "generate_hard_real_fixtures.R")
} else {
  sub("^--file=", "", script_arg)
}
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
data_root <- file.path(repo_root, "data", "lbfgsb_hard_real_2026-06-01")
out_dir <- file.path(repo_root, "fixtures", "deseq_hard_real_subset")
cases_per_contrast <- as.integer(Sys.getenv("HARD_REAL_CASES_PER_CONTRAST", unset = "6"))

if (!dir.exists(data_root)) {
  stop("missing ignored hard-real bundle: ", data_root)
}
if (is.na(cases_per_contrast) || cases_per_contrast < 1L) {
  stop("HARD_REAL_CASES_PER_CONTRAST must be a positive integer")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_tsv <- function(path) {
  read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
}

as_rows <- function(matrix) {
  lapply(seq_len(nrow(matrix)), function(index) unname(as.numeric(matrix[index, ])))
}

sanitize <- function(value) {
  value <- gsub("[^A-Za-z0-9]+", "_", value)
  value <- gsub("^_+|_+$", "", value)
  tolower(value)
}

nb_nll_without_constants <- function(beta, y, dispersion, x, size_factors) {
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

global_hardest <- read_tsv(file.path(data_root, "global_hardest_512.tsv"))
completed <- read_tsv(file.path(data_root, "completed_contrasts.tsv"))

selected_rows <- do.call(rbind, lapply(completed$contrast, function(contrast) {
  contrast_rows <- global_hardest[global_hardest$contrast == contrast, , drop = FALSE]
  selected <- list()
  selected_keys <- character()
  for (case_kind in c("actual_or_rough_optimizer_row", "force_optimizer_probe")) {
    kind_rows <- contrast_rows[contrast_rows$caseKind == case_kind, , drop = FALSE]
    if (nrow(kind_rows) > 0) {
      selected[[length(selected) + 1L]] <- kind_rows[1L, , drop = FALSE]
      selected_keys <- c(selected_keys, kind_rows$gene[[1L]])
    }
  }
  if (length(selected) < cases_per_contrast) {
    for (row_index in seq_len(nrow(contrast_rows))) {
      key <- contrast_rows$gene[[row_index]]
      if (!(key %in% selected_keys)) {
        selected[[length(selected) + 1L]] <- contrast_rows[row_index, , drop = FALSE]
        selected_keys <- c(selected_keys, key)
      }
      if (length(selected) >= cases_per_contrast) {
        break
      }
    }
  }
  do.call(rbind, selected)
}))
row.names(selected_rows) <- NULL

contrasts <- unname(lapply(split(selected_rows, selected_rows$contrast), function(rows) {
  contrast <- rows$contrast[[1L]]
  base <- file.path(data_root, "contrasts", contrast, "lbfgsb")
  design <- read_tsv(file.path(base, "design_matrix.tsv"))
  counts <- read_tsv(file.path(base, "selected_counts.tsv"))
  size_factors <- read_tsv(file.path(base, "size_factors.tsv"))
  dispersions <- read_tsv(file.path(base, "selected_dispersions.tsv"))
  coefficients <- read_tsv(file.path(base, "selected_coefficients_long.tsv"))

  x <- as.matrix(design[, -1, drop = FALSE])
  storage.mode(x) <- "double"
  size_factor_values <- as.numeric(size_factors$sizeFactor)
  dimension <- ncol(x)
  control <- list(
    maxit = 100,
    fnscale = 1.0,
    parscale = I(rep(1.0, dimension)),
    ndeps = I(rep(1e-3, dimension)),
    factr = 1e7,
    pgtol = 0.0,
    lmm = 5
  )

  cases <- lapply(seq_len(nrow(rows)), function(row_index) {
    row <- rows[row_index, , drop = FALSE]
    gene <- row$gene[[1L]]
    gene_counts <- counts[counts$gene == gene, , drop = FALSE]
    gene_dispersion <- dispersions[dispersions$gene == gene, , drop = FALSE]
    gene_coefficients <- coefficients[coefficients$gene == gene, , drop = FALSE]
    gene_coefficients <- gene_coefficients[order(gene_coefficients$coefficient_index_1based), ]

    y <- as.numeric(gene_counts[, -1])
    dispersion <- as.numeric(gene_dispersion$dispersion)
    initial_par <- as.numeric(gene_coefficients$betaNoOptim)
    force_optim <- as.numeric(gene_coefficients$betaForceOptim)
    lower <- rep(-30.0, dimension)
    upper <- rep(30.0, dimension)

    result <- optim(
      par = initial_par,
      fn = nb_nll_without_constants,
      method = "L-BFGS-B",
      lower = lower,
      upper = upper,
      control = list(maxit = 100),
      y = y,
      dispersion = dispersion,
      x = x,
      size_factors = size_factor_values
    )
    counts_list <- as.list(unname(result$counts))
    names(counts_list) <- c("function", "gradient")

    list(
      fixture = paste("deseq_hard", sanitize(contrast), sanitize(gene), sep = "_"),
      gene = gene,
      case_kind = row$caseKind[[1L]],
      actual_optim_routed = as.logical(row$actualOptimRouted[[1L]]),
      hard_score = as.numeric(row$hardScore[[1L]]),
      dispersion = dispersion,
      counts = I(unname(y)),
      initial_par = I(unname(initial_par)),
      lower = I(unname(lower)),
      upper = I(unname(upper)),
      control = control,
      gradient_supplied = FALSE,
      result = list(
        par = I(unname(result$par)),
        value = unname(result$value),
        counts = counts_list,
        convergence = unname(result$convergence),
        message = result$message %||% ""
      ),
      deseq2_reference = list(
        beta_force_optim = I(unname(force_optim)),
        force_optim_conv = as.logical(row$forceOptimConv[[1L]]),
        force_optim_iter = as.integer(row$forceOptimIter[[1L]])
      )
    )
  })

  list(
    contrast = contrast,
    samples = nrow(x),
    coefficients = dimension,
    coefficient_names = I(colnames(x)),
    design = I(as_rows(x)),
    size_factors = I(unname(size_factor_values)),
    cases = cases
  )
}))

fixture <- list(
  fixture = "deseq_hard_real_2026_06_01_subset",
  source = list(
    source_directory = "data/lbfgsb_hard_real_2026-06-01/",
    committed_directory = "fixtures/deseq_hard_real_subset/",
    selection = paste0("For each completed contrast, ", cases_per_contrast, " unique-gene hard rows are selected from global_hardest_512.tsv, preferring the highest-ranked actual optimizer row and highest-ranked force optimizer probe when available, then filling from the ranked list."),
    note = "The committed data contains only portable numeric inputs needed to replay objective-only R optim() on the NB GLM objective. This matches DESeq2's fallback optimizer shape and does not include DESeq2 source code."
  ),
  objective = "negative binomial GLM negative log likelihood without beta-independent constants; DESeq2 coefficients are log2-scale, so eta = log(2) * X beta + log(sizeFactor)",
  contrasts = contrasts,
  r_version = R.version.string,
  platform = R.version$platform
)

json <- jsonlite::toJSON(fixture, pretty = TRUE, auto_unbox = TRUE, digits = 17)
writeLines(json, file.path(out_dir, "optim_cases.json"))
