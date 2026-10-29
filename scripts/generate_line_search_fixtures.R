#!/usr/bin/env Rscript
# Public optim calls only. No optimizer implementation source is an input.
args <- commandArgs(TRUE)
out <- if (length(args)) args[[1]] else "fixtures/line_search/cases.json"
stopifnot(requireNamespace("jsonlite", quietly = TRUE))
if (Sys.getenv("OPENBLAS_NUM_THREADS") != "1") stop("Start Rscript with OPENBLAS_NUM_THREADS=1")
cases <- list()
for (scenario in c("wrong_gradient", "absolute_warning", "history_exhaustion")) {
  for (finite in c(FALSE, TRUE)) {
    calls <- list()
    fn <- function(x) {
      calls[[length(calls) + 1L]] <<- I(unname(x))
      if (scenario == "absolute_warning") abs(x[[1]] - 0.12345) else x[[1]] * x[[1]]
    }
    gr <- function(x) {
      if (scenario == "wrong_gradient") return(-2 * x)
      if (scenario == "absolute_warning") return(if (x[[1]] < 0.12345) -1 else 1)
      if (x[[1]] > 1.5) 2 * x else -x
    }
    initial <- switch(scenario, wrong_gradient = 1, absolute_warning = 0, history_exhaustion = 2)
    lower <- if (finite) -5 else -Inf
    upper <- if (finite) 5 else Inf
    control <- list(maxit = if (scenario == "absolute_warning") 0L else 100L,
                    factr = 0, pgtol = 0, lmm = 5)
    result <- optim(initial, fn, gr, method = "L-BFGS-B", lower = lower, upper = upper, control = control)
    cases[[length(cases) + 1L]] <- list(
      id = paste(scenario, finite, sep = ":"), scenario = scenario, finite = finite,
      initial = I(initial), lower = I(lower), upper = I(upper), control = control,
      result = list(par = I(unname(result$par)), value = result$value, counts = as.list(result$counts),
                    convergence = result$convergence, message = result$message), calls = calls)
  }
}
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1]])
script_hash <- strsplit(system2("shasum", c("-a", "256", shQuote(script_path)), stdout = TRUE), "[[:space:]]+")[[1]][[1]]
manifest <- list(generator = "scripts/generate_line_search_fixtures.R", generator_sha256 = script_hash,
                 r_version = R.version.string, platform = R.version$platform,
                 blas = unname(extSoftVersion()[["BLAS"]]), threads = Sys.getenv("OPENBLAS_NUM_THREADS"),
                 session = capture.output(sessionInfo()), cases = cases)
verify_roundtrip <- function(original, parsed) {
  if (is.numeric(original)) {
    x <- as.double(original); y <- as.double(unlist(parsed))
    stopifnot(identical(writeBin(x, raw(), size = 8, endian = "little"), writeBin(y, raw(), size = 8, endian = "little")))
  } else if (is.list(original)) {
    stopifnot(length(original) == length(parsed))
    for (i in seq_along(original)) verify_roundtrip(original[[i]], parsed[[i]])
  }
}
write_verified <- function() {
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(tmpdir = dirname(out)); on.exit(unlink(temporary))
  jsonlite::write_json(manifest, temporary, auto_unbox = TRUE, pretty = TRUE, digits = 17)
  verify_roundtrip(manifest, jsonlite::read_json(temporary, simplifyVector = FALSE))
  stopifnot(file.rename(temporary, out))
}
write_verified()
cat(length(cases), "cases written to", out, "\n")
