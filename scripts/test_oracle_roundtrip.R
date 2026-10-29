#!/usr/bin/env Rscript
# Independent boundary and corruption checks for oracle serialization.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[[1]])
source(file.path(dirname(script), "oracle_json.R"))
temporary <- tempfile("rcompat-json-test-")
dir.create(temporary)

run_checks <- function() {
  on.exit(unlink(temporary, recursive=TRUE))
  # Construct at runtime: bytecode constant pooling may merge literal +/-0.
  negative_zero <- -numeric(1)
  boundary <- c(0, negative_zero, 2^-1074, -2^-1074, .Machine$double.xmin,
                -.Machine$double.xmin, .Machine$double.xmax, -.Machine$double.xmax,
                1, 1+.Machine$double.eps, 1-.Machine$double.eps/2, Inf, -Inf)
  stopifnot(1/boundary[[1]] == Inf, 1/boundary[[2]] == -Inf)
  set.seed(20261029)
  sampled <- runif(2000, min=-1, max=1) * 2^sample(-1073:1023, 2000, replace=TRUE)
  original <- list(
    boundaries=I(boundary), sampled=I(sampled),
    mixed_bounds=I(c(-Inf, negative_zero, 1+.Machine$double.eps, Inf)),
    nested=list(par=I(c(1)), value=negative_zero, counts=list(functions=1, gradients=2)),
    text="quotes: \"; slash: \\; newline: \n; unicode: \u03bb",
    flag=TRUE, absent=NULL
  )
  path <- file.path(temporary, "roundtrip.json")
  write_verified_oracle_json(original, path)
  parsed <- jsonlite::read_json(path, simplifyVector=FALSE)
  verify_oracle_roundtrip(original, parsed)
  stopifnot(identical(parsed$text, original$text), identical(parsed$flag, TRUE), is.null(parsed$absent))
  expect_failure <- function(label, expression) {
    failure <- tryCatch({force(expression); NULL}, error=function(e)e)
    if (!inherits(failure, "error")) stop("expected rejection: ", label)
  }
  altered <- parsed
  altered$boundaries[[2]] <- 0.0
  expect_failure("signed zero", verify_oracle_roundtrip(original, altered))
  altered <- parsed
  altered$boundaries[[10]] <- 1.0
  expect_failure("one ULP", verify_oracle_roundtrip(original, altered))
  altered <- parsed
  altered$sampled <- altered$sampled[-1]
  expect_failure("missing sample", verify_oracle_roundtrip(original, altered))
  # Failed generation must leave a previously verified artifact untouched.
  sentinel <- readBin(path, "raw", n=file.info(path)$size)
  expect_failure("NaN generation", write_verified_oracle_json(list(value=NaN), path))
  stopifnot(identical(sentinel, readBin(path, "raw", n=file.info(path)$size)))
  stopifnot(identical(list.files(temporary), "roundtrip.json"))
  cat("2013 binary64 boundaries/samples plus mixed arrays round-trip exactly; four corruption/atomicity checks pass\n")
}
run_checks()
