#!/usr/bin/env Rscript
# Public counts: airway / GEO GSE52778 (Himes et al., 2014).
# Prerequisites: BiocManager::install(c("DESeq2", "airway")); jsonlite.
suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(airway))
suppressPackageStartupMessages(library(jsonlite))

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
root <- normalizePath(file.path(dirname(script), ".."))
source(file.path(root, "scripts", "oracle_json.R"))
file_sha256 <- function(path) strsplit(system2("shasum", c("-a", "256", shQuote(path)), stdout = TRUE), "[[:space:]]+")[[1]][[1]]
args <- commandArgs(TRUE)
out <- if (length(args)) args[[1]] else file.path(root, "fixtures", "deseq_airway", "cases.json")
reference_out <- if (length(args) >= 2L) args[[2]] else if (length(args)) file.path(dirname(out), "airway_reference.json") else file.path(root, "fixtures", "native_r462", "airway_reference.json")
data("airway", package = "airway")
dds <- DESeqDataSet(airway, design = ~cell + dex)
dds <- estimateSizeFactors(dds)
dds <- dds[rowSums(counts(dds)) >= 10, ]
normalized <- counts(dds, normalized = TRUE)
depth <- rowMeans(normalized)
imbalance <- abs(log2((rowMeans(normalized[, dds$dex == "trt"]) + 0.5) /
                     (rowMeans(normalized[, dds$dex == "untrt"]) + 0.5)))
# Span expression levels; include sparse and treatment-imbalanced genes.
ranked <- order(depth, rownames(dds))
selected <- unique(c(ranked[round(seq(1, length(ranked), length.out = 8))],
                     order(-imbalance, rownames(dds))[1:4],
                     order(-rowSums(counts(dds) == 0), depth, rownames(dds))[1:4]))
selected <- selected[!duplicated(as.data.frame(counts(dds)[selected, ]))]
rows <- function(x) lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))
contrasts <- lapply(list(paired = ~cell + dex, unpaired = ~dex), function(design) {
  design(dds) <- design
  fitted <- estimateDispersions(dds, quiet = TRUE)
  x <- model.matrix(design, as.data.frame(colData(dds)))
  list(contrast = deparse(design), samples = nrow(x), coefficients = ncol(x),
       coefficient_names = I(colnames(x)), design = rows(x),
       size_factors = I(sizeFactors(dds)), cases = lapply(selected, function(i) {
         list(fixture = paste(deparse(design), rownames(dds)[i], sep = ":"),
              gene = rownames(dds)[i], counts = I(as.numeric(counts(dds)[i, ])),
              dispersion = dispersions(fitted)[i], actual_optim_routed = TRUE,
              initial_par = I(rep(0, ncol(x))), lower = I(rep(-30, ncol(x))),
              upper = I(rep(30, ncol(x))))
       }))
})
inputs <- list(fixture = "airway_GSE52778", contrasts = unname(contrasts))
input_path <- tempfile(fileext = ".json")
reference_path <- tempfile(fileext = ".json")
write_json(inputs, input_path, auto_unbox = TRUE, digits = 17)
# One IRLS iteration deliberately exercises DESeq2's real fallback on unchanged
# published counts; stats::optim still receives its default maxit=100 controls.
status <- system2(file.path(R.home("bin"), "Rscript"),
                  c(shQuote(file.path(root, "scripts", "capture_deseq2_reference.R")),
                    shQuote(input_path), shQuote(reference_path)),
                  env = "DESEQ2_BASELINE_MAXIT=1")
stopifnot(status == 0)
reference <- read_json(reference_path, simplifyVector = FALSE)
for (j in seq_along(inputs$contrasts)) {
  for (i in seq_along(inputs$contrasts[[j]]$cases)) {
    case <- inputs$contrasts[[j]]$cases[[i]]
    fit <- Filter(function(f) f$fixture == case$fixture, reference$results)[[1]]
    case$actual_optim_routed <- !is.null(fit$optimizer_start)
    case$initial_par <- fit$optimizer_start
    inputs$contrasts[[j]]$cases[[i]] <- case
  }
}
inputs$source <- list(url = "https://bioconductor.org/packages/airway/",
                     accession = "GSE52778", doi = "10.1371/journal.pone.0099625",
                     package_version = as.character(packageVersion("airway")),
                     selection = "8 depth ranks, 4 largest treatment imbalances, 4 sparsest genes; deduplicated by count profile")
inputs$generator <- "scripts/generate_airway_fixtures.R"
inputs$generator_sha256 <- file_sha256(script)
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
write_verified_oracle_json(inputs, out)
# Capture against the final committed inputs so the reference pins their bytes.
status <- system2(file.path(R.home("bin"), "Rscript"),
                  c(shQuote(file.path(root, "scripts", "capture_deseq2_reference.R")),
                    shQuote(out), shQuote(reference_out)),
                  env = "DESEQ2_BASELINE_MAXIT=1")
stopifnot(status == 0)
final_reference <- read_json(reference_out, simplifyVector = FALSE)
stopifnot(identical(reference$results, final_reference$results))
unlink(c(input_path, reference_path))
cat("Wrote published airway fixtures to ", out, "\n", sep = "")
