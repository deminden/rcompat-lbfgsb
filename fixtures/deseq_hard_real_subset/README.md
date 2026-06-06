# DESeq2 hard real optimizer subset

This directory contains a compact subset generated from the ignored
`data/lbfgsb_hard_real_2026-06-01/` bundle.

The source bundle was produced with R 4.4.3 and DESeq2 1.46.0. Current local
DESeq2 parity work should use the conda reference environment
`/home/den/miniforge3/envs/rsfgsea-r460`, verified on 2026-06-06 with R 4.6.0,
Bioconductor 3.23, and DESeq2 1.52.0. The committed fixture does not depend on
DESeq2 at test time and does not copy DESeq2 source code. It stores only
portable numeric inputs and exact outputs from a direct objective-only
`stats::optim(method = "L-BFGS-B")` replay of the negative-binomial GLM
objective. This matches DESeq2's fallback optimizer shape.
The Rust regression test expects exact optimizer-level counts for most rows and
tight objective agreement for all rows; a few weakly identified rows are allowed
to differ more in individual coefficients when the objective remains equivalent.
At the current backend parity point, 36 of 48 rows match R's optimizer-level
counts exactly and the worst optimizer-count drift is 10 evaluations.

Selection is intentionally small: for each completed contrast, the generator
keeps six unique-gene hard rows, preferring the highest-ranked actual optimizer
row and the highest-ranked forced optimizer probe from `global_hardest_512.tsv`,
then filling from the ranked list.

Regenerate with the default six cases per contrast:

```sh
PATH=/home/den/miniforge3/envs/rsfgsea-r460/bin:$PATH \
  /home/den/miniforge3/envs/rsfgsea-r460/bin/Rscript scripts/generate_hard_real_fixtures.R
```

To change the compact subset size, set `HARD_REAL_CASES_PER_CONTRAST`.

For local path-drift investigation against the ignored source bundle:

```sh
PATH=/home/den/miniforge3/envs/rsfgsea-r460/bin:$PATH \
  HARD_REAL_OPTIM_TRACE=1 \
  /home/den/miniforge3/envs/rsfgsea-r460/bin/Rscript scripts/trace_hard_real_case.R pancreas_blocked_permutation_rep01 ADIPOQ 0
```

`HARD_REAL_OPTIM_TRACE` and `HARD_REAL_OPTIM_REPORT` are passed to R's
`optim()` control list; the positional third argument controls how many raw
objective-call points are printed (`0` means all).
