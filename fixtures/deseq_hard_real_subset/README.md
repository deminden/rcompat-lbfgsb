# DESeq2 Hard Real Optimizer Subset

This directory contains a compact subset generated from the ignored
`data/lbfgsb_hard_real_2026-06-01/` bundle.

The source bundle was originally produced with R 4.4.3 and DESeq2 1.46.0. The
committed optimizer outputs were replayed on 2026-07-22 with the current
reference stack: R 4.6.1, Bioconductor 3.23, and DESeq2 1.52.0.

The fixture does not depend on DESeq2 at test time and does not copy DESeq2 or
R source. It stores portable numeric inputs and exact outputs from a direct
objective-only `stats::optim(method = "L-BFGS-B")` replay of an independently
written negative-binomial GLM objective. This matches DESeq2's fallback
optimizer shape.

The Rust regression test requires tight endpoint and objective agreement for
all rows. All 48 rows reproduce R's
parameters, values, optimizer-level counts, convergence status, and message
bit-for-bit.

Selection is intentionally small: for each completed contrast, the generator
keeps six unique-gene hard rows, preferring the highest-ranked actual optimizer
row and the highest-ranked forced optimizer probe from `global_hardest_512.tsv`,
then filling from the ranked list.

After activating the pinned R 4.6.1 environment, regenerate the default six
cases per contrast with:

```sh
Rscript scripts/generate_hard_real_fixtures.R
```

Set `HARD_REAL_CASES_PER_CONTRAST` to change the compact subset size. The
separate `scripts/trace_hard_real_case.R` helper supports local path-drift
investigation against the ignored source bundle.
