# Supplied-gradient GLMs

`optim_cases.json` holds 68 negative-binomial models with starts, dispersions,
bounds, controls and R 4.6.2 results. TSV tables independently cross-check
counts, sample order, design, size factors, dispersions and starting coefficients.

```sh
cargo test --test deseq_real_subset
```

Every run checks all 68 cases. Apple Silicon full replay requires exact
bookkeeping and `1e-8` parameter/value limits; other hosts run scientific accuracy
checks. [Contract definitions](../../docs/validation.md)

Regenerate this and the objective-only cohort from committed inputs with
R 4.6.2 and jsonlite:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/regenerate_scientific_fixtures.R
```
