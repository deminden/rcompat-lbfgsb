# Published airway counts

`cases.json` contains 28 real DESeq2 fallback fits from 14 distinct gene-count
profiles in the [Bioconductor airway dataset](https://bioconductor.org/packages/airway/)
(airway 1.32.0; GEO GSE52778). The experiment has eight samples from four human
airway smooth-muscle cell lines treated with dexamethasone, published by
[Himes et al. (2014)](https://doi.org/10.1371/journal.pone.0099625).

The compact generator estimates size factors and dispersions with DESeq2,
then selects genes spanning expression levels, treatment imbalance, and
sparsity. It removes duplicate count profiles and exports paired (`~cell + dex`)
and unpaired (`~dex`) designs. Counts and sample factors are preserved from the
package; all inputs needed for replay are included in the JSON.

One IRLS iteration deliberately exercises the public `nbinomWaldTest` fallback.
The recorded `stats::optim` calls use their default controls, including
`maxit=100`, and have no supplied gradient. The fixture records the actual
starting coefficients, bounds, final coefficients, objective, counts, and
status under R 4.6.1 / DESeq2 1.52.0. Its `reference.platform` identifies the
oracle platform; these are additional scientific fixtures rather than the
existing pinned x86 optimizer fixtures.

Regenerate with the installed environment:

```sh
conda activate deseq2-r461
# Once, if needed: Rscript -e 'BiocManager::install("airway", ask=FALSE)'
Rscript scripts/generate_airway_fixtures.R
```

An optional first argument changes the output path. Generation uses the
public package API and shares the compact black-box capture script with the
existing hard-real baseline. Normal Rust tests require neither R nor network
access. With the independent full-NB/prior callback and `arm-fma` arithmetic,
the worst coefficient error falls from `0.818664` to `3.76263e-6` across all 28
fits, a 217,578x improvement reproduced in debug and release. The largest
observed per-fit error increase is `1.08e-12`.
Run `cargo test --features arm-fma --test deseq2_precision` to replay the comparison.
The separate fixture contract test validates the captured package outputs.
