# Public airway inputs

`cases.json` contains 28 fallback-fit inputs from 14 distinct count profiles in
[airway 1.32.0](https://bioconductor.org/packages/airway/) / GEO GSE52778:
eight samples from four human airway cell lines, treated with dexamethasone
([Himes et al., 2014](https://doi.org/10.1371/journal.pone.0099625)).

Selection spans expression, treatment imbalance and sparsity. DESeq2 estimates
size factors/dispersions; paired (`~cell + dex`) and unpaired (`~dex`) designs
share the published counts. All replay inputs are committed.

The canonical package outputs are
[airway_reference.json](../native_r462/airway_reference.json). One DESeq2 IRLS
iteration exercises fallback; the captured `optim` calls retain their default
`maxit=100` and have no supplied gradient.

```sh
cargo test --test deseq_airway --test native_reference_precision
```

[Validation](../../docs/validation.md) defines precision limits and platform scope.
Regenerate inputs and package outputs with R 4.6.2, DESeq2, airway and jsonlite:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_airway_fixtures.R
```

The generator writes both canonical files and verifies the reference against
the final input-file fingerprint.
