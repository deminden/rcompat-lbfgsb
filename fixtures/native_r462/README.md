# DESeq2 package references

Public `DESeq2::nbinomWaldTest` captures use R 4.6.2 / DESeq2 1.52.0 on Apple
Silicon, native compiler flags and single-threaded OpenBLAS 0.3.34.
`environment.json` and the [runtime snapshot](../reference_environments/r462-macos-arm-native.json)
record the capture environment.

| File | Fits | Optimizer-routed | Inputs |
| --- | ---: | ---: | --- |
| `hard_real_reference.json` | 48 | 46 | `../deseq_hard_real_subset/optim_cases.json` |
| `airway_reference.json` | 28 | 28 | `../deseq_airway/cases.json` |

Results include coefficients, convergence, deviance, optimizer calls/results and
recipe/input fingerprints. Replay requires no R installation:

```sh
cargo test --test native_reference_precision
python3 scripts/verify_oracles.py
```

[Precision limits and platform scope](../../docs/validation.md)

To regenerate the hard package capture, install R 4.6.2, DESeq2 and jsonlite,
then run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla \
  scripts/capture_deseq2_reference.R \
  fixtures/deseq_hard_real_subset/optim_cases.json \
  fixtures/native_r462/hard_real_reference.json
```

Use the [airway generator](../deseq_airway/README.md) for its paired input/reference
capture. Review new outputs before updating frozen fingerprints.
