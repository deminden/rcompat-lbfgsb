# Native R 4.6.2 / DESeq2 reference

The primary development reference is the separate Conda environment
`deseq2-r462-native`. R 4.6.2 and DESeq2 1.52.0 were built from source on the
Apple Silicon host with `-O3 -mcpu=native -fPIC`. OpenBLAS 0.3.34 is restricted to one
thread. `environment.json` records the source URL/SHA256, compiler versions,
CPU, flags, recipes, session and verification fit.

The original `deseq2-r461` environment remains usable and unchanged. The new
environment started as an independent `conda --copy` clone. Its source-built
R runtime replaces the bootstrap R package: `R --version` and the recorded
runtime fingerprint are authoritative; Conda's bootstrap package record still
names R 4.6.1. Copied package binaries had embedded paths to the old environment;
`repair_reference_links.py` relocates these paths in independent copies and
re-signs them. It refuses to edit a file sharing the original file's inode.

A verified 16-gene public `nbinomWaldTest` fit exercised all 16 optimizer
fallbacks and returned finite coefficients. These files then captured public
package calls on the existing committed inputs:

- `hard_real_reference.json`: 48 fits, 46 routed through `stats::optim`.
- `airway_reference.json`: 28 fits, all routed through `stats::optim`.

The native Rust release build preserves the measured worst coefficient errors:
`0.0013987208189476519` on the historical 46 and `3.7626286126801745e-6` on airway.
The previous callback is rerun against this new reference for the accuracy
comparison; old R 4.6.1 error numbers are not substituted as the new baseline.
Counts/messages are recorded observations in these scientific accuracy checks;
they are distinct from exact optimizer-result contracts. Both the old and new
reference accuracy gates remain active.

Activate and check the installed environment:

```sh
conda activate deseq2-r462-native
Rscript --vanilla -e 'cat(R.version.string,"\n"); packageVersion("DESeq2")'
```

Reproduce the native build on macOS ARM from an existing R 4.6.1 dependency env:

```sh
conda create --name deseq2-r462-native --clone deseq2-r461 --copy -y
conda install --name deseq2-r462-native -c conda-forge liblzma-devel pkg-config -y
bash scripts/build_reference_r462_native.sh \
  /Users/dendem/miniforge3/envs/deseq2-r462-native \
  /Users/dendem/.cache/rcompat-reference/r462-native \
  /Users/dendem/miniforge3/envs/deseq2-r461
```

The paths are the actual verified installation; substitute your own separate
prefixes when recreating it. The source build is intended for a headless
numerical reference. Its R source cache stays outside the repository. No R
optimizer source is read or used as implementation input, and the library and
normal verification commands have no R runtime dependency.

Capture or verify from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/baseline_deseq2_r461.R \
  fixtures/deseq_hard_real_subset/optim_cases.json fixtures/native_r462/hard_real_reference.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DESEQ2_BASELINE_MAXIT=1 Rscript --vanilla \
  scripts/baseline_deseq2_r461.R fixtures/deseq_airway/cases.json fixtures/native_r462/airway_reference.json
python3 scripts/verify_oracles.py
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native \
  cargo test --release --test native_reference_precision
```

Despite its historical filename, `baseline_deseq2_r461.R` records the actual
running R version. It now records input, recipe and serializer SHA256 values,
compiler flags, session and BLAS fingerprint, and checks binary64 round trips
before replacing outputs. Intentional regeneration requires reviewing and
updating frozen payload fingerprints.
