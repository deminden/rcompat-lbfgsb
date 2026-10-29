# Installed oracle fingerprints

The two JSON snapshots record the installed R 4.6.1 and native R 4.6.2 reference
environments on Apple Silicon/macOS 27.2. They include runtime/package versions,
package build records, session, BLAS thread controls, configured compiler flags,
binary SHA256 hashes and all 98/100 bootstrap Conda package archive URLs/hashes.
Hardware descriptions use the Apple Silicon family label rather than a chip
model; binary hashes and numerical-library configurations retain their detail.
The capture reads public R interfaces, distribution metadata and binary bytes
for hashing. It does not inspect optimizer implementation source.

Capture a fresh snapshot to a new file:

```sh
python3 scripts/capture_reference_environment.py \
  --prefix /Users/dendem/miniforge3/envs/deseq2-r461 \
  --output /tmp/r461-runtime.json
python3 scripts/capture_reference_environment.py \
  --prefix /Users/dendem/miniforge3/envs/deseq2-r462-native \
  --output /tmp/r462-runtime.json
python3 scripts/verify_oracles.py
python3 scripts/test_verify_oracles.py
```

The verifier pins each retained snapshot and the capture recipe. Normal replay
does not require either installed environment. To reconstruct a bootstrap
distribution, export its `bootstrap_distribution` URLs with `#md5` fragments
after an `@EXPLICIT` header and pass that explicit file to `conda create --file`.
For R 4.6.2, subsequently follow `../native_r462/README.md`: the source-built R
and DESeq2 replace bootstrap binaries. Conda's r-base 4.6.1 record does not
describe the installed R 4.6.2 runtime. Paths, binary signatures and compiler
output can change when rebuilding or relocating; a fresh build is not promised
to reproduce every retained binary hash.

`R CMD config` and the compiler executable currently installed in an environment
do not establish which compiler originally built a precompiled Conda archive.
The distribution URL and checksum identify that archive; this distinction is
explicit in both snapshots. The original July Linux x86 environment is no longer
available here. Its recorded R/BLAS/Rust stack is historical metadata in
`../../PROVENANCE.md` and the old fixtures; an exact regeneration of that entire
environment is not claimed. The available 68 + 48 result contracts passed the
pinned Linux x86 debug/release CI jobs before the local version bump.

The retained R 4.6.1 runtime regenerated both current polynomial oracle artifacts
byte for byte during the 0.3.0 audit:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/dendem/miniforge3/envs/deseq2-r461/bin/Rscript --vanilla \
  scripts/generate_mixed_bound_fixtures.R /tmp/mixed_bounds.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/dendem/miniforge3/envs/deseq2-r461/bin/Rscript --vanilla \
  scripts/generate_line_search_fixtures.R /tmp/line_search.json
cmp fixtures/mixed_bounds/cases.json /tmp/mixed_bounds.json
cmp fixtures/line_search/cases.json /tmp/line_search.json
```

Their SHA256 values remain
`052c5bdeae0bd6ec1a18d96a55337c3ea23cd4daeae804408003b1b9e43e70c5`
and `a46c9cd154d6aa49f5610bfad8b186b31e73eb7e989f405eadeee357e5226a7e`.
