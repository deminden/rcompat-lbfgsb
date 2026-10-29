# Mixed-bound reference models

37 public R 4.6.2 calls cover quadratic, linear and plateau objectives;
finite/mixed/infinite bounds; both gradient forms; shallow curvature; zero
budgets and unrepresentable targets.

```sh
cargo test --test mixed_bound_oracles
```

Replay selects `AppleSiliconOpenBlas`, requires exact bookkeeping and checks
parameter/value error within `1e-8`/`1e-10`. Initial callback prefixes check
stencils and normalization within `1e-12`; four plateau cases also require
exact endpoint bits and initial-call counts.

Regenerate from the repository root with R 4.6.2, jsonlite and `shasum`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_mixed_bound_fixtures.R
```

[Contracts and integrity checks](../../docs/validation.md) ·
[Provenance](../../docs/provenance.md)
