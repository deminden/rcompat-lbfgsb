# Line-search recovery references

Six public R 4.6.2 calls pair finite/infinite bounds with three deliberate
recovery scenarios: reversed quadratic gradient, absolute-distance bracket
warning and inconsistent-gradient history exhaustion.

```sh
cargo test --test line_search_oracles
```

Replay uses `AppleSiliconOpenBlas` and checks every result/callback bit, memory
reset and retained best point. These cases test failure handling; their
inconsistent gradients are intentional.

Regenerate from the repository root with R 4.6.2 and jsonlite:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_line_search_fixtures.R
```

[Contracts and integrity checks](../../docs/validation.md) ·
[Provenance](../../docs/provenance.md)
