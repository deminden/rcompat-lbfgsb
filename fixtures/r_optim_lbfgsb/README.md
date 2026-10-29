# Standard R optimizer fixtures

63 R 4.6.2 input/trace artifacts cover bounds, fixed/empty problems, both
gradient forms, scaling, controls, counts/messages and callback order.
`manifest.json` fingerprints the payloads and split generation recipes.

```sh
cargo test --test parity_fixtures
```

Polynomial contracts run on every host. The sinusoidal full-result replay
requires recorded Apple Silicon callback math; other hosts run its scientific
accuracy check. See [validation](../../docs/validation.md) for contract definitions.

Regenerate from the repository root with R 4.6.2 and jsonlite:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_r_fixtures.R
```

The entry script loads case definitions from `scripts/r_optim_fixtures/`.
[Provenance and regeneration policy](../../docs/provenance.md)
