# Objective-only GLMs

`optim_cases.json` holds 48 negative-binomial models across eight contrasts,
including every input needed for replay. The R 4.6.2 objective uses log2
coefficients and omits beta-independent likelihood constants.

```sh
cargo test --test deseq_hard_real_subset
```

Every run checks all 48 cases with `5e-3` parameter and `1e-5` objective limits.
Apple Silicon full replay also requires exact counts, convergence and messages;
portable scientific checks require matching convergence.
[Contract definitions](../../docs/validation.md)

[`regenerate_scientific_fixtures.R`](../../scripts/regenerate_scientific_fixtures.R)
regenerates both GLM cohorts from committed inputs; see the
[supplied-gradient guide](../deseq_real_subset/README.md) for the command.

Separate [DESeq2 package captures](../native_r462/README.md) contain 46
optimizer-routed fits and two unrouted probes. Their full likelihood/prior
callback has its own precision limits; it is distinct from this scalar objective.
