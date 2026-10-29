# Synthetic stress references

512 self-contained R 4.6.2 cases combine correlated/ill-conditioned quadratics,
coupled NB models and double wells across dimensions 1/2/4/9/10/11/13/32,
four bound routes, two starts, both gradient forms, scaling and maximization.

```sh
cargo test --test stress_oracles
```

Full replay on Apple Silicon requires `1e-8` parameter/value limits and exact
bookkeeping. All hosts also check 384 polynomial full results and the 128 NB
known optima under both profiles. The [active-set witness](../active_set/case_trace.json)
checks exact final results, 64 base calls and a digest of 1,472 raw calls.
[Accuracy limits and contracts](../../docs/validation.md)

`LBFGSB_STRESS_CASE` selects one case ID; unknown IDs fail.
`LBFGSB_STRESS_ORACLE` selects an alternate payload for corruption checks.
To run only full replay:

```sh
cargo test --test stress_oracles stress_corpus_matches_r -- --exact --include-ignored
```

Regenerate from the repository root with R 4.6.2 and jsonlite:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_stress_fixtures.R
```

The exact witness is extracted with `capture_reference_trace.py` and
`extract_active_set_trace.py` from public console/BLAS observations.
[Probe boundary and integrity policy](../../docs/provenance.md)
