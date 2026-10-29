# Architecture

The compatibility layer exposes R-style semantics; the backend minimizes in
scaled coordinates. All nonempty problems use the same compact solver for
finite, mixed and infinite bounds. Scientific likelihoods live in examples.

## Module map

| Module | Responsibility |
| --- | --- |
| `src/compat.rs` | Entry points, validation, start projection and result conversion |
| `src/bounds.rs`, `src/control.rs` | Bounds and control contracts |
| `src/scaling.rs`, `src/objective.rs`, `src/finite_diff.rs` | Checked scaling, callback evaluation and bounded differences |
| `src/backend/native.rs`, `src/backend/native/iteration.rs` | Backend adapter, iteration, stopping and recovery |
| `src/backend/native/compact.rs`, `src/backend/native/compact/` | Retained history, Cauchy point, free partition, subspace solve and algebra |
| `src/backend/native/line_search.rs`, `src/backend/native/line_search/` | More–Thuente search, interpolation and step geometry |
| `src/profile.rs` | Per-run reference arithmetic |
| `src/diagnostics.rs` | Observer events and typed outcomes |

## Numerical invariants

- Callbacks use user units; the backend uses internal coordinates and values.
  Scaling and finite-difference operations reject non-finite results, collapsed
  bounds and unrepresentable stencils. Final conversion is checked before success.
- History/factor storage is `O(n*m + m*m)`; production never forms an `n*n` Hessian.
  Cyclic correction columns and solve buffers are retained between iterations.
- Unconstrained retained models solve directly in the full subspace, avoiding
  Cauchy-step cancellation. Constrained models maintain the active/free partition.
- Operation order is part of compatibility: YY cache updates use
  `(old + entering_sum) - leaving_sum`; Apple Silicon cubic trial updates use
  fused affine rounding. Algebraic rewrites require reference replay.
- Arithmetic profiles are explicit per run, with no global or thread-local state.
  They model optimizer rounding, not user callback math.

## Search and stopping

Initial trial normalization, geometric feasibility and the `1e10` search limit
are separate. More–Thuente accepts strong-Wolfe, step-limit or bracket-warning
outcomes. A failed search with history can clear memory and retry once; final
failure preserves the last accepted point/value.

Stopping uses the bound-distance-capped projected gradient and signed reduction
`(old_f-new_f)/max(abs(old_f),abs(new_f),1)`. Tiny or blocked directions alone do
not establish stationarity. Controls are honored without dataset- or
dimension-specific overrides. See [usage](usage.md) for zero-tolerance semantics.

## Diagnostics and tests

`ScaledProblem` owns the observer. Events use the actual production model and
add no evaluations; disabled observers remove diagnostic-only work at compile
time. Initial stationarity returns after evaluation and `Started`, before
allocating unused history. Public `Finished` follows successful result conversion.

Independent BFGS/Cauchy/subspace models in `src/backend/native/reference/`
compile only for tests. Backend contracts live in `src/backend/native/tests/`;
compact identities and factorization checks sit beside the compact implementation.
Integration suites separate fixture loading, objective arithmetic and replay;
`tests/support` shares schemas and comparators. Numerical objective grouping
remains specific to each reference contract.

[Validation](validation.md) · [Performance](performance.md) ·
[Mathematical sources and clean-room boundary](provenance.md)
