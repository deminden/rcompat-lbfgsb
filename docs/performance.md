# Performance

The compact backend uses `O(n*m + m*m)` storage for dimension `n` and history
length `m`. It reuses coordinate, gradient, correction and factorization buffers;
initially stationary solves skip unused history allocation.

## Benchmarks

Run from the repository root:

| Command | Measures |
| --- | --- |
| `cargo bench --bench backend_cost` | Solver/callback time across bounds, gradients, history sizes and tracing |
| `cargo bench --bench observer_cost` | Observer overhead with result-bit equality |
| `cargo bench --bench allocation_growth` | Allocations, cumulative bytes, callback times and evaluation counts |
| `cargo bench --bench api_options` | Callback evaluation/allocation costs and complete-result equality |

Set `RUSTFLAGS='-C target-cpu=native'` for native compilation.
`backend_cost` accepts `LBFGSB_BENCH_DIMENSIONS=16,128,1024`; `report=0` suppresses
text tracing. Benchmark workloads do not run under `cargo test --all-targets`.
Timings include instrumentation and depend on the host. Cumulative allocated
bytes measure allocation requests, not peak live memory.

## Recorded evidence

[Raw CSVs and metadata](../measurements/2026-10-29) retain compiler/source hashes,
manifest snapshots and reversible baseline patches. Comparisons cover Rust
implementation variants. The 224 allocation and 192
callback configurations preserve complete-result fingerprints and evaluation counts.

| Workload | Baseline | Current |
| --- | ---: | ---: |
| Bounded supplied-gradient solve, n=128/m=10: allocations | 741 | 516 |
| Same solve: cumulative bytes | 395,981 | 327,965 |
| Stationary backend startup, n=128/m=10: allocations | 104 | 3 |

```sh
python3 scripts/verify_measurements.py
python3 scripts/verify_api_measurements.py
python3 scripts/test_verify_measurements.py
python3 scripts/measure_api_comparison.py
```

Verifiers check coverage, source/data fingerprints, result preservation and
allocation accounting. Dimension-growth checks reject quadratic temporary
storage; mutation tests exercise these gates. The comparison runner rebuilds
before/after copies with independent targets on the same host/compiler.
Use `--output NEW_DIRECTORY` to retain its builds and CSVs.

`api_options` compares separate/joint evaluation, fallible callbacks and buffer
reuse through benchmark-local adapters. Production callbacks are evaluated
separately, and supplied gradients are scaled in their returned owned vectors.
