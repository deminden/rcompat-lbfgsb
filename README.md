# rcompat-lbfgsb

Pure Rust, clean-room implementation of R's `stats::optim(method = "L-BFGS-B")` behavior.
Supports bounded optimization, supplied or numerical gradients, R-style controls,
per-run arithmetic profiles and structured diagnostics.

**0.3.0 · Rust 2024 · Rust 1.87+ · MIT**

The library has no runtime dependencies and requires neither R nor BLAS.

## Get started

```toml
[dependencies]
rcompat-lbfgsb = "0.3.0"
```

```rust
use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl, OptimError};

fn main() -> Result<(), OptimError> {
    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0])?,
        |x| (x[0] - 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )?;
    assert!(result.is_success());
    assert!((result.par[0] - 2.0).abs() < 1e-5);
    Ok(())
}
```

Run the included example:

```sh
cargo run --example quadratic
```

## Compatibility

Callbacks and returned values use user units. Starts are projected into bounds;
fixed coordinates require a supplied gradient. Evaluation counts follow R's
optimizer bookkeeping and exclude extra finite-difference callback calls.

Reference tests use R 4.6.2 and DESeq2 1.52.0. Callback math and floating-point
rounding can change optimization paths across platforms; compatibility follows
[declared test contracts](docs/validation.md). Text trace output is crate-specific.

## Validation and performance

Observed maximum absolute differences against R 4.6.2 / DESeq2 1.52.0 on
Apple Silicon, matching in debug and CPU-native release; rounded to six significant digits:

| Reference | Results | Max abs parameter difference | Max abs objective difference |
| --- | ---: | ---: | ---: |
| R stress corpus | 512 | `1.22125e-15` | `1.26021e-28` |
| R supplied-gradient GLMs | 68 | `4.23237e-10` | `4.39968e-11` |
| R objective-only GLMs | 48 | `4.63329e-5` | `8.99176e-8` |
| DESeq2 hard inputs | 46 | `1.39872e-3` | `2.81616e-7` |
| DESeq2 airway | 28 | `3.76263e-6` | `1.31095e-12` |

R replay also matches counts, convergence codes and messages exactly. DESeq2
uses the [scientific accuracy contract](docs/validation.md). Exact trace checks
cover six line-search scenarios and an active-set witness with 1,472 raw calls.

The [allocation probe](docs/performance.md), n=128/m=10, measures 741 → 516
allocations and 395,981 → 327,965 cumulative bytes, preserving complete results.

## Guides

- [Usage](docs/usage.md): gradients, controls, results, profiles and observers.
- [Validation](docs/validation.md): precision limits, platform scope and checks.
- [Performance](docs/performance.md): benchmarks and allocation evidence.
- [Architecture](docs/architecture.md): modules and numerical invariants.
- [Provenance](docs/provenance.md): clean-room boundary, references and fixture origins.
- [Changelog](CHANGELOG.md): release changes.
