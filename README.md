# rcompat-lbfgsb

Pure Rust, clean-room compatibility for R's
`stats::optim(..., method = "L-BFGS-B")`.

`rcompat-lbfgsb` is for applications that need R-shaped optimizer behavior
without embedding or invoking R. It preserves familiar controls such as
`fnscale`, `parscale`, `ndeps`, `factr`, `pgtol`, and `lmm`, and returns
R-style parameters, objective value, evaluation counts, convergence code, and
message.

> **Current reference target:** R 4.6.1 on `x86_64-conda-linux-gnu`, using
> single-threaded OpenBLAS 0.3.32.

## Current status

On the pinned x86 reference stack, the compatibility suite passes in both
debug and release builds. CPU and compiler arithmetic matter for exact parity.

| Validation set | Coverage | Current result |
| --- | ---: | --- |
| Standard generated `optim` fixtures | 63 fixture files | Passes the committed parameter, value, count, status, and trace contracts |
| DESeq-derived supplied-gradient GLMs | 68 cases | Bit-exact parameters, values, counts, convergence, and messages |
| Hard objective-only GLMs | 48 cases across 8 contrasts | Bit-exact parameters, values, counts, convergence, and messages |
| Synthetic finite-box stress suite | 512 cases | Bit-exact in debug and release |
| Frozen worst-case precision comparison | 100 cases | All 100 finish with zero endpoint error |

Here, **bit-exact** means equality of the serialized IEEE-754 result fields and
the public optimizer bookkeeping on the pinned reference inputs. It is not a
claim that every objective, compiler, CPU, BLAS implementation, or future R
version follows the same floating-point path.

## Reference software stack

The current fixtures and stress oracle were regenerated and verified on
2026-07-22.

| Layer | Pinned reference |
| --- | --- |
| Public behavior | `stats::optim(method = "L-BFGS-B")` |
| R | 4.6.1 (2026-06-24) |
| R platform | `x86_64-conda-linux-gnu` |
| BLAS/LAPACK | OpenBLAS 0.3.32, `OPENBLAS_NUM_THREADS=1` |
| Scientific fixture environment | Bioconductor 3.23, DESeq2 1.52.0 |
| Fixture numeric serialization | `%.17g` with required binary64 round trips |
| Rust validation toolchain | rustc/cargo 1.96.0, LLVM 22.1.2, `x86_64-unknown-linux-gnu` |

R, Bioconductor, DESeq2, and OpenBLAS are development-time reference tools
only. The crate has no runtime dependencies and does not call, link, shell out
to, or embed any of them. Detailed fixture origins, preserved comparison
environments, and regeneration evidence are recorded in
[Provenance](PROVENANCE.md).

## What is implemented

- objective-only optimization using R-shaped bounded finite differences;
- optimization with a supplied gradient;
- finite, mixed, and infinite bounds, including projected starting points;
- R-style scaling and maximization through `parscale` and `fnscale`;
- vector `ndeps`, `factr`, `pgtol`, `maxit`, and `lmm` controls;
- generalized Cauchy points, compact free-variable subspace minimization, and a
  bracketed More-Thuente line search for multidimensional finite boxes;
- R-compatible objective-call order for bounded finite-difference stencils;
- typed errors for invalid controls, bounds, dimensions, and non-finite values;
- zero-dimensional, fixed-parameter, active-bound, and line-search-refresh edge
  behavior covered by generated fixtures.

## Quick start

Add the crate:

```toml
[dependencies]
rcompat-lbfgsb = "0.2.2"
```

Optimize an objective without supplying a gradient:

```rust
use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl, OptimError};

fn main() -> Result<(), OptimError> {
    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0])?,
        |p| (p[0] - 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )?;

    assert!((result.par[0] - 2.0).abs() < 1e-5);
    assert!(result.is_success());
    Ok(())
}
```

For an analytic gradient, use `optim_lbfgsb_with_gradient` with the same
`Bounds` and `OptimControl` types.

## Real DESeq2 precision on macOS ARM

The optional `arm-fma` feature models the scalar multiply-add contraction of
the R 4.6.1 Clang build and the measured OpenBLAS 0.3.34 Vortex short-dot
reduction. Enable it explicitly for that reference profile; default arithmetic
continues to target the pinned x86 stack.

The independent [DESeq2 objective example](examples/support/deseq2_objective.rs)
retains the actual log2 link, full NB likelihood, Gaussian coefficient prior,
captured starts, and default optimizer controls. Stable deviance/Stirling
formulas avoid large log-gamma cancellation. Neither the example nor the
optimizer invokes R, and no runtime dependency was added.

Against real R 4.6.1 / DESeq2 1.52.0 fits on macOS ARM:

| Cohort | Fits | Previous worst coefficient error | New worst | Improvement |
| --- | ---: | ---: | ---: | ---: |
| Existing hard real inputs | 46 | 0.0152846 | 0.00139872 | 10.9x |
| Published airway counts, GSE52778 | 28 | 0.818664 | 0.00000376263 | 217,578x |

These compare against the frozen full-likelihood/prior callback, not the
historical reduced objective. The largest observed per-fit error increase is
`1.18e-7`. Both worst-case improvements were reproduced in debug and release.
This is a result for these cohorts and reference
profile, not a universal bit-exact DESeq2 claim.

The previous-callback numerical benchmark runs on its macOS ARM oracle
platform. Fixture contract checks run on every platform.

```sh
cargo test --features arm-fma --test deseq2_precision
cargo test --release --features arm-fma --test deseq2_precision
cargo run --features arm-fma --example deseq2
```

[Generate the published-count fixtures](scripts/generate_airway_fixtures.R)
with the separate `deseq2-r461` environment. Rust replay needs neither R nor
network access. On the current ARM host, the older x86-pinned suite has known
CELA3A and sine/quadratic evaluation-count differences; those assertions
remain unchanged.

## R-compatible semantics

- User objective and gradient closures always receive parameters in user scale.
- `parscale` maps user parameters to internal coordinates as
  `internal_x[i] = user_par[i] / parscale[i]`.
- `fnscale` maps objective values to internal minimization values as
  `internal_value = user_value / fnscale`; a negative value performs R-style
  maximization.
- `ndeps` is interpreted in scaled parameter space.
- `OptimResult.par` and `OptimResult.value` are returned in user scale.
- `OptimResult.counts` follows R's optimizer-level function/gradient counting,
  separate from extra closure calls used to construct finite differences.

## Validation

Normal validation does not require R:

```sh
cargo test --all-targets
cargo test --release --all-targets
cargo test --features arm-fma --test deseq2_precision
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

When the ignored R 4.6.1 stress bundle is available locally, its exact scan can
also be run directly:

```sh
LBFGSB_SYNTHETIC_STRESS_SCAN=1 \
LBFGSB_SYNTHETIC_STRESS_PAR_TOL=0 \
LBFGSB_SYNTHETIC_STRESS_VALUE_TOL=0 \
LBFGSB_SYNTHETIC_STRESS_VALUE_REL_TOL=0 \
LBFGSB_SYNTHETIC_STRESS_STRICT=1 \
cargo test --release --test synthetic_stress_scan \
  synthetic_stress_can_scan_ignored_reference_data -- --nocapture
```

The expected summary is 512/512 exact parameters, values, and counts, with zero
failures.

## Scope and limitations

- Compatibility targets R's L-BFGS-B method, not every `optim` method and not a
  general replacement for the R runtime.
- Floating-point-sensitive objectives can follow a different path if their
  callback arithmetic differs from the R expression used to create a fixture.
- Human-readable backend trace diagnostics are useful for audits but are not
  byte-for-byte copies of R's console trace output.
- Fixed coordinates without a supplied gradient return an error, matching the
  observed R finite-difference behavior covered by the fixtures.
- The parity claim is deliberately limited to the documented fixture families
  and pinned target stack; new platforms and workloads should add fixtures.

## Design and provenance

The implementation is independently written from public numerical-method
literature, public R documentation, and black-box input/output fixtures. It does
not copy, translate, or paraphrase R, C, Fortran, or other GPL optimizer source.

- [Architecture](ARCHITECTURE.md)
- [Provenance and clean-room boundary](PROVENANCE.md)
- [Fixture documentation](fixtures/r_optim_lbfgsb/README.md)
- [Changelog](CHANGELOG.md)

Licensed under MIT.
