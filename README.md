# rcompat-lbfgsb

Pure Rust, clean-room compatibility for R's
`stats::optim(..., method = "L-BFGS-B")`.

`rcompat-lbfgsb` is for applications that need R-shaped optimizer behavior
without embedding or invoking R. It preserves familiar controls such as
`fnscale`, `parscale`, `ndeps`, `factr`, `pgtol`, and `lmm`, and returns
R-style parameters, objective value, evaluation counts, convergence code, and
message.

> **Primary development reference:** source-built R 4.6.2 and DESeq2 1.52.0
> on macOS ARM / Apple Silicon, compiled with `-O3 -mcpu=native`, using one OpenBLAS
> 0.3.34 thread. The R 4.6.1 x86 and ARM fixtures remain regression references.

## Current status

Version **0.3.0** is committed locally on `main` and has not been published.
Implementation and local validation are complete. Debug, release and native CPU
all-target suites each pass 153 Rust tests across 28 targets, with six explicitly
ignored tests for unsupported reference hosts or unavailable historical bundles.
The final revision still requires remote CI before publication.

The version preserves the verified scientific precision while expanding
numerical validation. Local evidence recorded on 2026-10-29 is:

| Validation set | Coverage | Current result |
| --- | ---: | --- |
| Native R 4.6.2 DESeq2 reference | 74 routed fits | Accuracy gains preserved in debug and native-CPU release builds |
| R 4.6.1 scientific precision reference | Same 74 routed fits | Existing accuracy gates still pass |
| New R 4.6.2 replacement stress corpus | 512 cases | 512 complete-result matches; unchanged endpoint/value/count/status/message gates |
| Active-set cache rounding witness | 64 base / 1,472 raw calls | Exact result bits, base-call bits and complete raw-callback digest match R |
| Independent closed-form quadratics | 576 cases | Known constrained solutions through dimension 1,024, both profiles |
| Independent constant objectives | 96 cases | Exact projected parameters/value, initial counts and raw calls, including maximization |
| Equivalent unit/maximization transformations | 1,404 cases | Identical normalized result bits and callback sequences |
| Independent compact model identities | 594 states | Dense BFGS/Newton identities and cyclic storage checks pass |
| New mixed-bound and line-search R oracles | 37 + 6 cases | Declared complete-result and callback contracts pass |
| Portable polynomial stress replay | 384 cases | Complete results and exact active-set witness, without transcendental callback math |
| Independent negative-binomial optima | 256 runs | All 128 NB cases, both profiles, coefficient accuracy within 1e-5 |

Full-result replays include evaluation counts and termination messages and require
the reference callback math library. The 512-case replay runs on macOS ARM;
legacy transcendental and 116 scientific optimizer replays run on Linux x86.
Other hosts run the portable polynomial and scientific accuracy checks with
the same endpoint/value tolerances. Unsupported full replays are explicitly
marked ignored, with a reason, and remain mandatory on their matching CI host.
Exact-reference assertions are enabled with `LBFGSB_EXACT_REFERENCE=1`;
numerical tolerance checks do not substantiate a bit-exact claim.

Historical pinned-x86 validation recorded exact results for 68 supplied-gradient
GLMs, 48 hard objective-only GLMs, a 512-case synthetic bundle, and a 100-case
precision comparison. The last two source corpora are unavailable in this
checkout. They are historical evidence, not a claim that the current worktree
has reverified those runs. The user approved the separate replacement corpus;
its provenance and verification contracts are documented in
[fixtures/synthetic_v1/README.md](fixtures/synthetic_v1/README.md).
The new environment, reference captures and reproduction commands are in
[fixtures/native_r462/README.md](fixtures/native_r462/README.md).

Here, **bit-exact** means equality of the serialized IEEE-754 result fields and
all public optimizer bookkeeping under a declared reference contract. Profiles
model measured rounding; they cannot make arbitrary callback math libraries
identical across CPUs and operating systems.

Minimum supported Rust: **1.87**. CI checks the library, all targets/features,
and public numerical contracts with Rust 1.87.0. The pinned compatibility job
uses Rust 1.99.0; stable jobs exercise Linux and macOS in debug and release.

## Reference software stack

The following archived R 4.6.1 stack describes the legacy x86 fixtures generated
on 2026-07-22. The primary native R 4.6.2 stack is recorded separately above.

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
  bracketed More-Thuente line search for every nonempty bound configuration;
- R-compatible objective-call order for bounded finite-difference stencils;
- typed errors for invalid controls, bounds, dimensions, and non-finite values;
- zero-dimensional, fixed-parameter, active-bound, and line-search-refresh edge
  behavior covered by generated fixtures.

## Quick start

Add the crate:

```toml
[dependencies]
rcompat-lbfgsb = "0.3.0"
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

## Select arithmetic per optimization

Use `optim_lbfgsb_with_profile` or
`optim_lbfgsb_with_gradient_and_profile` to select
`ArithmeticProfile::R461X86` or `ArithmeticProfile::R461MacArm` for one run.
Both implementations are available in every build; selecting one explicitly
ignores the `arm-fma` Cargo feature. Profiles can be mixed in concurrent or
nested runs. `profile.id()` provides a stable reference identifier for logs.
The original entry points continue to use `ArithmeticProfile::legacy()`, which
preserves their existing feature-dependent defaults.

```rust
use rcompat_lbfgsb::{optim_lbfgsb_with_profile, ArithmeticProfile, Bounds, OptimControl};

let profile = ArithmeticProfile::R461X86;
let result = optim_lbfgsb_with_profile(
    vec![0.0], Bounds::unbounded(1), |x| (x[0] - 2.0).powi(2),
    OptimControl::default_for_dimension(1), profile,
).expect("valid finite quadratic");
```

Profile selection controls backend rounding, not the arithmetic inside user
callbacks or the host math library. A profile identifier alone does not promise
cross-platform endpoint parity. Fixture tests select their recorded arithmetic
explicitly. The ARM scientific precision gate runs on its macOS ARM oracle host.
Run `python3 scripts/check_profile_feature_invariance.py` to verify explicit
results across separate feature builds.

## Observe a run

`optim_lbfgsb_observed` and `optim_lbfgsb_with_gradient_observed` accept an
explicit profile and an `OptimObserver` (including an ordinary `FnMut(OptimEvent)`
closure). Events report initialization, the actual compact model's active/free
coordinates and history, accepted trials, correction outcomes, resets and typed
termination. Accepted trials distinguish strong Wolfe, step-limit and bracket
warning acceptance and report the directional derivatives used by the search.

```rust
use rcompat_lbfgsb::{optim_lbfgsb_observed, ArithmeticProfile, Bounds, OptimControl, OptimEvent};

let mut accepted_steps = 0;
let result = optim_lbfgsb_observed(
    vec![0.0], Bounds::unbounded(1), |x| (x[0]-2.0).powi(2),
    OptimControl::default_for_dimension(1), ArithmeticProfile::R461X86,
    |event| { if let OptimEvent::Accepted { .. } = event { accepted_steps += 1; } },
).expect("finite quadratic");
```

Diagnostic floats use internal scaled units; public results and objective/gradient
callbacks use user units. Model active/free counts describe the proposed target,
not the final accepted line-search point. Finished events are emitted only after
successful final conversion. Evaluation and output-conversion failures have
separate events; invalid inputs produce no events. An empty problem emits only
Finished. Counts exclude failed/incomplete evaluations and finite-difference
closure calls, as described by the optimizer count contract.

Observers run synchronously and may call independent nested optimizations.
They never cause extra objective/gradient evaluations. The original entry points
use a zero-sized disabled observer, with diagnostic-only work guarded by compile-time
constants. The observer controls its own logging/storage costs.
Run `cargo bench --bench observer_cost` for a CSV overhead probe that also checks
result bits across observed and unobserved runs.

## Real DESeq2 precision on macOS ARM

The `ArithmeticProfile::R461MacArm` profile models the scalar multiply-add contraction of
the R 4.6.1 Clang build and the measured OpenBLAS 0.3.34 Vortex short-dot
reduction. Select it explicitly for that reference profile. The optional `arm-fma`
feature selects the same behavior for the original entry points; explicit profile
selection is independent of that feature.

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
cargo test --test deseq2_precision
cargo test --release --test deseq2_precision
cargo run --example deseq2
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
cargo test --test deseq2_precision
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
  synthetic_stress_can_scan_ignored_reference_data -- --ignored --nocapture
```

The archived pinned-x86 run reported 512/512 exact parameters, values and counts.
The absent source bundle prevents reverifying that run from this checkout; the
self-contained replacement corpus is the current stress gate.

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

## Measuring optimizer and callback cost

Run `cargo bench --bench backend_cost` for CSV measurements across finite,
mixed and unbounded bounds, supplied and numerical gradients, history sizes,
callback costs, and disabled versus suppressed detailed tracing. Override the
size list with `LBFGSB_BENCH_DIMENSIONS=16,128,1024`. Each row reports the median
elapsed time of five runs, objective-only callback time from that same run, raw
objective calls, and optimizer-level function counts. Gradient callback time is
included in elapsed time, not in the objective-only column. Timers add overhead
for cheap callbacks, so these are diagnostic measurements rather than latency
promises. Tracing uses `report=0` to isolate computation from terminal output.
The benchmark workload does not run during `cargo test --all-targets`.

Allocation reductions currently reuse user-coordinate and finite-difference
storage and evaluate compact-factorization column products without temporary
vectors. History updates write directly into their retained cyclic column, and
compact factorization reuses one solve buffer across columns.

`cargo bench --bench allocation_growth` measures both callback times, raw and
optimizer evaluation counts, allocations, and cumulative allocated bytes across
32/128/512/2048 variables, histories 1/5/10/20, both profiles and all bound routes.
Numerical gradients use dimensions through 128. Five measured runs follow a
warm-up; every run must have identical complete results and allocation counts.
The profiler uses a development-only allocator wrapper; the library has no
runtime dependencies and continues to forbid unsafe Rust.

The retained [before/after measurements](measurements/2026-10-29/allocation_owned.csv)
preserve all 224 complete-result fingerprints. At 128 variables/history 10,
bounded solves decreased from 741 allocations/395,981 cumulative bytes to
516/327,965. At 2,048 variables the corresponding byte totals decreased from
5,722,957 to 4,825,501. This includes scaling each supplied gradient in its owned
buffer, saving one allocation per evaluation. Timing includes instrumentation and is host-specific.
Cumulative allocated bytes are not peak live memory. Verify coverage, raw-call
accounting, result preservation and linear dimension growth with
`python3 scripts/verify_measurements.py`. The [native CPU run](measurements/2026-10-29/allocation_native.csv)
also preserves all 224 result fingerprints. See the architecture for measurement
scope and interpretation.

`cargo bench --bench api_options` compares private joint/fallible callback and
buffer prototypes against the production evaluation path. All 192 configurations
preserve complete results in portable and native builds. The measurement led to
owned-gradient reuse and an initial-stationarity fast path: in the measured
128-variable backend startup, allocations fall from 104 to 3. The existing public
callback contract is retained. Verify the prototype evidence with
`python3 scripts/verify_api_measurements.py`. To repeat the before/after comparison
on another host, run `python3 scripts/measure_api_comparison.py`; it builds both
phases independently with the same compiler and callback math. The architecture
records the API decisions.

The production compact model now handles every nonempty bound configuration,
including one-dimensional and numerical-gradient problems. Initial non-box
trials use Euclidean normalization; unbounded searches can expand to a separate
1e10 algorithmic cap. `factr=0` sets a zero reduction threshold and can still
accept a rounded-zero reduction. See `fixtures/mixed_bounds/README.md` for the
new black-box oracle and `tests/generated_quadratics.rs` for closed-form checks.
The dense reference solve is compiled only in tests. Historical comparison-mode
scaffolding remains private and does not select public production routing.

The compact model also has an independent dense BFGS/Newton reference test over
594 cyclic-history states, active-set transition checks, and a retained-storage
identity of `2*n*m + 11*m*m` floating-point slots plus O(n) index storage. The
six line-search recovery fixtures compare every trial and complete result bit
for bit. These checks are reproducible without R:

```sh
cargo test --lib backend::native::compact::model_tests
cargo test --test line_search_oracles
python3 scripts/verify_oracles.py
python3 scripts/test_verify_oracles.py
python3 scripts/verify_measurements.py
python3 scripts/test_verify_measurements.py
python3 scripts/verify_api_measurements.py
cargo +1.87.0 check --all-targets --all-features
```

Generation is separate from these R-free verification commands. In the native
R reference environment, run `Rscript --vanilla scripts/test_oracle_roundtrip.R`
to independently test numeric serialization boundaries and failed-write atomicity.
Regenerate public DDOT measurements into a new directory with
`python3 scripts/generate_blas_oracle.py --prefix "$CONDA_PREFIX" --output /tmp/new_blas_oracle`.
The recorded environment and CSV must reproduce byte-identically on the pinned
reference host.

## Guarantee and verification map

Run these commands from a clean Git checkout. Normal replay requires Rust and,
for integrity/measurement scripts, Python 3. Oracle generation uses the separate
documented R environments. Each row identifies the actual comparison scope;
passing one tier does not establish a stronger tier.

| Contract | Reproduction command | Evidence/scope |
| --- | --- | --- |
| Checked scaling, unscaling, finite differences and typed numerical failures | `cargo test --test numerical_validity --test validation_contract` | Overflow, collapsed bounds, unrepresentable stencils, result shape/finiteness and deliberate field corruption |
| Public controls, bounds, scaling and callback semantics | `cargo test --test api_smoke --test bounds --test finite_diff --test quadratic --test scaling` | R-shaped public results and observable evaluation behavior |
| Known constrained optima and truly flat objectives | `cargo test --test generated_quadratics` | 576 quadratic plus 96 constant-objective cases; both profiles and gradient forms |
| Compact model identities and retained memory growth | `cargo test --lib backend::native::compact::model_tests` | 594 independent dense-BFGS/Newton states, active-set transitions and storage identity |
| Mixed/unbounded search and recovery | `cargo test --test mixed_bound_oracles --test line_search_oracles` | 37 declared full-result contracts; six bit-exact results and complete trial traces |
| Replacement stress and active-set arithmetic | `cargo test --test stress_oracles` | On macOS ARM, all 512 complete results; every host checks 384 polynomial full results/exact witness and 256 NB known-optimum runs |
| Unit changes and maximization | `cargo test --test transformations` | 1,404 equivalent normalized result/callback sequences |
| Profile isolation and observation invariance | `cargo test --test arithmetic_profiles --test observers` and `python3 scripts/check_profile_feature_invariance.py` | Alternating/concurrent/nested runs, full callbacks/results, events and feature-build independence |
| Scientific precision on the ARM reference host | `cargo test --test deseq2_precision --test native_reference_precision` | Old/new callback accuracy against 74 routed R 4.6.1 and R 4.6.2 package fits; bookkeeping observations remain separate |
| Native CPU scientific validation | `RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native cargo test --release --all-features --test native_reference_precision` | Same accuracy contract with native CPU compilation |
| Legacy x86 exact results | `LBFGSB_EXACT_REFERENCE=1 cargo test --test deseq_real_subset --test deseq_hard_real_subset` on Linux x86 | Full result bits/counts/status/messages; 68 + 48 cases passed the pinned debug/release CI jobs before the local 0.3.0 metadata change |
| Frozen oracle availability, fingerprints and corruption rejection | `python3 scripts/verify_oracles.py` and `python3 scripts/test_verify_oracles.py` | Six payloads, exact witness capture recipes and public BLAS arithmetic fingerprint; no R invocation |
| Allocation/callback cost and dimension growth | `cargo bench --bench allocation_growth` and `cargo bench --bench api_options` | Repeated full-result equality, raw calls, allocations/bytes, 224 growth and 192 callback configurations |
| Stored measurement coverage and mutations | `python3 scripts/verify_measurements.py`, `python3 scripts/verify_api_measurements.py`, `python3 scripts/test_verify_measurements.py` | Source/data hashes, before/after comparisons, exact per-evaluation savings and negative controls; `python3 scripts/measure_api_comparison.py` rebuilds both callback phases on the current host |
| Minimum Rust version | `cargo +1.87.0 check --all-targets --all-features` | Library, integration targets, examples and benchmarks compile on the declared minimum |
| Full local and release suites | `cargo test --all-targets --no-fail-fast` and `cargo test --release --all-features --all-targets --no-fail-fast` | Every supported contract runs; host-specific full replays and missing historical external bundles have explicit ignore reasons |

Quality/release prerequisites also include `cargo fmt --all -- --check`,
`cargo clippy --all-targets --all-features -- -D warnings`,
`RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features` and
`cargo package --allow-dirty` (use `cargo package` for the final clean release).
The final source revision must pass the configured Linux/macOS/MSRV CI jobs before
publication. The 0.3.0 commit remains local; no release tag has been created.
Historical missing 512/100-case corpora are not presented as clean-checkout
guarantees. Installed
oracle binary hashes, exact bootstrap package archives, configured compilers and
reproduction commands are recorded in `fixtures/reference_environments/`.
