# Architecture

This document describes the completed 0.3.0 implementation. Local validation
covers debug, release and native CPU builds; the final committed revision still
requires Linux/macOS/MSRV CI before publication.

`rcompat-lbfgsb` is split into two layers:

1. The compatibility layer owns R-style public semantics.
2. The backend layer owns numerical minimization in an internal scaled space.

The backend must not define the public behavior of this crate. It receives
already-scaled parameters, bounds, objective values, gradients, and backend
controls. It does not know about `fnscale`, `parscale`, `ndeps`, R result field
names, or fixture provenance.

## Compatibility layer

The compatibility layer validates input, applies scaling, maps non-finite values
to typed errors, converts gradients, and returns R-like result fields:

- `par`
- `value`
- `counts`
- `convergence`
- `message`

User objective and gradient closures always receive user-scale parameters.

## Backend layer

The native backend receives scaled objective/gradient evaluations and bounds.
Every nonempty production problem uses the stateful compact Cauchy/subspace
model and More-Thuente line search, for supplied or numerical gradients and
finite, mixed or infinite bounds. Zero-dimensional handling stays in the wrapper.

The compact model stores cyclic correction columns and small history matrices,
using O(n*m + m*m) scalar storage for dimension n and history length m. The
independent dense Hessian construction and solve are compiled only for tests.
Historical Armijo, dense-model and adjusted-control comparison probes live in
`src/backend/native/legacy.rs`, compiled only for tests; the public entry points always select the compact path and
use the requested factr, lmm and maxit without empirical overrides.

Stopping follows the finite projected-gradient norm and the signed reduction
`(old_f-new_f)/max(abs(old_f),abs(new_f),1)`. A zero pgtol requires an actually
zero projected norm. Production no longer uses gradient-noise floors or exact-zero
deferrals. A zero factr sets the reduction threshold to zero: a rounded zero or
negative reduction can still stop, as demonstrated by the new R plateau fixtures.
It does not disable the relative-reduction check.

Projected-gradient components retain the bound-distance cap recorded by R.
An unrepresentable compact target is distinguished from a failed factorization:
a zero displacement reports abnormal line-search termination rather than replacing
it with a raw gradient direction that R would not take. Factorization failures
clear history and retry construction. A failed line search with history clears
memory and retries the iteration once.

The More-Thuente search distinguishes the initial trial, the geometric feasible
cap, and its 1e10 algorithmic cap. For a first trial outside a fully finite box,
it normalizes by the Euclidean direction norm, including directions shorter than
one. First trials involving finite constraints retain the observed unit cap.
Unbounded searches can expand beyond one. These behaviors are checked against
37 independent black-box R 4.6.1 cases, including shallow quadratics, maxit=0,
linear expansion, factr=0 plateaus and unrepresentable targets.

The sufficient-decrease tolerance is 1e-3 and the search budget is 20 trials.
Successful Wolfe and feasible-cap warning steps have separate checks. A compact
unit step copies its precomputed target to preserve rounding at bounds. Narrow
brackets restore the best point at the recorded task threshold (one tenth of
the larger bracket endpoint). Production retries retain this same threshold;
the wider historical retry rule is confined to comparison probes. Six additional
black-box fixtures verify exhausted searches, warning restoration, exact callback
sequences, and restoration of the last accepted point after a single history reset.

Diagnostics report accepted trial count, alpha, cap, step norm, signed relative
reduction and curvature information. Detailed direction summaries use the actual
chosen direction and target, honoring the report interval. Tracing never runs
a second dense model. Structured observer events use the same actual model and accepted trial.

Finite differences retain forward-then-backward/base callback ordering, even
when a base evaluation could mathematically be cached. Candidate and user-scale
parameter storage are reused, while configuration is borrowed. Arithmetic checks
complete before user callbacks receive coordinates. Compact column products use
indexed reductions preserving each profile's rounding sequence.

Reference arithmetic is selected per run; the original APIs preserve their
feature-selected legacy defaults. The backend contains no scientific-dataset
logic, and the full DESeq2 example objective remains under examples/support.
Backend-specific model types do not leak into the public API.

## Literature-guided parity path

The clean-room implementation can use algorithm descriptions from the numerical
optimization literature as design inputs. The main L-BFGS-B path is described by
Byrd, Lu, Nocedal, and Zhu in "A Limited Memory Algorithm for Bound Constrained
Optimization" (<https://doi.org/10.1137/0916069>) and by Zhu, Byrd, Lu, and
Nocedal in "Algorithm 778: L-BFGS-B" (<https://doi.org/10.1145/279232.279236>).
The production multi-dimensional supplied-gradient path follows the
sufficient-decrease and curvature-condition strategy described by Moré and
Thuente (<https://doi.org/10.1145/192115.192132>). Its stage-one transition is
keyed to a nonnegative trial directional derivative, while the Armijo-scaled
derivative is used for the modified-function search test.

Test-only strong-Wolfe bracketing/zoom coverage remains in place, including a
bounded adapter that can expand beyond the unit trial up to the feasible cap,
accept sufficient-decrease steps at the cap, propagate objective evaluation
errors, and zoom back from overlarge trials.

The bounded strong-Wolfe adapter remains an internal test-only comparison mode.
It converges on Rosenbrock but is not R-compatible on counts, so it is useful as
a contrast case rather than as a production line search.

The old Armijo, capped-first-step, hybrid first-step, and strong-Wolfe modes are
kept as internal regression probes. They document why the promoted production
path needs the Cauchy/subspace model, More-Thuente extrapolation, and bracketed
step-state updates together, rather than just one of those pieces.

The committed default-Rosenbrock trace fixture now records the first sixteen R
function-evaluation points and is an active production expectation. The
promoted Cauchy/subspace plus More-Thuente path matches that prefix and the
default, loose-`factr`, and `lmm = 1` Rosenbrock final fixtures to floating-point
noise.

The DESeq-derived real-data subset is also an active production expectation for
multi-dimensional supplied-gradient parity. Its compact 4-D negative-binomial
GLM cases include seventeen actual optimizer-routed genes and fifty-one
force-optimizer probes.

The DESeq2 hard-real subset exercises objective-only finite-difference parity
on 8 finite-box contrasts and 48 hard optimizer rows. The README summarizes
current parity results, while `PROVENANCE.md` records fixture origins and replay
details.

Setting `DESEQ_PARITY_SCAN=1` keeps the same test running through every case and
prints per-gene errors for ignored-data scans. `DESEQ_PARITY_SCAN_VERBOSE=1`
adds actual and expected parameter vectors for failures, while
`DESEQ_TRACE_GENE=<gene>` and `DESEQ_BACKEND_TRACE=<level>` expose callback and
backend traces for future platform or fixture audits.

These papers are algorithmic references, not source-code inputs.

## Provenance boundary

Source provenance is architectural, not administrative. Upstream repositories may
be inspected only in ignored inspection directories or outside the repository,
and copied source must never become implementation input for this crate.

## Numerical validity

Scaling validates the arithmetic results as well as user inputs. Overflow in
coordinates, finite bounds, objective values, or gradient scaling returns
`OptimError::NumericalFailure` with the operation and optional coordinate.
Finite-difference denominators, perturbations, and derivatives must remain
finite; a stencil unable to perturb either side returns `InvalidControl`.
Intentional infinite bounds remain supported. Distinct bounds that collapse in
scaled coordinates are rejected. Final conversions are checked before returning.

A blocked or tiny direction above an explicit projected-gradient tolerance
returns convergence 52 instead of claiming stationarity. Legacy comparison probes retain their historical roundoff policy; the production
compact route uses the finite projected-gradient norm directly. Armijo steps that reach a
binding feasible cap copy the bound exactly to avoid an outward-blocked residual.

Finite-difference perturbation storage is reused across coordinates; forward,
backward and repeated base evaluations retain their observable order. Immutable
bounds and step configuration are borrowed rather than cloned for each gradient.

User-coordinate conversion now writes into a per-problem workspace shared by
objective and supplied-gradient evaluations. Checked conversion completes before
calling the user callback, including for finite-difference probes. Final returned
parameters own their storage. Compact Cholesky, triangular solves and Schur
updates use indexed reductions directly over matrix columns. Their plain/FMA
prefix-and-tail sequence is the same as for contiguous reference dots.

## Per-run arithmetic

`ArithmeticProfile` owns scalar multiply-add and contiguous/indexed short-dot
rounding policies in `src/profile.rs`. `BackendControl`, direction settings,
line-search requests and compact state carry a Copy profile value explicitly.
No profile state is stored globally or in thread-local storage. Existing BLAS
operations that were always fused retain that behavior in both profiles.

The original API resolves its feature-selected legacy profile once at entry.
Explicit-profile entry points bypass that resolution and both profiles are
compiled regardless of features. The shared fixture adapters pin x86 references;
ARM precision tests use the ARM profile explicitly. Portable CI covers default
and all-feature public APIs on Linux/macOS in debug/release. The reference job
pins Rust 1.99.0 while the portable jobs track stable. CI configuration is not
itself evidence that the remote platform jobs have passed.

## Observer and recovery boundaries

`src/diagnostics.rs` defines public scalar event records and typed termination,
model recovery, history-update and acceptance outcomes. The compact direction
builder returns a specific recovery reason instead of an unexplained None.
History updates distinguish stored corrections, skipped curvature and failed
factorization. Accepted steps retain the actual acceptance branch and directional
derivatives computed by the line search. The R-style result codes/messages are
rendered centrally from TerminationReason; no string parsing controls decisions.

ScaledProblem owns the generic observer. BackendProblem's OBSERVED constant
allows diagnostic work to be skipped at compile time for the zero-sized default
observer. Events contain no parameter copies or borrowed workspaces. The backend
emits events from actual operations; the compatibility wrapper emits Finished
only after checked conversion of the public result. Failed evaluations and final
conversion failures have distinct events. Zero-dimensional handling emits only
Finished. Validation errors before evaluation have no events.

Observer tests compare complete callback sequences and result bits across both
profiles and all bound routes, check public Wolfe/cap invariants, exercise nested
runs and failures, and verify explicit history skips and one-time memory resets.
Line-search failures have typed reasons for invalid initial/updated steps,
non-descent directions, trial exhaustion and unrepresentable targets. Accepted
events distinguish strong-Wolfe, step-limit and bracket-warning acceptance and
report initial/final directional derivatives and the curvature result.

## Production modules and independent identities

`native/iteration.rs` owns the production iteration and once-per-iteration
line-search retry. `native/line_search.rs` owns More-Thuente interpolation and
trial acceptance. `native/compact.rs` owns cyclic history and Cauchy/subspace
workspaces. The small `native.rs` adapter contains evaluation validation,
termination rendering, shared bound/projection helpers and text tracing.
Arithmetic policies live in `src/profile.rs`; public typed diagnostics live in
`src/diagnostics.rs`. `native/legacy.rs` is behind `cfg(test)`, including its
dimension-specific control heuristics. No production control or direction route
can select these probes. The production iteration does not allocate the old
unused displacement/history vectors or inspect supplied-gradient availability.

Before/after isolation replayed 288 combinations of dimensions, bound routes,
gradient forms, history sizes and profiles. Every result bit, count, status,
message, callback trace hash and observer event remained identical. The
scientific 74-fit accuracy gates and older callback comparison passed in debug
and release after isolation. Production ARM rounding and the independently
written precision callback remain active.

`compact/model_tests.rs` builds dense BFGS matrices independently and uses
pivoted Gaussian elimination, rather than compact factorizations, to verify 594
cyclic-history models and their unbounded Newton steps. It checks latest secant
identities, stored-pair rollover, all 256 eight-coordinate free masks per
profile, and the retained `2*n*m + 11*m*m` scalar-storage identity. This storage
identity does not measure temporary allocations; the allocation benchmark below
separately measures cumulative allocation requests.

Rust 1.87 is the declared minimum. Local checks include all targets/features
and public numerical/model tests with Rust 1.87.0. The remote MSRV job is a
release prerequisite; remote execution remains to be confirmed.

## Native R 4.6.2 reference and unconstrained subspace

The primary new oracle is an external source build of R 4.6.2 / DESeq2 1.52.0
with native CPU flags. Its fingerprints and self-contained captures live under
fixtures/native_r462; the original R 4.6.1 references remain separate.
Package accuracy checks recompute the previous callback baseline against this
new oracle and record bookkeeping differences explicitly. Exact-reference tests
use the same full comparator with an explicit bit contract; scan/diagnostic
printing never disables assertions. Historical external-data diagnostics are
labeled separately from full-result oracle replay.

When all bounds are infinite and history is present, the compact state solves
directly in the full subspace. Computing then cancelling a Cauchy displacement
is mathematically equivalent but changes rounding; black-box R 4.6.2 evidence
increased full stress matches from 478 to 489 out of 512 after removing it.
Further arithmetic calibration and the cached YY association fix raised this
to 512/512 under unchanged contracts. Finite-bound
routes retain their Cauchy/subspace construction and scientific precision gains.
An independent 1,404-case unit/maximization transformation test checks complete
normalized results and raw callback sequences in both arithmetic profiles.

## Measured arithmetic and allocation behavior

Independent calls to the public OpenBLAS DDOT interface determine the ARM
reduction policy without reading its implementation: lengths below 32 use a
sequential fused accumulator; full 32-element blocks accumulate in sixteen
fused lanes, reduce paired vector lanes, then append the fused scalar tail.
The 200 frozen public calls cover lengths 0 through 1024, boundary lengths and
eight seeds. Contiguous and indexed Rust reductions match every result bit.
The generator, C probe, payload and full native environment are fingerprinted
under fixtures/arithmetic and checked by scripts/verify_oracles.py.

The same explicit profile selects contraction in Cauchy breakpoint updates,
the first-trial norm, and the More-Thuente cubic discriminant. These changes
follow public optimizer callback traces and independently compiled mathematical
expressions. They do not alter dimension/history controls or select a dataset.
All 106 value and gradient calls of the ten-dimensional lower-bound analytic
diagnostic matched the native R oracle bit for bit after scalar calibration.
The formerly last stress case first exposed a one-bit subspace endpoint
difference, amplified by finite differences. High-precision public console/BLAS
traces located an earlier cached YY-product difference during simultaneous
entry/exit. The YY cache now computes (old + enter) - leave, preserving the
reference association. Other compact-cache blocks retain their measured orders.
All 1472 raw callback arguments/values and complete result bits now match R.
A broader reassociation of the other blocks passes numerical endpoint tolerances
but fails the new exact trace witness; it was rejected.

The allocation_growth benchmark uses stats_alloc 0.1.10 only as a development
dependency. Its safe instrumented System allocator counts allocations and
cumulative requested bytes for a complete solve, excluding caller input/setup
and including result construction. Objective callbacks allocate nothing. Each
supplied gradient returns one owned vector; those counts/bytes are reported
separately and subtracted when assessing solver-owned allocation cost. Both
callback times and raw function/gradient counts are reported. Numerical-gradient
raw calls must equal f + 2*n*g for these interior stencils.

The workload exhausts a fixed 24-iteration budget and covers n=32/128/512/2048,
m=1/5/10/20, both profiles, three bound routes and two callback costs. Numerical
gradients stop at n=128 to keep routine measurement practical. Every measured
run must preserve complete results and allocation counts relative to its warm-up.
Median timings come from five samples; allocator atomics and callback timers
are included, so these are profiling costs rather than uninstrumented latency.
Cumulative bytes bound transient storage but are not a direct peak-live measure.
The exact retained storage identity remains the independent memory contract.

Removing the history-step copy and reusing one compact solve buffer reduces
allocation counts and cumulative bytes in all 224 configurations while preserving
every complete-result fingerprint. For n=128,m=10, finite-gradient allocations
fall 741 to 545; bytes fall 395981 to 357661. For n=2048,m=10, bytes fall
5722957 to 5300637. Across 96 fourfold-dimension comparisons, cumulative bytes
grow by less than sixfold, rejecting dimension-quadratic temporary growth for
this fixed-budget workload. History matrices retain their m-squared component;
the measurements do not assert linear runtime in history length.

Reproduction and retained evidence:

```sh
cargo bench --bench allocation_growth > /tmp/current_allocations.csv
python3 scripts/verify_measurements.py --after /tmp/current_allocations.csv
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native \
  cargo bench --bench allocation_growth > /tmp/native_allocations.csv
```

measurements/2026-10-29 contains before/after and native CPU CSVs. All 224 native
results match the portable build fingerprints. Performance data supports the
internal buffer reuse; it does not prove that a public persistent-workspace,
joint-callback or fallible-callback API would be worth its additional contract.
The direct prototype comparison and API decisions are recorded below.

## Callback and workspace API decisions

The private api_options benchmark includes the same production backend/scaling
modules and overrides BackendProblem::value_and_gradient for comparison. It
checks 192 combinations of dimension 16/128/1024, history 1/10, both profiles,
one/16 units of callback work, stationary/nonstationary starts and four callback
forms. Every warm-up/sample compares complete parameter/value bits, counts,
status, message, typed reason and iteration count. Nine additional runs check
matching failures for nonfinite values and objective/gradient scaling overflow.
The prototypes use unit scales and finite bounds; they do not establish a new
public contract for arbitrary user error types, observers or persistent state.

| Option | Measured evidence | Decision in this change set |
| --- | --- | --- |
| Joint objective/gradient | Shared exponentials halve exactly, with complete results unchanged; expensive callbacks benefit most. | Keep the combined private backend hook and reproducible prototype. Preserve the public R-ordered separate callbacks; implicit coalescing would change their observable contract. |
| Fallible callback | The typed Result prototype preserves complete results and propagates numerical errors with no additional steady-state allocations. | Keep existing typed numerical errors. A generic user-error API needs a separate public error/observer contract; no performance requirement forces that API addition. |
| Caller gradient buffer | The original prototype saved one allocation per evaluation. | Achieve the same savings through the existing API: consume and scale the returned owned gradient Vec in place. All four forms now have equal steady-state allocation counts. |
| Persistent solver workspace | The initial-stationarity run exposed 100 unused setup allocations at n=128,m=10; multi-iteration solves spend most time after setup. | Avoid unused history construction for stationary starts and retain internal cyclic/buffer reuse. Keep persistent cross-problem state private rather than stabilizing dimension/history/reset semantics without a demonstrated batch workload. |

The API evaluation item is complete; it did not require adding every candidate
API. Its measured allocation gains are implemented without enlarging the public
surface. The initial projected-gradient test returns after the same initial
value/gradient evaluations and Started event, before history/factor allocation.
Observed supplied/numerical-gradient regressions check both zero and positive
tolerances, scaling, counts and the exact Started/Finished event order.

The ownership optimization preserves coordinate order and each multiplication/
division in gradient scaling. Length/nonfinite/overflow checks retain their
original typed failures. No per-run configuration or callback order changes.
The independent production-path allocation probe now measures 741 -> 516
allocations and 395981 -> 327965 cumulative bytes at n=128,m=10. At n=2048 its
bytes decrease from 5722957 to 4825501. The numerical-gradient path retains its
existing workspace behavior.

Four API CSVs under measurements/2026-10-29 record before/after portable/native
runs. api_environment.json fingerprints every production source, the prototype,
compiler, commands and data; callback_allocation.patch reproduces the before
state in an isolated source copy. That replay reproduced all 192 result
fingerprints and allocation metrics. Both optimizations preserve every recorded
result/count and all 192 native-build fingerprints. Timings include profiling
instrumentation and can vary with other host work; allocation/exponential counts
are deterministic. The API verifier and corruption tests run without R:

```sh
python3 scripts/verify_api_measurements.py
python3 scripts/measure_api_comparison.py
python3 scripts/test_verify_measurements.py
```

The fresh comparison uses isolated before/after source copies and separate Cargo
target directories on the same host and compiler. It verifies complete result
fingerprints, counts, work and per-evaluation allocation savings. The exponential
callback can produce different binary64 values with another host's math library;
comparing that host directly with the retained Apple CSV is not a valid
before/after experiment. Stored evidence remains independently fingerprinted.
