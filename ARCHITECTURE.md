# Architecture

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

Backends implement an internal trait over a scaled objective and gradient. The
current backend is an in-tree bounded limited-memory quasi-Newton implementation.
It reports optimizer-level function and gradient counts in the R result style,
separate from any extra user objective calls needed internally for finite
differences.

Line searches are intentionally bounded to a small fixed trial budget matching
the L-BFGS-B operating model. This keeps failed or flat searches from silently
following paths that R would not explore.

`factr = 0` disables relative-reduction convergence, matching the L-BFGS-B/R
control semantics; projected-gradient convergence and iteration limits still
apply.

Backend trace diagnostics include the accepted line-search trial count, accepted
alpha, feasible alpha cap, and step norm, plus a curvature-condition diagnostic.
High-verbosity traces also include the accepted parameters and gradient so R
trace output can be compared when investigating remaining path drift.

For multi-dimensional supplied-gradient problems, the native backend uses an
independently written generalized Cauchy point, dense free-variable subspace
minimizer, and More-Thuente-style line search with bracketed
cubic/quadratic/secant step updates. That is the current main R-compatibility
path.

The production More-Thuente sufficient-decrease tolerance is `1e-3`, matching
the L-BFGS-B line-search behavior observed through R fixtures. The older
projected Armijo fallback keeps its separate smaller sufficient-decrease
constant because it is used only for one-dimensional and finite-difference
paths that already match their R fixtures.

For bounded multi-dimensional supplied-gradient problems, the first constrained
More-Thuente trial is capped at `stp = 1`, matching the L-BFGS-B 2.3 code path
bundled with R. When that capped trial satisfies sufficient decrease and keeps
descending, the native backend accepts it as a successful R-style `STPMAX`
warning case instead of forcing another extrapolation.

The More-Thuente main path uses a machine-epsilon-scale minimum step so
finite-bound supplied-gradient problems can take R's final tiny cleanup step
before `pgtol = 0` convergence. The Armijo fallback keeps the older, larger
minimum step because lowering it changes finite-difference evaluation counts.
The near-zero projected-gradient compatibility shortcut is limited to
infinite-bound supplied-gradient cases, where committed fixtures show R and the
clean-room floating-point path can differ by representational noise.
No-gradient finite-difference problems with mixed finite/infinite bounds have
their own larger projected-gradient noise floor because the stencil can leave a
near-optimal point with a tiny nonzero gradient while R still reports
projected-gradient convergence. Fully unbounded finite-difference problems do
not use that shortcut because R takes one additional cleanup evaluation there.
The one-iteration exact-zero `pgtol` deferral is kept to supplied-gradient
infinite-bound cases only.
The final value refresh after multidimensional interpolation is also limited to
supplied-gradient problems; no-gradient finite-difference fixtures already have
the accepted final value and R does not charge an extra optimizer-level
evaluation there.
For multi-dimensional finite-difference paths with positive `pgtol` or
`maxit = 0`, the quadratic interpolation trial is not damped, matching R cases
where the first accepted trial is also the reported final point. The older
damping stays in place for default exact-zero `pgtol` and one-dimensional
fallback cases where committed fixtures depend on the extra cleanup step.

For one-dimensional and finite-difference/no-gradient problems, the backend
keeps the older projected direction plus Armijo interpolation path because that
matches R's count and `maxit` edge behavior for those committed fixtures.

L-BFGS history updates use the Algorithm 778 machine-epsilon-scale curvature
skip test based on the previous directional derivative, with a norm-scaled
positive-curvature fallback when that derivative is unavailable. This keeps the
update acceptance policy aligned with the algorithmic tolerance scale while
leaving separate, larger safeguards in the dense test-only subspace solve.

Backend-specific types should not leak into the public API.

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
GLM cases currently include seventeen actual optimizer-routed genes and fifty-one
force-optimizer probes selected from a full ignored-data scan where 491 of 512
DESeq rows matched the strict active contract. The committed cases match the
generated R `optim` outputs on counts, convergence message, final value, and
parameters within tight floating-point tolerances.

Setting `DESEQ_PARITY_SCAN=1` keeps the same test running through every case and
prints per-gene errors for ignored-data scans. `DESEQ_PARITY_SCAN_VERBOSE=1`
adds actual and expected parameter vectors for failures, while
`DESEQ_TRACE_GENE=<gene>` and `DESEQ_BACKEND_TRACE=<level>` expose paired R/Rust
objective and backend traces for the remaining drift targets.

These papers are algorithmic references, not source-code inputs.

## Provenance boundary

Source provenance is architectural, not administrative. Upstream repositories may
be inspected only in ignored inspection directories or outside the repository,
and copied source must never become implementation input for this crate.
