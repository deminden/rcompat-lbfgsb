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
alpha, feasible alpha cap, step norm, signed relative objective reduction, and
a curvature-condition diagnostic. High-verbosity traces also include the
accepted parameters and gradient so R trace output can be compared when
investigating remaining path drift.

The `factr` convergence check uses R's signed relative decrease
`old_f - new_f`, not an absolute objective delta. This is equivalent on normal
descent steps and preserves R behavior for rare accepted non-decreasing steps.

For multi-dimensional supplied-gradient problems, and for multi-dimensional
objective-only problems whose bounds form a fully finite box, the native backend
uses an independently written generalized Cauchy point, free-variable subspace
minimizer, and More-Thuente-style line search with bracketed
cubic/quadratic/secant step updates. That is the current main R-compatibility
path, including DESeq2-style `[-30, 30]` coefficient boxes.

For every multidimensional fully finite box on the default Cauchy/subspace and
More-Thuente path, the backend uses a coherent clean-room compact-matrix state
shaped after the public L-BFGS-B algorithmic formulas. The same state and
operation grouping apply to supplied-gradient and finite-difference problems;
one-dimensional and mixed/infinite-bound finite-difference problems retain the
older fallback path. Generated fixtures validate this routing across both
callback forms and dimensions 4 through 13; the current oracle version and
results are summarized in the README.

The objective-only finite-box main path includes one early limited-memory
refresh guard for the Cauchy/subspace model. R's bundled L-BFGS-B 2.3 can
discard correction history after a compact subspace factorization failure; the
clean-room backend mirrors the observed shape by restarting from an identity
model when the early subspace target is explosively far beyond the generalized
Cauchy step. This is guarded by hard-real trace fixtures rather than treated as
a tunable convergence setting.

The production More-Thuente sufficient-decrease tolerance is `1e-3`, matching
the L-BFGS-B line-search behavior observed through R fixtures. The older
projected Armijo fallback keeps its separate smaller sufficient-decrease
constant because it is used only for one-dimensional and mixed/infinite-bound
finite-difference paths that already match their R fixtures.

For bounded multi-dimensional supplied-gradient problems, the first constrained
More-Thuente trial is capped at `stp = 1`, matching the L-BFGS-B 2.3 code path
bundled with R. When that capped trial satisfies sufficient decrease and keeps
descending, the native backend accepts it as a successful R-style `STPMAX`
warning case instead of forcing another extrapolation.
For finite-box Cauchy/subspace targets that land exactly on a finite bound, the
main line search preserves R's full-step copy shape by reusing the precomputed
target when `stp == 1` instead of recomputing `x + d`. This is part of the
coherent compact state for both supplied-gradient and finite-difference paths.

The main path also has a narrow R-style line-search recovery for flat or
nonconvex cases. If a line search exhausts its trial budget while correction
history exists, the backend clears the limited-memory history and retries the
same iteration once from the current point. On that retry only, a sufficiently
decreasing point in a narrowed More-Thuente bracket can be accepted as a warning
shaped step. The committed `line_search_refresh_sin_quad` fixture guards the
public R behavior: a finite-box objective-only sinusoidal quadratic emits a
line-search memory refresh in R and converges by `factr` rather than returning
abnormal termination.

An exact finite-box More-Thuente task also restores and re-evaluates its stored
best point when the bracket width is at most one tenth of the current step cap.
The narrower threshold is distinct from the one-half retry-warning threshold;
black-box call traces and a captured-bit scalar regression test guard the
distinction. Versioned trace provenance is recorded in `PROVENANCE.md`.

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
The projected-gradient norm itself follows R's bound-distance cap: for a
negative gradient near an upper bound, or a positive gradient near a lower
bound, the component is limited by the distance to that bound rather than using
the raw gradient magnitude. The `pgtol_near_upper_uses_bound_distance` fixture
guards this initial-convergence behavior.
The objective-only finite-box Cauchy/subspace free-variable set uses exact
bound hits after projection, while supplied-gradient paths and the fallback
direction projection keep a tiny roundoff tolerance when blocking outward
motion. This split preserves R fixture behavior for near-zero finite-difference
bound cleanup and supplied-gradient DESeq cases while improving hard-real
objective-only subspace routing.
The final value refresh after multidimensional interpolation is also limited to
supplied-gradient problems; no-gradient finite-difference fixtures already have
the accepted final value and R does not charge an extra optimizer-level
evaluation there.
For multi-dimensional finite-difference paths with positive `pgtol` or
`maxit = 0`, the quadratic interpolation trial is not damped, matching R cases
where the first accepted trial is also the reported final point. The older
damping stays in place for default exact-zero `pgtol` and one-dimensional
fallback cases where committed fixtures depend on the extra cleanup step.

For one-dimensional no-gradient problems, and for no-gradient problems with at
least one infinite bound, the backend keeps the older projected direction plus
Armijo interpolation path because that matches R's count and `maxit` edge
behavior for those committed fixtures. Fully finite multi-dimensional
objective-only boxes use the main Cauchy/subspace path instead.

Finite-difference gradients intentionally follow R's bounded-stencil raw
objective-call order: evaluate the forward point first, then the backward point
or clamped base point. A pure objective gives the same gradient if the clamped
base value is cached or evaluated in the opposite one-sided order, but R's
black-box call trace exposes the order, so the compatibility layer preserves
the user-visible closure-call semantics.

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
