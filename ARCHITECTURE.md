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

For one-dimensional and finite-difference/no-gradient problems, the backend
keeps the older projected direction plus Armijo interpolation path because that
matches R's count and `maxit` edge behavior for those committed fixtures.

L-BFGS history updates use a machine-epsilon-scale positive-curvature gate.
This keeps the update acceptance policy aligned with the algorithmic tolerance
scale while leaving separate, larger safeguards in the dense test-only subspace
solve.

Backend-specific types should not leak into the public API.

## Literature-guided parity path

The clean-room implementation can use algorithm descriptions from the numerical
optimization literature as design inputs. The main L-BFGS-B path is described by
Byrd, Lu, Nocedal, and Zhu in "A Limited Memory Algorithm for Bound Constrained
Optimization" (<https://doi.org/10.1137/0916069>) and by Zhu, Byrd, Lu, and
Nocedal in "Algorithm 778: L-BFGS-B" (<https://doi.org/10.1145/279232.279236>).
The production multi-dimensional supplied-gradient path follows the
sufficient-decrease and curvature-condition strategy described by Moré and
Thuente (<https://doi.org/10.1145/192115.192132>).

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
GLM cases match the generated R `optim` outputs on counts, convergence message,
final value, and parameters within tight floating-point tolerances.

These papers are algorithmic references, not source-code inputs.

## Provenance boundary

Source provenance is architectural, not administrative. Upstream repositories may
be inspected only in ignored inspection directories or outside the repository,
and copied source must never become implementation input for this crate.
