# Changelog

## Unreleased

- Initialized the clean-room crate scaffold.
- Added provenance, architecture, validation, scaling, finite-difference, and
  bounded backend foundations.
- Removed the temporary `lbfgsb-rs-pure` runtime dependency and added an in-tree
  native bounded limited-memory quasi-Newton backend.
- Tightened committed R parity fixtures to check `par`, `value`, `counts`,
  `convergence`, and `message`.
- Added parity fixtures for active upper bounds, fixed parameters with supplied
  gradients, initial projected-gradient convergence, infinite bounds, mixed
  bounds, and finite differences with `parscale`.
- Matched R's fixed-parameter no-gradient edge by returning a typed error before
  finite-difference optimization.
- Added a compact DESeq-derived real-data fixture subset with generated R
  `optim` outputs and an exact-parity target for 4-D negative-binomial GLM
  cases.
- Tightened the native backend toward L-BFGS-B behavior by capping line-search
  trials at the R/Fortran default scale and preserving free quasi-Newton
  direction components even when the current coordinate gradient is zero.
- Matched R's zero-dimensional L-BFGS-B no-op behavior and allowed
  projectable infinite initial parameters with finite bounds.
- Matched R's 2-D `parscale` gradient fixture to floating-point noise by adding
  guarded multidimensional quadratic interpolation and R-style convergence
  counting after that interpolated path.
- Split the remaining ignored Rosenbrock control target into separate
  `factr`-loose and `lmm=1` fixtures so future precision work can track the two
  path drifts independently.
- Added tested identity-Hessian generalized Cauchy point scaffolding for future
  L-BFGS-B-style GCP/subspace implementation, currently exposed only through
  high-verbosity backend trace diagnostics.
- Extended that scaffold with clean-room limited-memory direct-Hessian products
  and a B-aware generalized Cauchy point, with tests that tie the direct product
  back to the existing two-loop inverse recursion.
- Added a tested dense subspace-minimization scaffold after the generalized
  Cauchy point, including remaining-bound clipping and high-verbosity trace
  diagnostics for the future production step path.
- Added line-search trial and step-norm diagnostics plus tests for full-step and
  interpolated-step accounting, giving future R trace comparisons a stable
  internal signal.
- Added R-generated parity fixtures for no-gradient finite-difference
  optimization starting within one `ndeps` of lower and upper bounds.
- Documented the primary clean-room algorithm references for the L-BFGS-B main
  path and added curvature-condition diagnostics to the line-search trace.
- Added a backend regression test for Armijo-accepted steps that are too short
  for the Wolfe curvature condition, documenting the next line-search parity gap
  without changing active optimizer behavior.
- Added a test-only strong-Wolfe bracketing/zoom scaffold that expands past
  too-short Armijo steps and zooms back from overlarge steps without changing
  the production line search yet.
- Extended line-search diagnostics with accepted alpha and feasible alpha cap,
  and added a test-only bounded strong-Wolfe adapter covering expansion beyond
  the unit trial and zooming under box constraints.
- Made the test-only bounded strong-Wolfe adapter propagate objective
  evaluation errors and accept sufficient-decrease steps at the feasible cap,
  matching the bounded-search shape needed before production use.
- Added an internal test-only line-search mode for complete optimizer probes and
  recorded that primary strong-Wolfe currently converges on Rosenbrock but
  drifts from R evaluation counts, so it remains gated off production.
- Added an internal test-only direction mode for complete GCP/subspace optimizer
  probes and recorded its Rosenbrock count drift, keeping production on the
  active fixture-matching projected L-BFGS direction.
- Added a capped-first-trial Cauchy/subspace probe that closely matches R's
  first Rosenbrock trace shape while documenting why that path remains test-only
  until the complete run also matches R counts.
- Added a hybrid first-Cauchy-then-projected probe showing that the R-shaped
  first step recovers the `lmm = 1` Rosenbrock evaluation count but still misses
  final point/value parity, narrowing the next backend work.
- Extended high-verbosity backend traces with accepted parameters and gradients
  so local Rust runs can be compared directly against R trace output.
- Split the L-BFGS history update curvature gate from dense-solve safeguards and
  moved history acceptance to a machine-epsilon-scale threshold.
- Matched the `factr = 0` control edge by disabling relative-reduction
  convergence while preserving projected-gradient convergence.
- Added a committed R-generated Rosenbrock evaluation-trace prefix and an
  ignored backend trace target to pin the remaining GCP/More-Thuente main-path
  divergence.
- Added a test-only More-Thuente higher-function-value interpolation probe that
  reproduces R's first Rosenbrock GCP line-search prefix exactly, while keeping
  the incomplete full optimizer mode gated by recorded count drift.
- Pinned the dev-dependency manifest to the latest stable `serde` and
  `serde_json` crates currently resolved from crates.io.
- Added a test-only full optimizer probe that uses the R-shaped
  Cauchy/More-Thuente first step and then returns to the projected L-BFGS path,
  proving the first seven Rosenbrock trace evaluations can match while the
  eighth evaluation and later counts still drift.
- Promoted the clean-room Cauchy/subspace direction plus More-Thuente
  line-search path for multi-dimensional supplied-gradient optimization,
  matching the default, loose-`factr`, and `lmm = 1` Rosenbrock fixtures.
- Extended the committed R Rosenbrock evaluation trace to sixteen function
  evaluations and made the production trace prefix an active test.
- Kept one-dimensional and no-gradient finite-difference cases on the projected
  Armijo path to preserve existing R count and `maxit` edge parity.
- Replaced the simplified More-Thuente fallback inside the production path with
  bracketed cubic/quadratic/secant step-state updates, preserving the
  Rosenbrock trace guardrail and promoting the DESeq-derived real-data parity
  target to an active `cargo test` case at `1e-8` tolerances.
