# Changelog

## 0.3.0 - Unreleased

Implementation and local validation are complete as of 2026-10-29. The version
is committed locally on `main`; no release tag or publication has been made.
Debug, release and native CPU all-target suites pass. Final-revision remote CI
remains required before publication.

- Add source-built native R 4.6.2 / DESeq2 1.52.0 reference captures while
  retaining R 4.6.1 regression data and the verified scientific precision gains.
- Replace unavailable stress data, with user approval, by a self-contained
  512-case corpus. All 512 complete-result contracts now pass.
- Preserve left association in the cached YY-product update during simultaneous
  free-set entry/exit. Add an exact 64-base/1,472-raw-call regression witness.
- Add source-fingerprinted public console/BLAS trace capture without consulting
  optimizer implementations, plus 96 independent constant-objective cases.
- Solve retained unconstrained models directly in the full subspace to avoid
  Cauchy cancellation roundoff, resolving 11 new reference mismatches.
- Add shared scientific accuracy/result contracts, exact-mode CI assertions,
  signed-zero-safe oracle serialization and real replay corruption tests.
- Separate host-specific full callback replays from portable accuracy contracts;
  retain strict native 512-case and x86 116-case gates. Add 256 independent NB
  optimum checks and same-host before/after callback allocation measurement.
- Fingerprint both installed R oracle runtimes and their bootstrap package archives.
- Use Apple Silicon hardware descriptions consistently in documentation,
  reference metadata and capture tools.
- Verify 1,404 equivalent unit/maximization transformations and native CPU builds.
- Calibrate ARM long-dot reduction against 200 independent public BLAS calls;
  correct profile-specific Cauchy and cubic-interpolation contraction and trial
  normalization. Extend independent dense model coverage to 594 states.
- Add allocation/growth profiling over 224 configurations with before/after and
  native CPU evidence. Reuse compact solve buffers and write history steps
  directly into retained columns, reducing allocations in every measured case
  while preserving complete-result fingerprints.
- Scale supplied gradients in their returned owned buffer and skip unused history
  construction when the initial projected gradient already meets tolerance.
- Measure joint/fallible callback and reusable-buffer prototypes over 192
  configurations, with full-result/error checks and native CPU comparisons.
- Add independent binary64 serialization boundaries, atomic-failure checks and
  mutation tests for BLAS fingerprints and allocation evidence.

- Declare Rust 1.87 as the minimum and add an MSRV release gate.
- Isolate old control/direction heuristics in test-only comparison code; split
  production iteration, line search and compact state without changing precision.
- Report typed line-search failure reasons and fix an overly permissive retry
  warning that could accept a rounded zero-length step after a memory reset.
- Add six exact R line-search traces, payload fingerprint/corruption checks,
  dense-reference compact model identities and cyclic/active-set storage tests.

- Add observed entry points with generic, synchronous scalar event callbacks;
  preserve callback order and result bits when observation is enabled.
- Add typed termination, model-recovery, history-update and acceptance outcomes,
  including explicit Wolfe/cap/bracket-warning classification and derivatives.
- Render public R-style termination messages centrally and emit final success
  only after checked result conversion. Add observer failure/nesting regressions
  and a standalone observer-cost benchmark.

- Promote the compact state and More-Thuente search to all nonempty production
  problems, including 1-D and mixed/infinite-bound numerical gradients. Confine
  the independent dense Hessian solve to test builds.
- Normalize first non-box trials by Euclidean direction length and permit
  unbounded expansion to the measured 1e10 cap. Remove production gradient-noise
  shortcuts and preserve R behavior for unrepresentable targets.
- Correct factr=0 semantics: the zero reduction threshold still accepts rounded
  zero reductions. Add 37 versioned black-box R 4.6.1 fixtures with generator
  hashes and checked binary64 serialization, plus 576 closed-form quadratic cases.

- Add explicit per-run `ArithmeticProfile` selection for both callback forms;
  preserve the original API's feature-dependent legacy default. Both arithmetic
  implementations are available regardless of feature settings.
- Isolate rounding policies in a profile module and pass profile values through
  the compact model, line search and fallback directions without global state.
- Add nested/concurrent profile tests, independent binary64 rounding checks and
  an executable comparison across separate feature builds.
- Pin fixture arithmetic explicitly, expand debug/release portable CI across
  Linux/macOS and feature configurations, and pin the reference Rust toolchain.

- Reject overflow after scaling and unscaling, invalid numerical stencils, and
  non-finite bounds membership checks. Add operation-specific numerical errors.
- Prevent small initial slopes and blocked directions from reporting false
  projected-gradient convergence; preserve post-progress reference roundoff rules.
- Reuse finite-difference candidate and user-coordinate storage, borrow gradient
  configuration, and remove temporary column vectors from compact factorization.
- Add a dependency-free CSV optimizer/callback benchmark over bound routes,
  dimensions, history sizes, callback costs and trace settings.
- Use the complete-result comparator for supplied-gradient scientific fixtures.
- Emit direction traces from the actual model without reconstructing a dense solve.
- Add public numerical regressions and mutation tests for a complete-result
  fixture comparator. Strict synthetic scans now check counts, status, message
  and executed case count in addition to endpoint error.
- External-data scans are explicitly ignored by default; append `-- --ignored`
  to the selected test command to execute them with the oracle bundle present.

## 0.2.2 - 2026-10-14

### macOS ARM arithmetic and real DESeq2 precision

- Added opt-in `arm-fma` arithmetic for the measured R 4.6.1 macOS ARM
  Clang/OpenBLAS Vortex profile. The default pinned x86 arithmetic remains
  available without enabling the feature; public controls are unchanged.
- Added a reusable, independent full negative-binomial/prior objective and
  a runnable DESeq2 example, without adding runtime dependencies.
- Reduced worst coefficient error across 46 existing actual DESeq2 fits from
  `0.0152846` to `0.00139872` (10.9x), and across 28 published airway fits
  from `0.818664` to `3.76263e-6` (217,578x). Both results were reproduced in
  debug and release. The largest observed per-fit error increase is `1.18e-7`.
- Added compact public-API fixture generation from airway/GSE52778 counts,
  real DESeq2 fallback capture, and comparisons with the frozen previous callback.
- Documented platform-specific validation and retained the existing strict
  x86 fixture assertions, including their known count differences on ARM.
- Split CI validation by arithmetic profile: default x86 tests on Linux,
  focused `arm-fma` precision tests on macOS ARM, with both required before
  the existing tagged publish job can run.

## 0.2.1 - 2026-07-22

### R 4.6.1 oracle refresh

- Regenerated all 63 standard fixtures plus the DESeq supplied-gradient and
  hard-real objective-only subsets with R 4.6.1. The update is metadata-only:
  every saved numeric result, count, trace, status, message, and input is
  unchanged from R 4.6.0.
- Added a separately preserved R 4.6.1 stress bundle. Its 512 cases and all
  payload files are byte-identical to the R 4.6.0 bundle, and zero-tolerance
  Rust scans remain exact in debug and release builds.

## 0.2.0 - 2026-07-20

### R 4.6.0 parity milestone

- Regenerated all optimizer fixtures with pinned R 4.6.0 and reached bit-exact
  endpoint, value, count, convergence, and message parity for all 68 committed
  DESeq supplied-gradient cases, all 48 committed hard-real objective-only
  cases, and all 512 ignored synthetic stress cases.
- Added a coherent clean-room compact Cauchy/subspace state, exact finite-box
  operation grouping, and R-shaped More-Thuente task-warning restoration. The
  all 100 cases in the frozen worst-case stress comparison now have zero
  endpoint error, with no regressions.
- Reworked the public documentation around the pinned R/OpenBLAS target stack,
  the evidence-bounded meaning of bit-exact parity, reproducible validation,
  and current limitations.

### Earlier development history

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
- Matched the production More-Thuente sufficient-decrease tolerance to the
  R/L-BFGS-B `1e-3` behavior, fixing a multi-dimensional `pgtol` stop after
  line search without regressing Rosenbrock or DESeq parity.
- Added R-generated parity fixtures for 3-D supplied-gradient quadratics,
  active 3-D box constraints, 2-D `pgtol` convergence after an optimizer step,
  and 2-D finite-difference optimization with both `parscale` and vector
  `ndeps`.
- Added active supplied-gradient parity fixtures for multi-dimensional fixed
  coordinates, covering both fixed+free and all-fixed parameter vectors.
- Split the minimum-step floor by backend path: the More-Thuente
  supplied-gradient path now accepts R's final machine-epsilon-scale cleanup
  step, while the Armijo finite-difference fallback keeps its existing count
  behavior.
- Added active R-generated fixtures for all-unbounded 2-D supplied-gradient
  optimization, 2-D `factr = 0`, supplied-gradient initial projected-gradient
  convergence, and combined `fnscale`/`parscale` supplied-gradient scaling.
- Matched a mixed finite/infinite no-gradient finite-difference case where R
  reports projected-gradient convergence despite tiny stencil noise at the
  final point, without changing finite-bound finite-difference termination.
- Added active fixtures for 2-D mixed-bound finite differences, finite
  differences near both sides of a box, scaled bounds with supplied gradients,
  2-D `maxit = 1`, and `lmm = 2` Rosenbrock parity.
- Avoided an extra final optimizer-level evaluation for no-gradient
  finite-difference paths after multidimensional interpolation, matching R's
  count for combined `fnscale`/`parscale` finite-difference optimization.
- Added an active combined `fnscale`/`parscale` finite-difference fixture, and
  recorded a scaled active-bound cleanup-count drift in ignored exploration
  notes rather than promoting it before a principled rounding-path fix exists.
- Matched positive-`pgtol` multi-dimensional finite-difference interpolation by
  using an undamped quadratic trial only for that fallback path, preserving the
  damped exact-zero `pgtol` and one-dimensional fixtures.
- Added active fixtures for 2-D `maxit = 0`, loose-`factr` 2-D quadratics,
  positive-`pgtol` finite differences, and `lmm = 10` Rosenbrock parity.
- Matched 2-D no-gradient `maxit = 0` finite-difference parity by using R's
  undamped first quadratic interpolation for that terminal Armijo fallback path,
  and added the adjacent `maxit = 1` finite-difference fixture as a guardrail.
- Expanded the committed DESeq-derived real-data subset with three more actual
  optimizer-routed genes that preserve the active `1e-8` R parity contract, and
  recorded nearby count-drift probes in ignored exploration notes.
- Added an active all-unbounded 2-D finite-difference fixture for the
  no-gradient Armijo fallback path, and narrowed the finite-difference
  infinite-bound projected-gradient noise floor to mixed finite/infinite boxes.
- Added active lower-only and upper-only half-infinite 2-D finite-difference
  fixtures to guard the same no-gradient fallback path.
- Added an active no-gradient finite-difference fixture for initial
  projected-gradient convergence under positive `pgtol`.
- Expanded the committed DESeq-derived real-data subset from nine to sixty-eight
  active parity cases; the full ignored-data scan now finds 491 of 512 selected
  DESeq rows preserving exact count and `1e-8` final-value/parameter parity,
  and the generator supports opt-in `DESEQ_FIXTURE_GENES=all` scans.
- Aligned the L-BFGS history-update skip threshold with Algorithm 778's
  directional-derivative machine-epsilon criterion, retaining a norm-scaled
  fallback for non-descent diagnostic paths.
- Corrected the More-Thuente stage-one transition to use the trial directional
  derivative crossing zero, and added an opt-in `DESEQ_PARITY_SCAN=1` test mode
  for full ignored-data parity scans.
- Matched R's bundled L-BFGS-B 2.3 first constrained line-search cap (`stp = 1`)
  and accepted the corresponding sufficient-decrease `STPMAX` warning shape as
  a successful step, improving the full ignored DESeq scan from 487/512 to
  490/512 strict passes without losing any previously passing row.
- Promoted `LINC02470`, `DKKL1P1`, and `ENSG00000305324` into the active DESeq
  subset, and added verbose DESeq trace controls for inspecting the remaining
  drift targets.
- Added an R-generated synthetic `first_constrained_linear_2d` fixture that
  exercises the same first constrained `STPMAX` acceptance path without relying
  on real DESeq data.
- Corrected the mirrored More-Thuente cubic-step gamma sign for left-of-best
  trial steps, improving the full ignored DESeq scan to 491/512 strict passes,
  promoting `IGLV3-27`, and reducing `FGG` from a count split to exact R counts.
