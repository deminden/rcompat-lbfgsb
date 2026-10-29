# 0.3.0 implementation and verification

All ten parts of the accepted review are implemented and locally validated as
of 2026-10-29. The changes are committed locally on main. This version has not
been pushed or published. Final-revision Linux/macOS/MSRV CI remains
a prerequisite for a future release.

The checklist and audit below describe the completed state. Intermediate
progress notes have been removed; numerical rationale and retained evidence
remain in ARCHITECTURE.md, PROVENANCE.md and measurements/.

## Reviewed baseline

Started at e7f8bc8 on aarch64 macOS, Rust 1.99.0. The reviewed baseline already
failed three default tests: the CELA3A count (61 vs 62), and two tests of the
sine/quadratic count (23 vs 44). These compare against the pinned x86 oracle.
These differences are now covered by explicit reference-host and portable
scientific contracts, described below, without count exceptions or increased
tolerances. Both ARM precision tests passed before edits.

## 1. Numerical validity

- [x] Check arithmetic after parameter, bound, objective and gradient scaling.
- [x] Check final parameter and objective unscaling.
- [x] Reject collapsed distinct scaled bounds and non-finite bounds membership.
- [x] Add operation/coordinate-specific NumericalFailure errors.
- [x] Check finite-difference denominator, coordinates, derivative and ability to perturb.
- [x] Regression tests reproduce the original failures and pass after the fixes.
- [x] Complete release and cross-profile validation of all validity changes.

## 2. Convergence semantics

- [x] A small initial gradient cannot trigger a zero-tolerance noise shortcut.
- [x] Above-tolerance blocked/tiny directions report failure, not stationarity.
- [x] Preserve legacy roundoff allowances only after accepted gradient reduction
      with relative-reduction stopping enabled; document this compatibility policy.
- [x] Armijo binding-cap steps copy bounds to avoid outward-blocked residuals.
- [x] Add public regressions for tiny gradients and near-bound stagnation.
- [x] Introduce typed termination, model-recovery and history-update outcomes.
- [x] Audit zero-factr and roundoff policies against the new 37 black-box R fixtures.

## 3. Validation contracts

- [x] Complete-result comparator supports exact bits and numerical tolerances.
- [x] Mutation tests cover each result field, non-finite values and vector lengths.
- [x] Standard fixture harness uses the comparator.
- [x] Strict synthetic scans check counts, convergence, message and case count.
- [x] External-data scans are explicitly ignored rather than silently returning.
- [x] Adopt shared comparison throughout the remaining scientific harnesses.
      Complete optimizer references use the full comparator; package accuracy
      checks explicitly record counts/messages as observations. Archived scans
      share finite/shape-safe endpoint metrics and preserve their diagnostic label.
- [x] Add exact-profile assertions for all published exact-parity claims.
      Current bit claims use explicit profiles and complete comparators/traces;
      legacy x86 full-bit mode has a dedicated CI gate. Missing historical
      corpora are labeled historical, not current reproducible guarantees.
      Remote exact-profile execution remains part of the release audit.
- [x] Provide the synthetic corpus in the repo or immutable checksummed storage.
      The user approved an alternative when the historical bundle could not be
      found. The new self-contained 512-case R 4.6.2 corpus is included in the repository;
      all 512 full-result contracts now pass, with an additional exact trace witness.
- [x] Verify strict scan behavior using deliberately corrupted oracle datasets.
      A positive selected case plus nine mutations run through the actual replay
      executable; all six frozen payloads also have SHA256 corruption gates.

## 4. Large-problem complexity

- [x] Extend compact production state to mixed/infinite bounds.
- [x] Use the compact limited-memory model for unbounded problems.
- [x] Confine the independent dense solver to reference test builds.
- [x] Verify model identities and active-set/history transitions.
- [x] Commit dimension/history benchmarks proving intended memory/runtime growth.
      Committed benchmarks and retained before/after/native measurements cover
      224 configurations; exact retained-storage tests and 96 dimension-growth
      comparisons accompany timing data. Publication remains a separate release action.

## 5. Unbounded line search

- [x] Separate initial trial, feasibility cap and algorithmic search cap.
- [x] Support unbounded expansion and verify first-step normalization.
- [x] Add R fixtures for expansion, warning, restoration and exhaustion.
      The 37 mixed-bound cases plus six complete line-search traces cover these
      outcomes, including a single history reset before final exhaustion.
- [x] Independently verify Wolfe/bracket invariants and warning acceptance reasons.

## 6. Diagnostics

- [x] Remove reconstruction of a separate dense model for trace output.
- [x] Emit summaries of actual chosen directions and honor reporting intervals.
- [x] Test equality of callback sequence and result with trace enabled.
- [x] Add structured events with active/free counts, history and restart reasons.
- [x] Provide observed entry points, verify invariance and add an overhead benchmark.

## 7. Arithmetic profiles and CI

- [x] Add explicit per-run profile selection with existing-entry-point compatibility.
- [x] Record full oracle environment fingerprints, commands and hashes.
      Both installed ARM R runtimes now have binary/distribution snapshots,
      capture recipe hashes and reproduction commands. Original prebuilt
      compiler provenance and the unavailable July x86 environment are labeled
      historical/unknown rather than inferred from current configuration.
- [x] Separate portable invariants, pinned exact parity and scientific accuracy jobs.
      CI has a dedicated exact x86 contract lane, portable invariant lanes and
      ARM/native scientific lanes. Saved CI evidence confirms MSRV, both exact
      x86 lanes and ARM precision passed at the preceding numerical revision.
      Final local validation repairs the two cross-host test contracts described
      below; the final CI configuration has not been pushed.
- [x] Test public integration behavior on ARM in debug and release.
- [x] Configure a pinned Rust 1.99.0 reference job and stable portable jobs.
      The final revision still needs these jobs before any future publication.

## 8. Backend organization

- [x] Prove production reachability, remove obsolete heuristics or isolate probes.
- [x] Split iteration, line search, compact state, fallback, arithmetic and diagnostics.
- [x] Replace ambiguous boolean/Option recovery outcomes with explicit states.
      Direction/recovery, history updates, search failure and termination are typed;
      low-level optional numerical solves are mapped at the model boundary.
- [x] Verify mechanical refactors separately from numerical changes.
- [x] Reconcile architecture documentation with final production routes.

## 9. Allocation performance

- [x] Reuse one finite-difference perturbation vector, retaining callback order.
- [x] Borrow bounds and step configuration rather than cloning per gradient.
- [x] Reuse user-coordinate conversion workspaces.
- [x] Remove temporary compact-factorization column allocations, preserving reduction order.
- [x] Improve retained history storage if the legacy path survives the routing audit.
      The vector-removal path is test-only; production uses fixed cyclic columns.
- [x] Benchmark allocations, optimizer overhead, callback cost and raw evaluations.
      Development-only allocator instrumentation separates callback allocations
      and both callback times, reports raw/optimizer calls and preserves results.

## 10. Numerical corpus and release contract

- [x] Add deterministic generated quadratics with known constrained solutions.
- [x] Cover correlated/ill-conditioned, flat, linear, nonconvex and maximization cases.
      The 512-case R corpus, 96 independent constant-objective cases, 37 mixed
      linear/plateau oracles and 1,404 transformations exercise these families.
- [x] Exercise dimensions 1,2,9,10,11,13 and large dimensions across routes/profiles.
- [x] Add independent dense-reference compact/model tests and transformation checks.
      594 dense-reference model states and 1,404 equivalent unit/maximization
      transformations preserve complete results/callback sequences.
- [x] Separate oracle generation from verification with binary64 round-trip checks.
      Frozen replay/integrity requires no R. Independent native-R boundary tests
      cover 2,013 numbers, signed zero, subnormals, corruption and atomic failure.
- [x] Declare and validate an MSRV.
- [x] Map every supported guarantee to a reproducible clean-checkout command.
      README maps public/scientific/exact/model/allocation/MSRV contracts to
      specific commands and identifies final-revision CI as a release prerequisite.
- [x] Evaluate joint/fallible callback and solver workspace APIs from measurements.
      Direct private prototypes preserve complete results across 192 configurations
      in portable/native builds. Implemented owned-gradient reuse and the measured
      stationary-start fast path; retained the existing public callback surface.
      The architecture records the rationale and limits of each API decision.
- [x] Audit the full original review against final sources and validation evidence.
      The final ten-part audit and local 0.3.0 validation are recorded below.
      Remote publication and final-revision CI execution remain subsequent
      release actions requiring an explicit request.


## Final ten-part audit and local 0.3.0 delivery

Every item in the accepted original review now has an implementation and a
reproducible verification contract. Version 0.3.0 is recorded in local Git
history on main. No remote push, release tag or publication is part of this
delivery.

1. Numerical validity: inspected `scaling.rs`, `finite_diff.rs`, `objective.rs`,
   `bounds.rs`, `compat.rs` and typed errors. Arithmetic is checked after
   scaling/unscaling and stencils are checked for representability. Public
   numerical-validity and complete-result mutation tests pass.
2. Convergence: inspected the production compact iteration's initial projected
   gradient, blocked/tiny direction policy, reset/search failures and final
   stopping precedence. The constant-objective, tiny-gradient, roundoff,
   zero-factr, bound and search exhaustion oracles cover the accepted policies.
3. Validation: exact/numerical full-result comparison checks every field;
   scientific comparison keeps separate bookkeeping observations. Frozen
   payloads, actual replay subprocess corruptions and measurement corruptions
   have positive controls and reject alterations, missing rows and bad shapes.
4. Complexity: every nonempty production run enters `minimize_compact`;
   retained columns use O(n*m) storage and factors O(m*m). The n*n independent
   dense reference is test-only. All 594 model states, retained-storage checks,
   224 measured configurations and 96 growth comparisons pass.
5. Search: initial trial, feasibility cap and search cap are distinct. The
   37 mixed/unbounded cases and six complete trial traces verify expansion,
   warning acceptance, restoration and exhausted retries without count changes.
6. Diagnostics: public generic observers emit scalar data from the actual
   production state. Eight observer tests, callback/result equality, feature
   invariance and retained overhead measurements verify the public contract.
7. Profiles/environments/CI: profiles are explicit per run. Both installed ARM
   R environments now have ten executable/library hashes each, 98/100 exact
   bootstrap archive records, runtime/build metadata and a fingerprinted capture
   recipe. The old 37/6 R artifacts regenerated byte-identically. The unavailable
   original July Linux environment remains explicitly historical. Saved CI
   evidence confirms pinned x86 debug/release, ARM precision and MSRV succeeded
   at the preceding numerical revision. Final-revision CI is required before
   publication, with matching reference architectures checked explicitly.
8. Organization: inspected production reachability and cfg boundaries. Iteration,
   line search, compact model, arithmetic and diagnostics have separate modules;
   obsolete control-specific alternatives and dense solves remain test-only.
   Recovery, history, termination and acceptance outcomes are typed at model
   boundaries; architecture documentation describes the current routes.
9. Allocation: production reuses finite-difference/user-coordinate/compact
   workspaces and cyclic history columns, scales owned supplied gradients in
   place, and avoids history setup for initially stationary problems. Frozen
   source/data hashes and all per-evaluation allocation checks pass. A fresh
   same-host before/after runner now builds each phase with independent targets,
   preserving all 192 complete-result fingerprints and accounting. An altered
   fingerprint across all callback forms is rejected even when within-run
   callback equality remains intact.
10. Corpus/API/release: all 512 native reference results and the 64-base/1,472-raw
    exact witness pass. Independent checks cover 576 constrained quadratics,
    96 constant models, 1,404 transformations and 256 NB global-optimum runs.
    The measured joint/fallible/buffer API decision remains documented rather
    than expanding the public API speculatively. MSRV, native compilation,
    source cleanliness, packaging and the README guarantee map are verified.

### Explicit callback-math contracts

The prior three Mac failures compared x86 callback trajectories: CELA3A counts
61/62 and the duplicated sine/quadratic counts 23/44. The saved x86 jobs pass
the strict 68+48 result-bit contracts. These whole scientific replay suites and
the trigonometric full replay now require Linux x86; their unchanged endpoint/
value/convergence contracts also run independently as portable scientific tests.
Unsupported full replays have explicit ignore reasons and stay mandatory in
the pinned job. There are no case-ID count exceptions or tolerance increases.

The saved Linux stress failure likewise involves NB callback exp/log rounding:
one numerical-gradient case changes counts and its stopping message. Full
512-result replay remains mandatory on macOS ARM with its original 1e-8
endpoint/value limits and exact bookkeeping. Every host additionally runs all
384 polynomial full-result contracts, including the exact active-set witness,
and all 128 NB models under both arithmetic profiles against the independent
global optimum. The NB accuracy controls are explicitly separate: maxit=500,
factr=100, pgtol=1e-7, coefficient error <=1e-5 and objective error bounded by
1e-9*(1+abs(optimal_value)). The original R replay controls are unchanged.

The other saved Linux failure compared fresh exponential callback benchmarks
with Apple CSVs. CI now verifies the retained experiment separately and rebuilds
both before/after phases on its own host and compiler. Full result fingerprints,
counts and allocation/work accounting remain strict.

### Final local verification

On Apple Silicon/macOS 27.2, Rust 1.99.0, package 0.3.0:

- Default all-target debug, all-feature all-target release, and native-CPU
  all-feature all-target release each pass 153 Rust tests across 28 targets.
  Parameterized case/state counts are listed above. Each suite explicitly
  reports three unsupported x86 full replays and three missing historical
  external scans as ignored, with reasons.
- Native compilation uses `RUSTFLAGS='-C target-cpu=native'` and a separate
  target directory; all 512 results and independent accuracy checks pass there.
- Clippy and rustdoc with warnings denied, formatting, diff whitespace, Rust
  1.87 all-target/all-feature compilation and 12-case feature isolation pass.
- Six frozen corpora, 200 public DDOT results, both installed runtime snapshots,
  their recipe hashes and all oracle mutation gates pass without installed R.
- All retained 224 allocation/192 API configurations, 96 growth comparisons,
  same-host fresh API phase builds and measurement mutations pass.
- Local package verification passes: 204 files, no
  inspection/data/build/private artifacts. No runtime dependencies were added.
- Package/README/changelog are 0.3.0; the historical measured 0.2.2 manifest is
  explicitly retained as a measurement input. Only main remains locally and in
  remote-tracking refs. Deliverable source/content contains no tool branding.

## Release state

All implementation, fixtures, measurement evidence and current-state notes are
included in the local 0.3.0 commit. The working tree is clean after committing.
No release tag or remote push is authorized by this local commit request. Before
a future publication, run the final committed revision through the configured
Linux/macOS/MSRV CI jobs and review their results.
