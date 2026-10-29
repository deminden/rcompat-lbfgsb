# Self-contained replacement stress corpus

The user approved generating an alternative after the historical 2026-07-22
512-case bundle could not be found. These are **new cases**, not a recovered or
byte-identical copy of that historical dataset.

`cases.json` contains 512 black-box R 4.6.2 results generated on the native
Apple Silicon reference environment documented in `../native_r462/README.md`.
The deterministic cross product covers dimensions 1,2,4,9,10,11,13,32, four
bound routes, two starts, supplied/numerical gradients, shallow/deep history,
parameter-unit changes and maximization through a negative `fnscale`.

The independent mathematical families are correlated quadratics,
ill-conditioned quadratics, negative-binomial GLMs with coupled design rows,
and coupled nonconvex double wells. Every case stores all its inputs, controls
and complete R result. No ignored data, network download or R installation is
needed for replay.

Current optimizer verification matches **512/512** cases under the unchanged
endpoint/value/count/status/message contract. The initial run matched
478/512; directly solving the retained full subspace on unconstrained problems
removed 11 mismatches, including the negative-binomial count mismatch. No case
or bookkeeping field was omitted to obtain that increase. Independent public
BLAS probes and native compiler arithmetic checks then corrected long-dot
reduction, Cauchy scalar contraction, first-trial normalization and cubic
interpolation rounding, resolving a further 22 cases. The last case,
`ill_conditioned_quadratic:11:mixed:1:FALSE`, exposed the association of a cached
YY-product update when one variable enters the free set and another leaves it.
Adding the entering sum to the cached entry before subtracting the leaving sum
removes the discrepancy. All 1,472 raw callbacks then match R bit for bit.

```sh
# In the native R 4.6.2 environment, regenerate explicitly:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 Rscript --vanilla scripts/generate_stress_fixtures.R

# No R needed for verification:
python3 scripts/verify_oracles.py
python3 scripts/test_verify_oracles.py
cargo test --test stress_oracles
```

The full 512-case replay requires the recorded macOS ARM callback math library
and is mandatory in that CI job. It is explicitly ignored on other hosts,
where Rust's exponential/logarithm implementations can change a numerical
gradient, count or stopping message even with the same optimizer profile.
Every host also runs all 384 polynomial cases under the unchanged complete-result
contract, including the exact active-set witness. A separate accuracy test runs
all 128 NB cases under both profiles with `maxit=500`, `factr=100`, `pgtol=1e-7`:
the independent global optimum is zero because counts equal size factors and
the design includes the identity. Coefficients must be within 1e-5 and the
objective within 1e-9 times one plus its optimal magnitude. This accuracy test
does not substitute for the native 512-case result replay.

`LBFGSB_STRESS_CASE` selects one explicit case ID in a chosen replay test; an
unknown selection fails. To run only the full replay, use
`cargo test --test stress_oracles stress_corpus_matches_r -- --exact --include-ignored`.
`LBFGSB_STRESS_ORACLE` selects an alternate artifact for replay. A subprocess
regression first verifies an unchanged selected case and then checks that nine
corruptions fail through the actual strict replay harness: parameter, objective,
both count fields, convergence, message, missing case, wrong vector length and
non-numeric parameter. Integrity tests separately reject payload changes by SHA256.

The generator and serializer have separate fingerprints. Numeric serialization
uses 17 significant digits and explicitly preserves negative zero. It verifies
binary64 bytes after rereading, including heterogeneous finite/infinite bound
arrays without first coercing all entries through decimal strings. The retained
R 4.6.1 arithmetic profile identifies the comparison rounding policy; the
artifact's R version and native build metadata identify the actual new oracle.
These labels do not imply universal bit parity across callback math libraries.

The focused witness in `../active_set/case_trace.json` strengthens that last
case to exact result/base-call bits and a digest of all raw arguments/values.
It stores 64 optimizer-level base calls rather than the large full BLAS log.
The public runtime capture records high-precision console values and public
DDOT/DAXPY arguments/results, checking the complete R result before accepting
the capture. No optimizer/BLAS implementation source or private memory is read.

```sh
python3 scripts/capture_reference_trace.py --prefix "$CONDA_PREFIX" \
  --case 'ill_conditioned_quadratic:11:mixed:1:FALSE' \
  --output /tmp/cache_capture --blas
python3 scripts/extract_active_set_trace.py --capture /tmp/cache_capture \
  --output /tmp/cache_witness.json
```

The adapter targets macOS and compiles outside the checkout with native CPU
flags. It preserves dynamic printf argument consumption and buffer checks;
13 formatter tests run before capture. The same case was independently replayed
in the retained R 4.6.1 environment, where every raw argument/value bit also
matches the new R 4.6.2 capture.
