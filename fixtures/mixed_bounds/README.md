# Mixed/infinite-bound black-box oracle

`cases.json` contains 37 independent scalar quadratic/linear optim calls, generated
with R 4.6.1 on macOS ARM. The embedded session, BLAS path, thread setting and
SHA256 of `scripts/generate_mixed_bound_fixtures.R` identify the environment and
recipe. The generator verifies every numeric field's binary64 serialization
before replacing the artifact. No optimizer source is inspected or copied.

Regenerate from the repository root:

```
OPENBLAS_NUM_THREADS=1 /path/to/Rscript scripts/generate_mixed_bound_fixtures.R
```

The generator requires jsonlite and the `shasum` command. Start R with the
thread environment set, rather than changing BLAS initialization afterward.
Run `python3 scripts/verify_mixed_bound_fixture.py` to check recipe provenance
without installing R. The four plateau/unrepresentable-target cases declare an
exact-bit endpoint and initial-call contract.

The Rust test explicitly selects the ARM arithmetic profile. It checks exact
counts, convergence codes and messages, parameter error <=1e-8, value error
<=1e-10, and the initial callback stencil/normalized trial to <=1e-12. These are
numerical endpoint contracts, not universal bit-exact callback claims. Reference
first-call prefixes are retained up to 80 calls. Cases cover dimensions 1/2/4/13,
finite/mixed/infinite bounds, analytic/numerical gradients, shallow curvature,
maxit=0, linear cap expansion, factr=0 rounded plateaus, and targets whose
displacement is unrepresentable at the starting magnitude.
