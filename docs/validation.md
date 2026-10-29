# Validation

Tests use committed inputs and reference results. Python 3 runs integrity and
measurement checks. R is needed only to regenerate references.

## Contracts and platform scope

| Contract | Required match |
| --- | --- |
| Exact | Parameter/value bits, counts, convergence and message |
| Numerical full result | Declared floating tolerances; exact counts, convergence and message |
| Scientific accuracy | Shape, finiteness, accuracy limits and matching convergence; counts/messages are observations |

Full replay requires the recorded callback math as well as the chosen optimizer
profile. Host-specific tests have explicit ignore reasons and run in matching
Apple Silicon CI jobs. Portable scientific checks still run on other hosts.
Checksums protect evidence; known-solution and replay tests check numerical behavior.

## Measured differences against R

Maximum absolute parameter difference is `max(abs(Rust.par[i] - R.par[i]))`
over every case and coordinate; objective difference is `max(abs(Rust.value - R.value))`.
The following observed maxima match in debug and CPU-native release on Apple
Silicon using `AppleSiliconOpenBlas` and the R 4.6.2 references. Values are rounded
to six significant digits.

| Reference suite | Results | Max abs parameter difference | Max abs objective difference |
| --- | ---: | ---: | ---: |
| Standard optimizer results | 59 | `0` | `0` |
| Mixed-bound models | 37 | `0` | `0` |
| Line-search recovery | 6 | `0` | `0` |
| Stress corpus | 512 | `1.22125e-15` | `1.26021e-28` |
| Supplied-gradient GLMs | 68 | `4.23237e-10` | `4.39968e-11` |
| Objective-only GLMs | 48 | `4.63329e-5` | `8.99176e-8` |

The 59 standard optimizer results belong to 63 artifacts including trace/error
fixtures. All rows also match counts, convergence and messages exactly. The
line-search suite additionally checks every callback point bit for bit.

Acceptance limits are separate from these observations: standard fixtures have
per-case limits; mixed-bound models use `1e-8`/`1e-10` for parameters/objective;
stress and supplied-gradient GLMs use `1e-8`/`1e-8`; objective-only GLMs use
`5e-3`/`1e-5`. Host-specific full replays require the recorded Apple Silicon
callback math. Portable scientific checks retain their declared accuracy limits.

Portable stress checks cover 384 polynomial full results and all 128 NB models
under both profiles. The NB known optimum is zero; limits are `1e-5` for
coefficients and `1e-9 * (1 + abs(optimal_value))` for objective error, using
`maxit=500`, `factr=100`, `pgtol=1e-7`.

The active-set witness also checks exact final results, 64 base calls and a
digest of 1,472 raw calls. Known-optimum, dense-BFGS/Newton, scaling/maximization,
numerical-error and observer tests provide independent coverage.

## DESeq2 precision

Public package captures use R 4.6.2 / DESeq2 1.52.0 on Apple Silicon with
single-threaded OpenBLAS 0.3.34 and native compilation. The independent
[likelihood/prior callback](../examples/support/deseq2_objective.rs) has these
observed maximum absolute differences against the package optimizer results:

| Cohort | Routed fits | Max abs coefficient difference | Max abs objective difference |
| --- | ---: | ---: | ---: |
| Hard real inputs | 46 | `1.39872e-3` | `2.81616e-7` |
| Airway/GSE52778 | 28 | `3.76263e-6` | `1.31095e-12` |

Debug and CPU-native release agree. The acceptance limits remain `0.0014` and
`1e-6` for hard coefficients/objectives, and `3.8e-6` and `1e-9` for airway.
`native_reference_precision` also requires matching convergence on macOS ARM;
counts/messages are observations. These measurements describe the recorded cohorts.

Reference tests print `max_abs_differences` when run with
`-- --nocapture --test-threads=1`. For example:

```sh
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native \
  cargo test --release --all-features --test native_reference_precision \
  -- --nocapture --test-threads=1
```

## Local checks

Run from the repository root:

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native \
  cargo test --release --all-features --all-targets
RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features
cargo +1.87.0 check --all-targets --all-features
cargo package
python3 scripts/verify_oracles.py
python3 scripts/test_verify_oracles.py
python3 scripts/check_profile_feature_invariance.py
```

Use `cargo package --allow-dirty` when checking uncommitted work. CI also covers
Linux/macOS, both feature settings, debug/release and the minimum Rust version.
Portable jobs run the backend and public API tests; scientific jobs run the R and
package comparisons, including pinned and CPU-native Apple Silicon builds.
Evidence integrity and corruption checks run once; live allocation checks run on
both platforms.

The oracle verifier pins eight corpora, 63 standard payloads, 200 DDOT outputs,
runtime metadata and generation recipes. Mutation tests check rejection of
corrupted evidence; stress tests also corrupt inputs to the actual replay process.
See [performance](performance.md) for allocation checks.

## Regeneration

Use R 4.6.2 with each generator's declared packages and start it with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The shared JSON writer preserves
binary64 values and signed zero, rereads outputs and replaces them atomically.
Its round-trip check is `Rscript --vanilla scripts/test_oracle_roundtrip.R`.
Review regenerated results before updating frozen fingerprints.

[Fixture inventory and sources](provenance.md) ·
[Package captures](../fixtures/native_r462/README.md) ·
[Runtime fingerprints](../fixtures/reference_environments)
