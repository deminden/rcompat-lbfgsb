# Provenance

`rcompat-lbfgsb` is independently written and MIT-licensed. Implementation inputs
are public mathematical literature, public interfaces, original probes and
black-box input/output observations. R/GPL optimizer source is not read, copied,
translated, paraphrased or used as an implementation input.

R runs externally to capture references. The crate does not invoke, link or
embed R/BLAS, and packages no external numerical binaries. Runtime/compiler
fingerprints document the capture environment, not a required user installation.

## Mathematical sources

- Byrd, Lu, Nocedal and Zhu: [A Limited Memory Algorithm for Bound Constrained Optimization](https://doi.org/10.1137/0916069).
- Zhu et al.: [Algorithm 778: L-BFGS-B](https://doi.org/10.1145/279232.279236).
- Moré and Thuente: [Line search](https://doi.org/10.1145/192115.192132).
- Independent NB likelihood/prior callback: [Loader's saddlepoint formulas](https://people.math.ethz.ch/~maechler/doc/CLoader-dbinom-2002.pdf)
  and [NIST DLMF psi series](https://dlmf.nist.gov/5.7#E4); code appendices and
  R/DESeq2 implementation sources are not implementation inputs.

## Fixture inventory

Paths below are relative to `fixtures/`. Every replay input is committed.
Current optimizer/package captures use R 4.6.2 on Apple Silicon; package captures
use DESeq2 1.52.0. Artifacts record their actual runtime and generation hashes.

| Directory | Contents | Generator/capture in `scripts/` |
| --- | --- | --- |
| `r_optim_lbfgsb` | 63 control, bound and trace artifacts | `generate_r_fixtures.R` and `r_optim_fixtures/` |
| `deseq_real_subset` | 68 supplied-gradient NB GLMs with independent input tables | `regenerate_scientific_fixtures.R` |
| `deseq_hard_real_subset` | 48 objective-only NB GLMs | `regenerate_scientific_fixtures.R` |
| `deseq_airway` | 28 fallback-fit inputs from public airway/GSE52778 counts | `generate_airway_fixtures.R` |
| `mixed_bounds` | 37 quadratic/linear/plateau models | `generate_mixed_bound_fixtures.R` |
| `line_search` | Six recovery scenarios and callback traces | `generate_line_search_fixtures.R` |
| `synthetic_v1` | 512 quadratic, NB and double-well models | `generate_stress_fixtures.R` |
| `native_r462` | Public DESeq2 captures for hard and airway inputs | `capture_deseq2_reference.R` |
| `active_set` | Exact stress-case witness | `capture_reference_trace.py`, `extract_active_set_trace.py` |
| `arithmetic` | 200 public DDOT results | `generate_blas_oracle.py`, `probe_reference_blas.c` |
| `reference_environments` | Runtime, package, compiler and binary fingerprints | `capture_reference_environment.py` |

Fixture READMEs give suite-specific contracts and regeneration commands.
The airway dataset's publication and selection details are in its
[fixture guide](../fixtures/deseq_airway/README.md).

## Probe boundary

`probe_reference_blas.c` and `probe_reference_triangular.c` call public DDOT/DTRSV
interfaces and compare independently written mathematical reductions/solves.
`trace_reference_runtime.c` forwards public stdio/BLAS calls, improves printed
double precision and records DDOT/DAXPY arguments/results. It uses Apple's
[public interposing interface](https://github.com/apple-oss-distributions/dyld/blob/main/include/mach-o/dyld-interposing.h)
and R's [public printing boundary](https://stat.ethz.ch/CRAN/doc/manuals/r-devel/R-exts.html#Printing).
No private optimizer state or implementation source is inspected.

The capture runner checks 13 formatter/argument-preservation cases and complete
R result bits. The retained witness has 64 base calls and a digest of 1,472 raw
calls. Payloads, recipes, inputs and runtime binaries have separate fingerprints.

## Evidence and licensing

Numeric writers check binary64 round trips, including signed zero, before atomic
replacement. Frozen hashes and corruption tests protect artifacts; independent
numerical tests validate behavior. Review new reference outputs before accepting
updated hashes. See [validation](validation.md) and
[measurement evidence](performance.md).

The library forbids unsafe Rust and has no runtime dependencies. Development
uses `serde`, `serde_json` and `stats_alloc`; inspect Cargo metadata for licenses
of the resolved dependency set. Review licenses when updating dependencies or
reference material.
