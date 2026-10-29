# DESeq2 Hard Real Optimizer Subset

This directory contains a compact subset generated from the ignored
`data/lbfgsb_hard_real_2026-06-01/` bundle.

The source bundle was originally produced with R 4.4.3 and DESeq2 1.46.0. The
committed optimizer outputs were replayed on 2026-07-22 with the historical
Linux x86 reference stack: R 4.6.1, Bioconductor 3.23, and DESeq2 1.52.0.
The primary 0.3.0 reference is now native R 4.6.2 / DESeq2 1.52.0 on macOS ARM;
the original R 4.6.1 captures remain unchanged as regression data.

`deseq2_r461_nbinomWaldTest.json` is a second, same-version baseline produced
by the installed DESeq2 1.52.0 package under R 4.6.1. It records fits from the
public `DESeq2::nbinomWaldTest()` interface, including its coefficients,
convergence flag, iteration count, deviance, and the actual `stats::optim`
inputs and outputs. This captures wrapper initialization and integration
behavior that the objective-only replay below does not. Two force-optimizer
probes are included for diagnosis but are marked as not routed through
optimization in the source analysis.

The retained legacy-callback comparison replays the 46 routed fits from their captured
DESeq2 starting coefficients, using objective-only finite differences and the
full negative-binomial likelihood plus DESeq2's `N(0, 1000^2)` coefficient
prior. The prior includes both its normalization constant and its quadratic
term. This reduces the worst coefficient delta from `6.24` for the likelihood-
only objective to `0.0153` (about `408x`); 45 of 46 rows are within `5e-3`.
The remaining LCE3D case is the largest at `0.0153`, despite a maximum start
difference of `319.35` between the original saved replay and the public
DESeq2 wrapper. The original objective-only R parity fixtures remain unchanged
and continue to check the generic optimizer's existing R behavior.

The fixtures do not depend on DESeq2 at test time and do not copy DESeq2 or R
source. They store portable numeric inputs and outputs from a direct
objective-only `stats::optim(method = "L-BFGS-B")` replay and from the public
DESeq2 interface.

The Rust regression test keeps the original objective-only route separate from
the public-package comparison above. Its diagnostic scan currently reports
47 of 48 exact optimizer-count matches; the largest parameter delta is
`3.44e-4` and the largest objective delta is `9.56e-7`. One existing hard row
differs by one function and gradient evaluation while reaching a nearly
identical endpoint. These are macOS observations, not relaxed x86 parity checks.
The 0.3.0 full-result contract requires Linux x86 and checks all 48 cases,
including every result bit when `LBFGSB_EXACT_REFERENCE=1`. It passed the pinned
debug/release CI jobs at the preceding numerical revision. Every host also runs
a separate scientific check with the unchanged 5e-3 parameter/1e-5 objective
limits and convergence status. Unsupported full replays have explicit ignore
reasons; final-revision remote CI remains a publication prerequisite.

The current mathematical likelihood callback is tested separately against both
the old and new package captures. Across the 46 routed hard fits, its worst
coefficient error is about 0.00139872, preserving the measured 10.9x improvement
over the naive callback. See `../native_r462/README.md` for the primary native
reference, build commands and the additional 28 airway fits.

Selection is intentionally small: for each completed contrast, the generator
keeps six unique-gene hard rows, preferring the highest-ranked actual optimizer
row and the highest-ranked forced optimizer probe from `global_hardest_512.tsv`,
then filling from the ranked list.

After activating the pinned R 4.6.1 environment, regenerate the default six
cases per contrast with:

```sh
Rscript scripts/generate_hard_real_fixtures.R
```

With R 4.6.1 and DESeq2 1.52.0 installed, regenerate the public-package
baseline with:

```sh
Rscript scripts/baseline_deseq2_r461.R \
  fixtures/deseq_hard_real_subset/optim_cases.json \
  fixtures/deseq_hard_real_subset/deseq2_r461_nbinomWaldTest.json
```

Set `HARD_REAL_CASES_PER_CONTRAST` to change the compact subset size. The
separate `scripts/trace_hard_real_case.R` helper supports local path-drift
investigation against the ignored source bundle.
