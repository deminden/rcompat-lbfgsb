# DESeq-Derived Real-Data Subset

This directory contains a small committed subset copied from the ignored
`data/` directory by `scripts/generate_r_fixtures.R`.

The source data was produced from a real DESeq workflow. These fixtures are
behavioral test inputs only: they contain selected numeric rows and generated
R `optim(..., method = "L-BFGS-B")` outputs, not DESeq2 or R implementation
source.

The optimizer cases in `optim_cases.json` use an independently written standard
negative-binomial GLM negative log-likelihood with fixed per-gene dispersion,
the committed design matrix, size factors, and selected count rows. The
objective omits beta-independent constants so normal Rust tests do not need an
R or DESeq runtime.

The Rust test suite treats these 68 cases as active parity checks for
the multi-dimensional supplied-gradient path, covering seventeen actual
optimizer-routed genes and fifty-one force-optimizer probes. The optimizer
outputs were regenerated with R 4.6.1 on 2026-07-22; all 68 cases currently
reproduce the saved parameters, values, counts, convergence codes, and messages
bit-for-bit on the pinned callback arithmetic.

For 0.3.0, the full-result replay requires Linux x86 and is mandatory in pinned
debug/release CI. Enable result-bit checks with `LBFGSB_EXACT_REFERENCE=1`.
Other hosts explicitly ignore that replay and run the separate portable
scientific test with the same 1e-8 parameter/value limits and convergence status;
counts and messages remain reported observations in that accuracy tier.

For broader local sweeps, regenerate with `DESEQ_FIXTURE_GENES=all` and run the
DESeq test with `DESEQ_PARITY_SCAN=1`. Optional verbose and trace controls are
documented alongside the test implementation.
