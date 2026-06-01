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

The Rust test suite treats these sixty-eight cases as active parity checks for
the multi-dimensional supplied-gradient path, covering seventeen actual
optimizer-routed genes and fifty-one force-optimizer probes.

For local ignored-data sweeps, regenerate with `DESEQ_FIXTURE_GENES=all` and run
the DESeq test with `DESEQ_PARITY_SCAN=1`; normal test runs keep the strict
per-case assertions. `DESEQ_PARITY_SCAN_VERBOSE=1` prints actual and expected
parameters for failures, `DESEQ_TRACE_GENE=<gene>` emits objective-call traces,
and `DESEQ_BACKEND_TRACE=<level>` enables the native backend trace during the
Rust side of a probe.
