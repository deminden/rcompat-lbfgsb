# Changelog

## 0.3.0

- Use the compact limited-memory backend for every nonempty problem, with
  retained buffers and owned-gradient reuse to reduce allocations.
- Correct stopping, recovery, trial normalization and reference rounding;
  reject invalid scaling, collapsed bounds and unrepresentable differences.
- Add explicit arithmetic profiles and structured observer events.
- Regenerate references with R 4.6.2; add DESeq2 1.52.0 precision captures,
  a self-contained 512-case stress corpus and exact search/active-set witnesses.
- Separate exact/full-result parity from portable scientific accuracy, with
  integrity, corruption, allocation and feature-isolation checks. Report observed
  maximum parameter/objective differences separately from acceptance limits.
- Remove experimental solvers, heuristic control overrides, private-data tools
  and redundant captures; require complete scientific replay.
- Split backend, test and benchmark modules; share fixture/integrity helpers.
- Use Rust 2024 with Rust 1.87 minimum; condense usage, validation and performance
  guides and update the scientific example to load canonical package references.

## 0.2.2 - 2026-10-14

- Add optional ARM arithmetic, an independent NB/prior example and scientific precision checks.

## 0.2.1 - 2026-07-22

- Refresh optimizer fixture metadata and callback traces.

## 0.2.0 - 2026-07-20

- Add the compact Cauchy/subspace model, More–Thuente search and scientific parity checks.
