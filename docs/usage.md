# Usage

## Callbacks and bounds

`optim_lbfgsb` estimates gradients with bounded finite differences.
`optim_lbfgsb_with_gradient` accepts a separate gradient callback returning one
entry per parameter. Both callbacks receive user-scale parameters.

Use `Bounds::new(lower, upper)` or `Bounds::unbounded(n)`. Infinite limits are
supported; starts are projected into the box. Fixed coordinates require a
supplied gradient. Empty problems return successfully without calling callbacks.

For an analytic gradient and explicit arithmetic profile:

```rust
use rcompat_lbfgsb::{
    optim_lbfgsb_with_gradient_and_profile, ArithmeticProfile, Bounds, OptimControl,
};

let result = optim_lbfgsb_with_gradient_and_profile(
    vec![0.0],
    Bounds::unbounded(1),
    |x| (x[0] - 2.0).powi(2),
    |x| vec![2.0 * (x[0] - 2.0)],
    OptimControl::default_for_dimension(1),
    ArithmeticProfile::AppleSiliconOpenBlas,
).expect("finite quadratic");
assert!(result.is_success());
```

## Controls and results

Start with `OptimControl::default_for_dimension(n)` and change the needed fields.

| Control | Default | Meaning |
| --- | --- | --- |
| `fnscale` | `1` | Internal value is `user_value / fnscale`; negative selects maximization |
| `parscale` | All `1` | Internal coordinate is `user_par[i] / parscale[i]` |
| `ndeps` | All `1e-3` | Finite-difference steps in internal coordinates |
| `factr` | `1e7` | Relative-reduction threshold multiplier |
| `pgtol` | `0` | Projected-gradient tolerance in internal units |
| `maxit` | `100` | R-style iteration budget |
| `lmm` | `5` | Retained correction pairs |
| `trace`, `report` | `0`, `10` | Text diagnostics and reporting interval |

Scaling vectors must have length `n` and finite positive entries. `fnscale` must
be finite and nonzero; `factr` and `pgtol` must be finite and nonnegative; `lmm`
must be positive. `factr=0` can still accept a rounded zero reduction.
`pgtol=0` requires an exactly zero projected gradient for that stopping condition.
The iteration limit is checked after an accepted step, so `maxit=0` can take one step.

`OptimResult` returns `par` and `value` in user units, plus `counts`, `convergence`
and `message`. Counts exclude extra objective calls used for finite differences.
A returned `Ok` can have a nonzero convergence code: `0` is success, `1` is the
iteration limit and `52` is a search/direction failure. Check `is_success()`.
Invalid inputs, callback outputs and unrepresentable arithmetic return `OptimError`.

## Arithmetic profiles

`X86OpenBlas` and `AppleSiliconOpenBlas` model measured reference rounding.
Both work on any host and support nested/concurrent runs. Explicit selection
uses `optim_lbfgsb_with_profile` or `optim_lbfgsb_with_gradient_and_profile`.

The simpler entry points select `X86OpenBlas` by default and
`AppleSiliconOpenBlas` with the `arm-fma` feature. Explicit profiles ignore that
feature. Profiles affect optimizer arithmetic; callback math remains the caller's.

## Observers

`optim_lbfgsb_observed` and `optim_lbfgsb_with_gradient_observed` take an explicit
profile and an `OptimObserver`, including a `FnMut(OptimEvent)` closure:

```rust
use rcompat_lbfgsb::{optim_lbfgsb_observed, ArithmeticProfile, Bounds, OptimControl, OptimEvent};

let mut accepted = 0;
let result = optim_lbfgsb_observed(
    vec![0.0], Bounds::unbounded(1), |x| (x[0] - 2.0).powi(2),
    OptimControl::default_for_dimension(1), ArithmeticProfile::X86OpenBlas,
    |event| { if let OptimEvent::Accepted { .. } = event { accepted += 1; } },
).expect("finite quadratic");
assert!(result.is_success());
assert!(accepted > 0);
```

Events report the actual model, accepted steps, history, recovery and termination.
Diagnostic values use internal units; active/free counts describe the proposed
target. `Finished` follows successful result conversion. Evaluation/conversion
failures have distinct events; validation errors emit none; empty problems emit
only `Finished`. Observers run synchronously and add no numerical evaluations.
Logging and storage costs belong to the observer.

## Native builds and scientific example

Compile for the current CPU:

```sh
RUSTFLAGS='-C target-cpu=native' CARGO_TARGET_DIR=target/native cargo build --release
```

Run the independent negative-binomial likelihood/prior example without R:

```sh
cargo run --release --example deseq2
```

Optional arguments are the input JSON and its package-reference JSON.
See [validation](validation.md) for reference cohorts and accuracy limits.
