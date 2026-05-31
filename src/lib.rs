//! Clean-room Rust implementation of R-compatible
//! `stats::optim(..., method = "L-BFGS-B")` semantics.
//!
//! The crate focuses on R-style wrapper behavior: `fnscale`, `parscale`,
//! `ndeps`, bounds, optional gradients, typed errors, and R-like result fields.
//! It does not call R at runtime.
//!
//! ```
//! use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl};
//!
//! let result = optim_lbfgsb(
//!     vec![0.0],
//!     Bounds::new(vec![-10.0], vec![10.0])?,
//!     |p| (p[0] - 2.0).powi(2),
//!     OptimControl::default_for_dimension(1),
//! )?;
//!
//! assert!((result.par[0] - 2.0).abs() < 1e-5);
//! # Ok::<(), rcompat_lbfgsb::OptimError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

mod backend;
mod bounds;
mod compat;
mod control;
mod error;
mod finite_diff;
mod objective;
mod result;
mod scaling;

pub use bounds::Bounds;
pub use compat::{optim_lbfgsb, optim_lbfgsb_with_gradient};
pub use control::OptimControl;
pub use error::OptimError;
pub use result::{OptimCounts, OptimResult};
