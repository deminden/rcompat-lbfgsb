//! Observer contracts are checked against externally observable callback behavior.
use rcompat_lbfgsb::*;
use std::cell::RefCell;

mod errors;
mod events;
mod replay;
use replay::replay;

mod controls;
