#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl, OptimError};

fn main() -> Result<(), OptimError> {
    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0])?,
        |p| (p[0] - 2.0).powi(2),
        OptimControl::default_for_dimension(1),
    )?;

    println!("{result:?}");
    Ok(())
}
