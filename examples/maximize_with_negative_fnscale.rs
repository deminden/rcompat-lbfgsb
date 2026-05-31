#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb, Bounds, OptimControl, OptimError};

fn main() -> Result<(), OptimError> {
    let mut control = OptimControl::default_for_dimension(1);
    control.fnscale = -1.0;

    let result = optim_lbfgsb(
        vec![0.0],
        Bounds::new(vec![-10.0], vec![10.0])?,
        |p| -(p[0] - 3.0).powi(2),
        control,
    )?;

    println!("{result:?}");
    Ok(())
}
