#![allow(missing_docs)]

use rcompat_lbfgsb::{optim_lbfgsb_with_gradient, Bounds, OptimControl, OptimError};

fn main() -> Result<(), OptimError> {
    let result = optim_lbfgsb_with_gradient(
        vec![-1.2, 1.0],
        Bounds::new(vec![-5.0, -5.0], vec![5.0, 5.0])?,
        |p| 100.0 * (p[1] - p[0] * p[0]).powi(2) + (1.0 - p[0]).powi(2),
        |p| {
            vec![
                -400.0 * p[0] * (p[1] - p[0] * p[0]) - 2.0 * (1.0 - p[0]),
                200.0 * (p[1] - p[0] * p[0]),
            ]
        },
        OptimControl::default_for_dimension(2),
    )?;

    println!("{result:?}");
    Ok(())
}
