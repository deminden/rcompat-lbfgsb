use crate::error::OptimError;

pub(crate) fn finite_difference_gradient<F>(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    ndeps: &[f64],
    mut objective: F,
) -> Result<Vec<f64>, OptimError>
where
    F: FnMut(&[f64]) -> Result<f64, OptimError>,
{
    let mut gradient = vec![0.0; x.len()];

    for index in 0..x.len() {
        let h = ndeps[index];
        let forward_room = if upper[index].is_finite() {
            (upper[index] - x[index]).max(0.0)
        } else {
            h
        };
        let backward_room = if lower[index].is_finite() {
            (x[index] - lower[index]).max(0.0)
        } else {
            h
        };

        let forward_step = h.min(forward_room);
        let backward_step = h.min(backward_room);

        let denominator = forward_step + backward_step;
        if denominator > 0.0 {
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            plus[index] += forward_step;
            minus[index] -= backward_step;
            let f_plus = objective(&plus)?;
            let f_minus = objective(&minus)?;
            gradient[index] = (f_plus - f_minus) / denominator;
        } else {
            gradient[index] = 0.0;
        }
    }

    Ok(gradient)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approximates_quadratic_gradient() {
        let gradient = finite_difference_gradient(
            &[3.0],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &[1e-5],
            |x| Ok((x[0] - 2.0).powi(2)),
        )
        .unwrap();
        assert!((gradient[0] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn uses_forward_difference_at_lower_bound() {
        let gradient = finite_difference_gradient(&[0.0], &[0.0], &[10.0], &[1e-5], |x| {
            Ok((x[0] - 2.0).powi(2))
        })
        .unwrap();
        assert!((gradient[0] + 4.0).abs() < 1e-4);
    }

    #[test]
    fn uses_asymmetric_stencil_near_upper_bound() {
        let gradient = finite_difference_gradient(&[0.99995], &[-1.0], &[1.0], &[1e-4], |x| {
            Ok((x[0] - 0.25).powi(2))
        })
        .unwrap();
        assert!((gradient[0] - 1.49985).abs() < 1e-12);
    }

    #[test]
    fn uses_asymmetric_stencil_near_lower_bound() {
        let gradient = finite_difference_gradient(&[-0.99995], &[-1.0], &[1.0], &[1e-4], |x| {
            Ok((x[0] + 0.25).powi(2))
        })
        .unwrap();
        assert!((gradient[0] + 1.49985).abs() < 1e-12);
    }

    #[test]
    fn fixed_parameter_has_zero_gradient() {
        let gradient = finite_difference_gradient(&[1.0], &[1.0], &[1.0], &[1e-5], |x| {
            Ok((x[0] - 2.0).powi(2))
        })
        .unwrap();
        assert_eq!(gradient, vec![0.0]);
    }

    #[test]
    fn repeats_base_value_for_each_one_sided_coordinate() {
        let mut calls = Vec::new();
        let gradient =
            finite_difference_gradient(&[0.0, 1.0], &[0.0, -1.0], &[2.0, 1.0], &[0.1, 0.1], |x| {
                calls.push(x.to_vec());
                Ok(x[0] + 2.0 * x[1])
            })
            .unwrap();

        assert!((gradient[0] - 1.0).abs() < 1e-12);
        assert!((gradient[1] - 2.0).abs() < 1e-12);
        assert_eq!(
            calls,
            vec![
                vec![0.1, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
                vec![0.0, 0.9],
            ]
        );
    }
}
