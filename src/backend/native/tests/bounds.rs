use super::*;

#[test]
fn projected_gradient_norm_caps_components_by_bound_distance_like_r() {
    let norm = projected_gradient_norm(
        &[0.99, -1.99, 0.0, 2.0],
        &[-10.0, 8.0, 0.005, -3.0],
        &[-1.0, -2.0, f64::NEG_INFINITY, 2.0],
        &[1.0, 3.0, f64::INFINITY, 2.0],
    );

    assert!((norm - 0.01).abs() <= 1e-15);
}

#[test]
fn project_direction_keeps_free_quasi_newton_components() {
    let mut direction = vec![1.0, -2.0];
    project_direction(
        &[0.0, 0.0],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &mut direction,
    );
    assert_eq!(direction, vec![1.0, -2.0]);
}

#[test]
fn project_direction_blocks_outward_bound_components() {
    let mut direction = vec![-1.0, 2.0, 1.0];
    project_direction(
        &[0.0, 1.0, 3.0],
        &[0.0, 0.0, 3.0],
        &[5.0, 1.0, 3.0],
        &mut direction,
    );
    assert_eq!(direction, vec![0.0, 0.0, 0.0]);
}

#[test]
fn project_direction_blocks_outward_components_with_roundoff_tolerance() {
    let mut direction = vec![-1.0, 2.0];
    project_direction(
        &[4.0e-16, 1.0 - 4.0e-16],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &mut direction,
    );

    assert_eq!(direction, vec![0.0, 0.0]);
}

#[test]
fn subspace_free_set_requires_exact_bound_hit() {
    let lower = vec![0.0, 0.0, 0.0, 0.0];
    let upper = vec![1.0, 1.0, 1.0, 1.0];
    let point = vec![4.0e-16, 1.0 - 4.0e-16, 0.0, 1.0];

    assert_eq!(
        free_indices_with_activity(&point, &lower, &upper, BoundActivity::Exact),
        vec![0, 1]
    );
    assert_eq!(
        active_count_with_activity(&point, &lower, &upper, BoundActivity::Exact),
        2
    );
}
