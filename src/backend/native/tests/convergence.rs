use super::*;

#[test]
fn relative_reduction_uses_signed_decrease_like_r() {
    assert!((relative_objective_reduction(10.0, 9.0) - 0.1).abs() <= 1e-15);
    assert!((relative_objective_reduction(10.0, 10.5) + (0.5 / 10.5)).abs() <= 1e-15);
    assert!((relative_objective_reduction(0.25, 0.5) + 0.25).abs() <= 1e-15);
}
