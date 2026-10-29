use super::*;
#[test]
fn negative_binomial_stress_reaches_the_independent_global_optimum() {
    // Each row has count == size_factor. Its negative log likelihood is
    // strictly convex in eta, with unique minimum eta=0. The first n rows
    // contain the identity design, so the unique coefficient optimum is x=0.
    // This checks accuracy independently of the R trajectory or its libm.
    let fixture = fixture();
    let mut executed = 0;
    for c in fixture["cases"].as_array().unwrap() {
        if c["family"] != "negative_binomial" {
            continue;
        }
        let p = Problem::new(c);
        assert_eq!(p.counts, p.sf);
        let lower = numbers(&c["lower"]);
        let upper = numbers(&c["upper"]);
        assert!(lower.iter().zip(&upper).all(|(&l, &u)| l < 0.0 && u > 0.0));
        let optimum = p.value(&vec![0.0; p.n]);
        for profile in [
            ArithmeticProfile::X86OpenBlas,
            ArithmeticProfile::AppleSiliconOpenBlas,
        ] {
            let mut control = OptimControl::default_for_dimension(p.n);
            control.fnscale = p.sign;
            control.parscale = numbers(&c["control"]["parscale"]);
            control.ndeps = numbers(&c["control"]["ndeps"]);
            control.lmm = c["control"]["lmm"].as_u64().unwrap() as usize;
            // Accuracy controls are intentionally separate from the unchanged
            // full R-result replay. Do not compare their counts or messages.
            control.maxit = 500;
            control.factr = 100.0;
            control.pgtol = 1e-7;
            let initial = numbers(&c["initial"]);
            let bounds = Bounds::new(lower.clone(), upper.clone()).unwrap();
            let f = |x: &[f64]| p.value(x);
            let result = if c["gradient"].as_bool().unwrap() {
                optim_lbfgsb_with_gradient_and_profile(
                    initial,
                    bounds,
                    f,
                    |x| p.gradient(x),
                    control,
                    profile,
                )
            } else {
                optim_lbfgsb_with_profile(initial, bounds, f, control, profile)
            }
            .unwrap();
            let id = c["id"].as_str().unwrap();
            assert_eq!(result.convergence, 0, "{id}: {profile:?}: {result:?}");
            assert_eq!(result.par.len(), p.n);
            for (i, &x) in result.par.iter().enumerate() {
                assert!(x.is_finite() && x >= lower[i] && x <= upper[i]);
                assert!(x.abs() <= 1e-5, "{id}: {profile:?}: parameter {i}: {x}");
            }
            assert!(result.value.is_finite());
            let excess = p.sign * (result.value - optimum);
            assert!(
                excess.abs() <= 1e-9 * (1.0 + optimum.abs()),
                "{id}: {profile:?}: objective excess {excess}"
            );
            executed += 1;
        }
    }
    assert_eq!(
        executed, 256,
        "all 128 NB cases must run under both profiles"
    );
}
