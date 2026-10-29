use super::*;

pub(super) fn replay_objective_cases(full_result: bool) {
    let fixture = hard_fixture();
    let mut differences = support::MaxAbsoluteDifferences::default();
    let mut executed = 0;
    let mut failures = Vec::new();
    for contrast in &fixture.contrasts {
        for case in &contrast.cases {
            let result = optim_lbfgsb(
                case.initial_par.clone(),
                Bounds::new(case.lower.clone(), case.upper.clone()).unwrap(),
                |beta| {
                    nb_nll_without_constants(
                        beta,
                        &case.counts,
                        &contrast.design,
                        &contrast.size_factors,
                        case.dispersion,
                    )
                },
                control_from_case(&case.control),
            )
            .unwrap();
            let expected = case.result.to_result();
            differences.record(&result, &expected);
            let issues = if full_result {
                support::compare(
                    &result,
                    &expected,
                    support::Contract::Numerical {
                        parameters: 5e-3,
                        value: 1e-5,
                    },
                )
            } else {
                let comparison = support::compare_scientific(&result, &expected, 5e-3, 1e-5);
                if !comparison.bookkeeping_differences.is_empty() {
                    eprintln!("{}: {:?}", case.fixture, comparison.bookkeeping_differences);
                }
                comparison.failures
            };
            if !issues.is_empty() {
                failures.push(format!("{}: {}", case.fixture, issues.join("; ")));
            }
            executed += 1;
        }
    }
    differences.report("objective_only_glms");
    assert_eq!(executed, 48, "every hard-real case must run");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
