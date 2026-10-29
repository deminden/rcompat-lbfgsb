#![allow(missing_docs)]
use support::fixtures::{RecordedControl as HardControl, RecordedResult as HardResult};
mod data;
#[path = "../support/mod.rs"]
mod support;
use data::*;
mod objective;
use objective::*;
mod replay;
use replay::replay_objective_cases;

use rcompat_lbfgsb::{Bounds, OptimControl};
use serde::Deserialize;

#[test]
fn deseq_hard_real_fixture_is_consistent() {
    let fixture = hard_fixture();
    assert_eq!(fixture.fixture, "deseq_hard_real_2026_06_01_subset");
    assert_eq!(fixture.contrasts.len(), 8);

    for contrast in &fixture.contrasts {
        assert_eq!(
            contrast.samples,
            contrast.design.len(),
            "{}",
            contrast.contrast
        );
        assert_eq!(
            contrast.samples,
            contrast.size_factors.len(),
            "{}",
            contrast.contrast
        );
        assert_eq!(
            contrast.coefficients,
            contrast.coefficient_names.len(),
            "{}",
            contrast.contrast
        );
        for row in &contrast.design {
            assert_eq!(contrast.coefficients, row.len(), "{}", contrast.contrast);
        }
        assert_eq!(contrast.cases.len(), 6, "{}", contrast.contrast);
        for case in &contrast.cases {
            assert!(
                !case.gradient_supplied,
                "{} should replay DESeq2's objective-only fallback",
                case.fixture
            );
            assert_eq!(case.counts.len(), contrast.samples, "{}", case.fixture);
            assert_eq!(
                case.initial_par.len(),
                contrast.coefficients,
                "{}",
                case.fixture
            );
            assert_eq!(case.lower, vec![-30.0; contrast.coefficients]);
            assert_eq!(case.upper, vec![30.0; contrast.coefficients]);
            assert_eq!(
                case.result.par.len(),
                contrast.coefficients,
                "{}",
                case.fixture
            );
            assert!(case.result.value.is_finite(), "{}", case.fixture);
        }
    }
}

#[test]
#[cfg_attr(
    not(all(target_arch = "aarch64", target_os = "macos")),
    ignore = "full NB replay requires the recorded Apple Silicon callback math; mandatory in pinned CI"
)]
fn deseq_hard_real_objective_only_optim_cases_match_r() {
    replay_objective_cases(true);
}

#[test]
fn deseq_hard_real_objective_cases_have_portable_scientific_accuracy() {
    replay_objective_cases(false);
}

use support::optim_reference as optim_lbfgsb;
