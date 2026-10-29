use super::*;
pub(super) fn active_set_trace_issues(
    actual: &OptimResult,
    calls: &[Value],
    witness: &Value,
) -> Vec<String> {
    let mut issues = support::compare(
        actual,
        &support::recorded_result(&witness["result"]),
        support::Contract::Exact,
    );
    let expected_calls = witness["raw_calls"].as_u64().unwrap() as usize;
    if calls.len() != expected_calls {
        issues.push(format!(
            "active-set raw calls: {} != {expected_calls}",
            calls.len()
        ));
        return issues;
    }
    let mut digest = 0xcbf29ce484222325_u64;
    let mut add = |byte: u8| {
        digest = (digest ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    };
    for call in calls {
        let x = call["x"].as_array().unwrap();
        for byte in (x.len() as u64).to_le_bytes() {
            add(byte);
        }
        for value in x
            .iter()
            .map(|v| v.as_f64().unwrap())
            .chain(std::iter::once(call["value"].as_f64().unwrap()))
        {
            for byte in value.to_bits().to_le_bytes() {
                add(byte);
            }
        }
    }
    if format!("{digest:016x}") != witness["raw_callback_fnv1a64"].as_str().unwrap() {
        issues.push("active-set raw callback argument/value digest differs".to_owned());
    }
    let bases = witness["base_calls"].as_array().unwrap();
    assert_eq!(bases.len(), 64);
    assert_eq!(calls.len(), 23 * bases.len());
    for (ordinal, expected) in bases.iter().enumerate() {
        assert_eq!(expected["ordinal"].as_u64().unwrap() as usize, ordinal);
        let actual = &calls[23 * ordinal];
        let x = numbers(&actual["x"]);
        let reference = numbers(&expected["x"]);
        if x.len() != reference.len()
            || x.iter()
                .zip(&reference)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            issues.push(format!(
                "active-set base callback {ordinal} differs in parameter bits"
            ));
        }
        if actual["value"].as_f64().unwrap().to_bits()
            != expected["value"].as_f64().unwrap().to_bits()
        {
            issues.push(format!(
                "active-set base callback {ordinal} differs in value bits"
            ));
        }
    }
    issues
}
