use super::*;
#[test]
fn strict_stress_gate_rejects_corrupted_result_evidence() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temporary(std::path::PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let directory = Temporary(std::env::temp_dir().join(format!(
        "rcompat-stress-gate-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let original = fixture();
    let id = original["cases"][0]["id"].as_str().unwrap();
    let path = directory.0.join("oracle.json");
    for mutation in 0..10 {
        let mut data = original.clone();
        match mutation {
            0 => {}
            1 => data["cases"][0]["result"]["par"][0] = Value::from(123.0),
            2 => data["cases"][0]["result"]["value"] = Value::from(123.0),
            3 => data["cases"][0]["result"]["counts"]["function"] = Value::from(999),
            4 => data["cases"][0]["result"]["counts"]["gradient"] = Value::from(999),
            5 => data["cases"][0]["result"]["convergence"] = Value::from(52),
            6 => data["cases"][0]["result"]["message"] = Value::from("corrupted evidence"),
            7 => {
                data["cases"].as_array_mut().unwrap().remove(0);
            }
            8 => {
                data["cases"][0]["result"]["par"]
                    .as_array_mut()
                    .unwrap()
                    .clear();
            }
            _ => data["cases"][0]["result"]["par"][0] = Value::String("NaN".to_owned()),
        }
        std::fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "stress_corpus_matches_r",
                "--include-ignored",
                "--nocapture",
            ])
            .env("LBFGSB_STRESS_ORACLE", &path)
            .env("LBFGSB_STRESS_CASE", id)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            mutation == 0,
            "mutation {mutation}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
