//! Stable byte encoding for benchmark result evidence.
pub fn result_fingerprint(fields: impl IntoIterator<Item = u64>, message: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in fields {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    for byte in message.bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    hash
}
