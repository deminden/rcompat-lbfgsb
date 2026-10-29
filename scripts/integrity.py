"""Small shared helpers for frozen evidence and source fingerprints."""
import hashlib


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hashes(root, entries, description):
    for name, expected in entries.items():
        require(sha256_file(root / name) == expected, f"{description}: {name}")
