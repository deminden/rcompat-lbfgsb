#!/usr/bin/env python3
"""Freeze a small exact callback witness from a verified public runtime capture."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def callback_digest(calls):
    result = 0xcbf29ce484222325
    for call in calls:
        payload = struct.pack("<Q", len(call["x"]))
        payload += b"".join(struct.pack("<d", x) for x in [*call["x"], call["value"]])
        for byte in payload:
            result = ((result ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return f"{result:016x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.capture / "metadata.json").read_text())
    for name, expected in metadata["artifacts"].items():
        assert digest(args.capture / name) == expected, f"capture changed: {name}"
    for name, expected in metadata["sources"].items():
        assert digest(ROOT / name) == expected, f"capture recipe changed: {name}"
    reference = json.loads((args.capture / "reference.json").read_text())
    inputs = ROOT / "fixtures/synthetic_v1/cases.json"
    assert digest(inputs) == metadata["fixture_sha256"]
    case = next(c for c in json.loads(inputs.read_text())["cases"] if c["id"] == reference["id"])
    assert case["gradient"] is False and case["n"] == 11
    raw = reference["value_calls"]
    width = 2 * case["n"] + 1
    assert len(raw) == width * reference["result"]["counts"]["function"] == 1472
    assert not reference["gradient_calls"]
    witness = {"schema": 1, "identity": "active_set_cache_rounding", "id": case["id"],
        "arithmetic_profile": "apple-silicon-openblas-0.3.34", "r_version": reference["r_version"],
        "platform": reference["platform"], "blas": reference["blas"], "threads": reference["threads"],
        "session": reference["session"], "build_config": metadata["build_config"], "generator": "scripts/extract_active_set_trace.py",
        "generator_sha256": digest(Path(__file__)), "input_sha256": digest(inputs),
        "capture": metadata, "raw_calls": len(raw), "raw_callback_fnv1a64": callback_digest(raw),
        "result": reference["result"],
        "base_calls": [{"ordinal": i, **call} for i, call in enumerate(raw[::width])]}
    payload = json.dumps(witness, indent=2, allow_nan=False) + "\n"
    parsed = json.loads(payload)
    for before, after in zip(witness["base_calls"], parsed["base_calls"]):
        assert all(struct.pack("<d", a) == struct.pack("<d", b) for a, b in zip([*before["x"], before["value"]], [*after["x"], after["value"]]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(f"Wrote {len(witness['base_calls'])} exact base calls and a digest of all {len(raw)} raw calls")


if __name__ == "__main__":
    main()
