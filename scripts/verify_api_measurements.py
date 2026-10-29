#!/usr/bin/env python3
"""Verify private callback prototypes, initial-stop allocation savings and coverage."""
import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DIRECTORY = ROOT / "measurements/2026-10-29"
PROFILES = ["r461-x86-openblas-0.3.32", "r461-macos-arm-openblas-0.3.34"]
IDENTITY = ["dimension", "lmm", "profile", "callback_work", "stationary", "mode"]


def load(path, owned_conversion):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        tuple(map(str, row)) for row in product([16, 128, 1024], [1, 10], PROFILES,
            [1, 16], ["false", "true"], ["separate", "joint", "fallible", "buffer"])
    }
    indexed = {tuple(row[k] for k in IDENTITY): row for row in rows}
    assert len(rows) == len(indexed) == 192 and set(indexed) == expected, "incomplete/duplicate API coverage"
    for key, row in indexed.items():
        n, _, _, work, stationary, mode = key
        calls = int(row["function_calls"])
        assert calls == int(row["gradient_calls"])
        assert int(row["exponentials"]) == int(n) * int(work) * calls * (2 if mode == "separate" else 1), "shared-work accounting changed"
        assert int(row["elapsed_ns"]) > 0 and int(row["allocations"]) > 0 and int(row["bytes_allocated"]) > 0
        assert re.fullmatch("[0-9a-f]{16}", row["result_fingerprint"])
        assert (calls == 1 and int(row["iterations"]) == 0) if stationary == "true" else (calls > 24 and int(row["iterations"]) == 25)
        baseline = indexed[(*key[:-1], "separate")]
        assert all(row[k] == baseline[k] for k in ["result_fingerprint", "function_calls", "gradient_calls", "iterations"]), "prototype changed optimizer behavior"
        removed = calls if mode == "buffer" and not owned_conversion else 0
        assert int(row["allocations"]) == int(baseline["allocations"]) - removed, "buffer allocation savings changed"
        assert int(row["bytes_allocated"]) == int(baseline["bytes_allocated"]) - 8 * int(n) * removed, "buffer byte savings changed"
    return indexed


def verify(before, after):
    before, after = load(before, False), load(after, True)
    for key, row in after.items():
        old = before[key]
        assert all(row[k] == old[k] for k in ["result_fingerprint", "function_calls", "gradient_calls", "iterations", "exponentials"]), "initial-stop optimization changed behavior"
        if row["stationary"] == "true":
            assert int(row["allocations"]) < int(old["allocations"])
            assert int(row["bytes_allocated"]) < int(old["bytes_allocated"])
        else:
            removed = int(row["gradient_calls"]) if row["mode"] != "buffer" else 0
            assert int(row["allocations"]) == int(old["allocations"]) - removed, "owned-gradient allocation savings changed"
            assert int(row["bytes_allocated"]) == int(old["bytes_allocated"]) - 8 * int(row["dimension"]) * removed, "owned-gradient byte savings changed"
    print("192 callback configurations preserve complete-result fingerprints and accounting")
    print("96 stationary solves avoid history setup; owned gradients save one allocation per evaluation")


def verify_fingerprints():
    metadata = json.loads((DIRECTORY / "api_environment.json").read_text())
    for name, digest in metadata["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"API measurement input/evidence changed: {name}"
    for name, digest in metadata["production_after"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"API prototype production source changed: {name}; rerun measurements"
    for phase in ["before", "after"]:
        portable, native = load(DIRECTORY / f"api_options_{phase}.csv", phase == "after"), load(DIRECTORY / f"api_options_native_{phase}.csv", phase == "after")
        for key, row in portable.items():
            assert row["result_fingerprint"] == native[key]["result_fingerprint"], "native compilation changed a prototype result"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=DIRECTORY / "api_options_before.csv")
    parser.add_argument("--after", type=Path, default=DIRECTORY / "api_options_after.csv")
    args = parser.parse_args()
    verify_fingerprints()
    verify(args.before, args.after)


if __name__ == "__main__":
    main()
