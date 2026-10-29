#!/usr/bin/env python3
"""Check allocation probe coverage, result preservation and dimension growth."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
from integrity import sha256_file, verify_hashes
import re

ROOT = Path(__file__).resolve().parent.parent
IDENTITY = ["dimension", "lmm", "route", "callback_work", "profile"]
PROFILES = ["x86-openblas-0.3.32", "apple-silicon-openblas-0.3.34"]


def load(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        tuple(map(str, (n, m, route, work, profile)))
        for n, m, route, work, profile in product(
            [32, 128, 512, 2048], [1, 5, 10, 20],
            ["finite", "mixed", "unbounded", "finite_difference"], [1, 16], PROFILES
        ) if route != "finite_difference" or n <= 128
    }
    indexed = {tuple(row[key] for key in IDENTITY): row for row in rows}
    assert len(indexed) == len(rows) == len(expected) == 224, "incomplete or duplicate measurement set"
    assert set(indexed) == expected, "unexpected measurement configuration"
    for row in rows:
        assert re.fullmatch(r"[0-9a-f]{16}", row["result_fingerprint"]), "missing result fingerprint"
        assert int(row["convergence"]) == 1, "fixed iteration budget was not exercised"
        assert 0 <= int(row["callback_allocations"]) < int(row["allocations"])
        assert 0 <= int(row["callback_bytes"]) < int(row["bytes_allocated"])
        assert 0 < int(row["objective_ns"]) + int(row["gradient_ns"]) < int(row["elapsed_ns"])
        functions = int(row["optimizer_function_calls"])
        gradients = int(row["optimizer_gradient_calls"])
        assert functions == gradients > 24, "incomplete fixed-budget solve"
        if row["route"] == "finite_difference":
            assert int(row["raw_gradient_calls"]) == int(row["callback_allocations"]) == 0
            assert int(row["raw_objective_calls"]) == functions + 2 * int(row["dimension"]) * gradients
        else:
            assert int(row["raw_objective_calls"]) == functions
            assert int(row["raw_gradient_calls"]) == gradients == int(row["callback_allocations"])
            assert int(row["callback_bytes"]) == 8 * int(row["dimension"]) * gradients
    return indexed


def verify_archives():
    directory = ROOT / "measurements/2026-10-29"
    metadata = json.loads((directory / "environment.json").read_text())
    verify_hashes(ROOT, metadata["files"], "recorded measurement input/evidence changed")
    assert sha256_file(ROOT / "src/backend/native/compact.rs") == metadata["compact_after_sha256"], "compact source changed: rerun the allocation baseline comparison"
    verify_hashes(ROOT, metadata["production_after"], "allocation production source changed; rerun measurements")
    after, native = load(directory / "allocation_after.csv"), load(directory / "allocation_native.csv")
    owned = load(directory / "allocation_owned.csv")
    for key, row in after.items():
        assert row["result_fingerprint"] == native[key]["result_fingerprint"], "native compilation changed a recorded result"
        current=owned[key]
        assert all(row[k]==current[k] for k in ["result_fingerprint","raw_objective_calls","raw_gradient_calls","optimizer_function_calls","optimizer_gradient_calls"]), "owned-gradient conversion changed behavior"
        removed=int(row["raw_gradient_calls"])
        assert int(current["allocations"])==int(row["allocations"])-removed, "owned-gradient allocation savings changed"
        assert int(current["bytes_allocated"])==int(row["bytes_allocated"])-8*int(row["dimension"])*removed, "owned-gradient byte savings changed"


def verify(before, after):
    before, after = load(before), load(after)
    for key, row in after.items():
        old = before[key]
        for field in ["result_fingerprint", "raw_objective_calls", "raw_gradient_calls", "optimizer_function_calls", "optimizer_gradient_calls", "convergence"]:
            assert row[field] == old[field], f"behavior changed: {key}, {field}"
        assert int(row["allocations"]) < int(old["allocations"]), f"allocations did not decrease: {key}"
        assert int(row["bytes_allocated"]) < int(old["bytes_allocated"]), f"allocated bytes did not decrease: {key}"
    # Timing is deliberately not a pass/fail threshold. The scalar-storage
    # identity is tested independently. These cumulative allocated-byte bounds
    # additionally reject dimension-quadratic temporary storage on this workload.
    for m, route, work, profile in product([1, 5, 10, 20], ["finite", "mixed", "unbounded"], [1, 16], PROFILES):
        for low, high in [(128, 512), (512, 2048)]:
            a = after[tuple(map(str, (low, m, route, work, profile)))]
            b = after[tuple(map(str, (high, m, route, work, profile)))]
            assert int(b["bytes_allocated"]) < 6 * int(a["bytes_allocated"]), "temporary allocation growth exceeds linear envelope"
    print("224 complete-result fingerprints preserved; every allocation count/byte total reduced")
    print("96 fourfold-dimension comparisons satisfy the cumulative allocated-byte growth envelope")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "measurements/2026-10-29"
    parser.add_argument("--before", type=Path, default=default / "allocation_before.csv")
    parser.add_argument("--after", type=Path, default=default / "allocation_owned.csv")
    args = parser.parse_args()
    verify_archives()
    verify(args.before, args.after)


if __name__ == "__main__":
    main()
