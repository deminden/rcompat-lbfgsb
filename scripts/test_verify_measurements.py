#!/usr/bin/env python3
"""Exercise the actual allocation gate with incomplete and corrupted evidence."""
import copy
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class MeasurementGate(unittest.TestCase):
    def test_api_positive_control_and_corruptions(self):
        with (ROOT / "measurements/2026-10-29/api_options_after.csv").open() as stream:
            original = list(csv.DictReader(stream))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "altered.csv"
            for mutation in ["original", "missing", "duplicate", "result", "result_all_modes", "allocations", "exponentials", "stationary"]:
                with self.subTest(api_mutation=mutation):
                    rows = copy.deepcopy(original)
                    if mutation == "missing": rows.pop()
                    elif mutation == "duplicate": rows[-1] = rows[0]
                    elif mutation == "result": rows[0]["result_fingerprint"] = "0000000000000000"
                    elif mutation == "result_all_modes":
                        # Preserve within-run mode equality. The before/after
                        # comparison must still reject altered result bits.
                        for row in rows[:4]:
                            row["result_fingerprint"] = "0000000000000000"
                    elif mutation == "allocations": rows[0]["allocations"] = "999999"
                    elif mutation == "exponentials": rows[0]["exponentials"] = "1"
                    elif mutation == "stationary":
                        next(r for r in rows if r["stationary"] == "true")["iterations"] = "1"
                    with path.open("w") as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(original[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    result=subprocess.run([sys.executable,str(ROOT/"scripts/verify_api_measurements.py"),"--after",str(path)],capture_output=True,text=True)
                    self.assertEqual(result.returncode==0,mutation=="original",result.stderr)

    def test_positive_control_and_corruptions(self):
        with (ROOT / "measurements/2026-10-29/allocation_after.csv").open() as stream:
            original = list(csv.DictReader(stream))
        mutations = ["original", "missing", "duplicate", "result", "allocations", "raw_calls", "callback_bytes", "growth"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "altered.csv"
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    rows = copy.deepcopy(original)
                    if mutation == "missing": rows.pop()
                    elif mutation == "duplicate": rows[-1] = rows[0]
                    elif mutation == "result": rows[0]["result_fingerprint"] = "0000000000000000"
                    elif mutation == "allocations": rows[0]["allocations"] = "9999999"
                    elif mutation == "raw_calls": rows[0]["raw_objective_calls"] = "1"
                    elif mutation == "callback_bytes": rows[0]["callback_bytes"] = "0"
                    elif mutation == "growth":
                        # Preserve callback accounting and before/after savings,
                        # but make the fourfold-dimension growth implausible.
                        row = next(r for r in rows if r["dimension"] == "128" and r["lmm"] == "1" and r["route"] == "finite" and r["callback_work"] == "1" and r["profile"].startswith("r461-x86"))
                        row["bytes_allocated"] = str(int(row["callback_bytes"]) + 1)
                    with path.open("w") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(original[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_measurements.py"), "--after", str(path)], capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, mutation == "original", result.stderr)
                    if mutation == "growth":
                        self.assertIn("growth exceeds linear envelope", result.stderr)


if __name__ == "__main__":
    unittest.main()
