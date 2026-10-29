#!/usr/bin/env python3
"""Corruption checks for the actual command-line oracle gate."""
import copy
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verify_oracles import CONFIGS, ROOT


class FrozenOracleGate(unittest.TestCase):
    def test_runtime_snapshot_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            for name in ["r461-macos-arm.json", "r462-macos-arm-native.json"]:
                shutil.copyfile(ROOT / "fixtures/reference_environments" / name, destination / name)
            command = [sys.executable, str(ROOT / "scripts/verify_oracles.py"), "--runtime-directory", str(destination)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            for name in ["r461-macos-arm.json", "r462-macos-arm-native.json"]:
                path = destination / name
                original = path.read_bytes()
                data = json.loads(original)
                data["binary_sha256"][next(iter(data["binary_sha256"]))] = "0" * 64
                path.write_text(json.dumps(data))
                process = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn("runtime snapshot changed", process.stderr)
                path.write_bytes(original)

    def test_blas_payload_and_environment_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            for mutation in [None, "payload", "environment"]:
                for name in ["arm_dot.csv", "environment.json"]:
                    shutil.copyfile(ROOT / "fixtures/arithmetic" / name, destination / name)
                if mutation == "payload":
                    path = destination / "arm_dot.csv"
                    path.write_bytes(path.read_bytes().replace(b"0000000000000000", b"0000000000000001", 1))
                elif mutation == "environment":
                    path = destination / "environment.json"
                    data = json.loads(path.read_text())
                    data["threads"] = 2
                    path.write_text(json.dumps(data))
                process = subprocess.run([sys.executable, str(ROOT / "scripts/verify_oracles.py"), "--blas-directory", str(destination)], capture_output=True, text=True)
                self.assertEqual(process.returncode == 0, mutation is None, process.stderr)
                if mutation:
                    self.assertIn("BLAS", process.stderr)

    def gate(self, name, path):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_oracles.py"), "--corpus", name, "--fixture", str(path)],
            capture_output=True, text=True, check=False,
        )

    def test_originals_and_mutations(self):
        for name,config in CONFIGS.items():
            source = ROOT / config.get("path", f"fixtures/{name}/cases.json")
            self.assertEqual(self.gate(name, source).returncode, 0)
            original = json.loads(source.read_text())
            def result(f):
                if config.get("top_result"):
                    return f["result"]
                row=f[config["field"]][0]
                return row["result"] if config["field"]=="cases" else row["optimizer_result"]
            def mutate_counts(f):
                counts=result(f)["counts"]
                counts[0 if isinstance(counts,list) else "function"]=999
            mutations = {
                "parameter": lambda f: result(f)["par"].__setitem__(0, 999.0),
                "objective": lambda f: result(f).__setitem__("value", 999.0),
                "counts": mutate_counts,
                "status": lambda f: result(f).__setitem__("convergence", 999),
                "message": lambda f: result(f).__setitem__("message", "corrupted"),
                "missing-case": lambda f: f[config["field"]].pop(),
                "profile": lambda f: f.__setitem__("platform", "wrong-platform"),
                "recipe": lambda f: f.__setitem__("generator_sha256", "0"*64),
            }
            with tempfile.TemporaryDirectory() as folder:
                for field, mutate in mutations.items():
                    with self.subTest(corpus=name, mutation=field):
                        fixture = copy.deepcopy(original)
                        mutate(fixture)
                        path = Path(folder) / "corrupted.json"
                        path.write_text(json.dumps(fixture))
                        process = self.gate(name, path)
                        self.assertNotEqual(process.returncode, 0)
                        self.assertIn("payload changed", process.stderr)


if __name__ == "__main__":
    unittest.main()
