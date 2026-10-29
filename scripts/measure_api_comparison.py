#!/usr/bin/env python3
"""Measure both callback-allocation phases with the same host and toolchain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from verify_api_measurements import DIRECTORY, ROOT, verify, verify_fingerprints


def run(directory):
    # First validate the retained experiment and the current numerical sources.
    # A new source revision needs a reviewed baseline, rather than a stale patch.
    verify_fingerprints()
    metadata = json.loads((DIRECTORY / "api_environment.json").read_text())
    for phase in ["before", "after"]:
        checkout = directory / phase
        checkout.mkdir()
        for name in ["Cargo.toml", "Cargo.lock", "README.md"]:
            if (ROOT / name).exists():
                shutil.copy2(ROOT / name, checkout / name)
        for name in ["src", "benches"]:
            shutil.copytree(ROOT / name, checkout / name)
        if phase == "before":
            subprocess.run([
                "git", "apply", "--reverse", "--whitespace=nowarn",
                str(DIRECTORY / "callback_allocation.patch"),
            ], cwd=checkout, check=True)
        for name, expected in metadata[f"production_{phase}"].items():
            actual = hashlib.sha256((checkout / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{phase}: unexpected benchmark source: {name}")
        # Each phase has a fresh target directory. Copied source timestamps must
        # never let Cargo reuse a binary from the other phase.
        environment = dict(os.environ, CARGO_TARGET_DIR=str(checkout / "target"))
        environment.setdefault("LBFGSB_API_SAMPLES", "3")
        with (directory / f"{phase}.csv").open("w") as output:
            subprocess.run([
                "cargo", "bench", "--offline", "--bench", "api_options",
            ], cwd=checkout, env=environment, stdout=output, check=True)
    verify(directory / "before.csv", directory / "after.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory retaining the two builds and CSV files")
    args = parser.parse_args()
    if args.output:
        directory = args.output.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        run(directory)
    else:
        with tempfile.TemporaryDirectory(prefix="rcompat-api-comparison-") as temporary:
            run(Path(temporary))


if __name__ == "__main__":
    main()
