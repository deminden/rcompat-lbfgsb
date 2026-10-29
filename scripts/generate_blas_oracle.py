#!/usr/bin/env python3
"""Capture public DDOT results without consulting library implementation sources."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True, help="native R reference environment")
    parser.add_argument("--output", type=Path, required=True, help="new directory for both artifacts")
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    source = ROOT / "scripts/probe_reference_blas.c"
    library = prefix / "lib/libopenblas.0.dylib"
    compiler = shlex.split(subprocess.check_output([prefix / "bin/R", "CMD", "config", "CC"], text=True).strip())
    flags = ["-O3", "-mcpu=native"]
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="rcompat-public-blas-") as temporary:
        executable = Path(temporary) / "probe"
        subprocess.run(compiler + flags + [str(source), str(library), f"-Wl,-rpath,{prefix / 'lib'}", "-o", str(executable)], check=True, env=environment)
        payload = subprocess.check_output([executable], env=environment)
    lines = payload.decode("ascii").splitlines()
    assert lines[0] == "length,seed,result_bits" and len(lines) == 201
    assert len(set(line.rsplit(",", 1)[0] for line in lines[1:])) == 200
    # This is a public version/configuration query, not implementation inspection.
    loaded = ctypes.CDLL(str(library))
    loaded.openblas_get_config.restype = ctypes.c_char_p
    metadata = {
        "identity": "macos-arm-openblas-0.3.34-ddot",
        "cases": 200,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "generator": "scripts/generate_blas_oracle.py",
        "generator_sha256": digest(Path(__file__)),
        "source": "scripts/probe_reference_blas.c",
        "source_sha256": digest(source),
        "library": str(library),
        "library_sha256": digest(library),
        "library_config": loaded.openblas_get_config().decode(),
        "compiler": subprocess.check_output(compiler[:1] + ["--version"], text=True).strip(),
        "compile_flags": flags,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": "Apple Silicon" if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64") else platform.machine(),
        "threads": 1,
        "command": "python3 scripts/generate_blas_oracle.py --prefix " + shlex.quote(str(prefix)) + " --output NEW_DIRECTORY",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arm_dot.csv").write_bytes(payload)
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Captured 200 independent public DDOT results and environment hashes")


if __name__ == "__main__":
    main()
