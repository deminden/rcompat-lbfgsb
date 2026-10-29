#!/usr/bin/env python3
"""Capture high-precision public R console/BLAS traces with result-bit verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new capture directory")
    parser.add_argument("--fixture", type=Path, default=ROOT / "fixtures/synthetic_v1/cases.json")
    parser.add_argument("--blas", action="store_true", help="also record public DDOT/DAXPY calls")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("this adapter targets the macOS reference environment")
    prefix, fixture = args.prefix.resolve(), args.fixture.resolve()
    cases = json.loads(fixture.read_text())["cases"]
    selected = [case for case in cases if case["id"] == args.case]
    if len(selected) != 1 or selected[0]["family"] != "ill_conditioned_quadratic":
        parser.error("select exactly one ill-conditioned quadratic reference case")
    source = ROOT / "scripts/trace_reference_runtime.c"
    recipe = ROOT / "scripts/trace_stress_case.R"
    library = prefix / "lib/libopenblas.0.dylib"
    compiler = shlex.split(subprocess.check_output([prefix / "bin/R", "CMD", "config", "CC"], text=True).strip())
    flags = ["-O3", "-mcpu=native", "-Wall", "-Wextra", "-Werror"]
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    with tempfile.TemporaryDirectory(prefix="rcompat-public-trace-") as temporary:
        temporary = Path(temporary)
        check, adapter = temporary / "format_check", temporary / "trace.dylib"
        subprocess.run(compiler + flags + ["-DRCOMPAT_FORMAT_TEST", str(source), "-o", str(check)], check=True)
        subprocess.run([check], check=True)
        subprocess.run(compiler + flags + ["-dynamiclib", str(source), str(library), f"-Wl,-rpath,{prefix / 'lib'}", "-o", str(adapter)], check=True)
        environment = dict(os.environ, R_HOME=str(prefix / "lib/R"),
            DYLD_LIBRARY_PATH=f"{prefix / 'lib/R/lib'}:{prefix / 'lib'}",
            DYLD_INSERT_LIBRARIES=str(adapter), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", ORACLE_TRACE_LEVEL="6")
        environment.pop("RCOMPAT_BLAS_TRACE", None)
        if args.blas:
            environment["RCOMPAT_BLAS_TRACE"] = str(output / "blas.jsonl")
        command = [str(prefix / "lib/R/bin/exec/R"), "--vanilla", "--slave",
            f"--file={recipe}", "--args", str(fixture), args.case, str(output / "reference.json")]
        with (output / "console.log").open("w") as console:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=console, stderr=subprocess.STDOUT, check=True)
    reference = json.loads((output / "reference.json").read_text())
    expected, actual = selected[0]["result"], reference["result"]
    for key in ["par", "value"]:
        x = expected[key] if isinstance(expected[key], list) else [expected[key]]
        y = actual[key] if isinstance(actual[key], list) else [actual[key]]
        assert len(x) == len(y) and all(struct.pack("<d", a) == struct.pack("<d", b) for a, b in zip(x, y))
    assert all(expected[key] == actual[key] for key in ["counts", "convergence", "message"])
    count = 0
    if args.blas:
        with (output / "blas.jsonl").open() as stream:
            for row in stream:
                record = json.loads(row)
                assert record["seq"] == count and record["fn"] in ["dot", "axpy"]
                count += 1
        assert count > 0
    artifacts = [path for path in output.iterdir() if path.is_file()]
    metadata = {"case": args.case, "r_version": reference["r_version"], "platform": reference["platform"], "prefix": str(prefix),
        "public_blas_calls": count, "raw_objective_calls": len(reference["value_calls"]),
        "cpu": "Apple Silicon" if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64") else platform.machine(),
        "compiler": subprocess.check_output(compiler[:1] + ["--version"], text=True).strip(),
        "flags": flags, "threads": 1, "library_sha256": digest(library),
        "r_library_sha256": digest(prefix / "lib/R/lib/libR.dylib"),
        "r_executable_sha256": digest(prefix / "lib/R/bin/exec/R"),
        "build_config": {key: subprocess.check_output([prefix / "bin/R", "CMD", "config", key], text=True).strip()
            for key in ["CC", "CFLAGS", "CXX", "CXXFLAGS", "FC", "FFLAGS"]},
        "sources": {str(path.relative_to(ROOT)): digest(path) for path in [source, recipe, Path(__file__).resolve(), ROOT / "scripts/oracle_json.R"]},
        "fixture_sha256": digest(fixture), "artifacts": {path.name: digest(path) for path in artifacts},
        "boundary": "public console formatting and DDOT/DAXPY arguments/results only; no optimizer/BLAS implementation source or private state is inspected"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Verified complete R result bits; {len(reference['value_calls'])} objective calls; {count} public BLAS calls")


if __name__ == "__main__":
    main()
