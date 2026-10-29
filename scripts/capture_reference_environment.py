#!/usr/bin/env python3
"""Fingerprint an installed external R oracle through public runtime interfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(prefix):
    prefix = prefix.resolve(strict=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    r = prefix / "bin/R"
    rscript = prefix / "bin/Rscript"
    expression = r'''
packages <- c("DESeq2", "jsonlite", "Rcpp", "RcppArmadillo")
description <- lapply(packages, function(p) {
  list(version=as.character(packageVersion(p)),
       libraries=I(list.files(system.file("libs", package=p), full.names=TRUE)))
})
names(description) <- packages
cat(jsonlite::toJSON(list(
  r_version=R.version.string, platform=R.version$platform,
  r_home=R.home(), blas=unname(extSoftVersion()[["BLAS"]]),
  libraries=I(c(file.path(R.home("lib"), if(Sys.info()[["sysname"]]=="Darwin") "libR.dylib" else paste0("libR", .Platform$dynlib.ext)),
                system.file("libs", paste0("stats", .Platform$dynlib.ext), package="stats"))),
  session=I(capture.output(sessionInfo())), packages=description
), auto_unbox=TRUE, pretty=TRUE))
'''
    runtime = json.loads(subprocess.check_output(
        [str(rscript), "--vanilla", "-e", expression], env=environment, text=True,
    ))
    config = {}
    for name in ["CC", "CFLAGS", "CXX", "CXXFLAGS", "FC", "FFLAGS", "BLAS_LIBS", "LAPACK_LIBS"]:
        config[name] = subprocess.check_output(
            [str(r), "CMD", "config", name], env=environment, text=True,
        ).strip()
    compilers = {}
    for name in ["CC", "CXX", "FC"]:
        executable = shlex.split(config[name])[0]
        try:
            compilers[name] = subprocess.check_output(
                [executable, "--version"], env=environment, text=True, stderr=subprocess.STDOUT,
            ).splitlines()[0]
        except (OSError, subprocess.CalledProcessError):
            compilers[name] = "configured executable unavailable at capture time"
    paths = [str(r), str(rscript), str(prefix / "lib/R/bin/exec/R"), runtime["blas"], *runtime["libraries"]]
    for package in runtime["packages"].values():
        paths.extend(package["libraries"])
    binaries = {name: digest(name) for name in sorted(set(paths))}
    cpu = "Apple Silicon" if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64") else platform.machine()
    return {
        "generator": "scripts/capture_reference_environment.py",
        "generator_sha256": digest(__file__),
        "prefix": str(prefix), "system": platform.platform(), "cpu": cpu,
        "threads": {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        "runtime": runtime, "configured_build": config,
        "configured_compiler_versions": compilers,
        "compiler_scope": "R CMD config and currently installed compiler versions. These do not establish the original compiler of a prebuilt distribution archive.",
        "binary_sha256": binaries,
        "source_boundary": "Only runtime output, package metadata, configuration and binary hashes were read. No optimizer implementation source was inspected.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = capture(args.prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
