#!/usr/bin/env python3
"""Verify frozen oracle payloads and generation fingerprints without running R."""
import argparse
import hashlib
import json
from pathlib import Path
from integrity import require, sha256_file

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {'mixed_bounds': {'r_version': '4.6.2',
                  'platform': 'aarch64-apple-darwin27.2.0',
                  'cases': 37,
                  'recipe': 'scripts/generate_mixed_bound_fixtures.R',
                  'field': 'cases',
                  'id': 'id',
                  'payload_sha256': '2935825e11a4b93ce452408ca92232df3e47372c93fd3b9c2f8b341f47a42816'},
 'line_search': {'r_version': '4.6.2',
                 'platform': 'aarch64-apple-darwin27.2.0',
                 'cases': 6,
                 'recipe': 'scripts/generate_line_search_fixtures.R',
                 'field': 'cases',
                 'id': 'id',
                 'payload_sha256': '2a9463f2277445a6ec4aac3e5286a738b8dfccc8dd3e5e035bb797e5c88ad87c'},
 'synthetic_v1': {'r_version': '4.6.2',
                  'platform': 'aarch64-apple-darwin27.2.0',
                  'cases': 512,
                  'recipe': 'scripts/generate_stress_fixtures.R',
                  'field': 'cases',
                  'id': 'id',
                  'serializer': 'scripts/oracle_json.R',
                  'native': True,
                  'payload_sha256': 'bea70a22de3640c103e14ad3f840a197fc5169e0a3498f2dfdc3aa33945fe273'},
 'real_glm': {'r_version': '4.6.2',
              'platform': 'aarch64-apple-darwin27.2.0',
              'path': 'fixtures/deseq_real_subset/optim_cases.json',
              'cases': 68,
              'recipe': 'scripts/regenerate_scientific_fixtures.R',
              'field': 'cases',
              'id': 'fixture',
              'payload_sha256': '9aac1fb9e0cd59292934d64cf8ce35a41f2e0b4f06666341b4603d3e420ebe88'},
 'hard_glm': {'r_version': '4.6.2',
              'platform': 'aarch64-apple-darwin27.2.0',
              'path': 'fixtures/deseq_hard_real_subset/optim_cases.json',
              'cases': 48,
              'recipe': 'scripts/regenerate_scientific_fixtures.R',
              'field': 'contrasts',
              'nested': 'cases',
              'id': 'fixture',
              'payload_sha256': '3a8e5eb3ecef8e9e0bd3316a04a4691534e28a17c3fcb0758efb659f78b0368c'},
 'native_r462_hard': {'r_version': '4.6.2',
                      'platform': 'aarch64-apple-darwin27.2.0',
                      'path': 'fixtures/native_r462/hard_real_reference.json',
                      'cases': 48,
                      'recipe': 'scripts/capture_deseq2_reference.R',
                      'field': 'results',
                      'id': 'fixture',
                      'serializer': 'scripts/oracle_json.R',
                      'native': True,
                      'input': 'fixtures/deseq_hard_real_subset/optim_cases.json',
                      'payload_sha256': '7caba7aa185ae21bdc03aaaff7a37f02189880a0a276d4a5f359f33246b1d5f0'},
 'native_r462_airway': {'r_version': '4.6.2',
                        'platform': 'aarch64-apple-darwin27.2.0',
                        'path': 'fixtures/native_r462/airway_reference.json',
                        'cases': 28,
                        'recipe': 'scripts/capture_deseq2_reference.R',
                        'field': 'results',
                        'id': 'fixture',
                        'serializer': 'scripts/oracle_json.R',
                        'native': True,
                        'input': 'fixtures/deseq_airway/cases.json',
                        'input_recipe': 'scripts/generate_airway_fixtures.R',
                        'payload_sha256': '313950bc5ef80bfe73cef04a0878405462ad00db67a6abb1176dd4687f6b8b14'},
 'active_set_trace': {'r_version': '4.6.2',
                      'platform': 'aarch64-apple-darwin27.2.0',
                      'path': 'fixtures/active_set/case_trace.json',
                      'cases': 64,
                      'recipe': 'scripts/extract_active_set_trace.py',
                      'field': 'base_calls',
                      'id': 'ordinal',
                      'native': True,
                      'input': 'fixtures/synthetic_v1/cases.json',
                      'top_result': True,
                      'exact_trace': True,
                      'payload_sha256': '0cb8d06d9319fe47c8df417d44e9dbdbca96e14991bb02887dea8e2e13687f85'}}


def verify(name, path=None):
    config = CONFIGS[name]
    path = Path(path) if path else ROOT / config.get("path", f"fixtures/{name}/cases.json")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == config["payload_sha256"],
            f"{name}: frozen oracle payload changed; review and regenerate its fingerprint")
    fixture = json.loads(payload)
    require(fixture["generator"] == config["recipe"], f"{name}: unexpected recipe")
    recipe_hash = sha256_file(ROOT / config["recipe"])
    require(fixture["generator_sha256"] == recipe_hash, f"{name}: recipe changed without oracle regeneration")
    rows=fixture[config["field"]]
    if config.get("nested"):
        rows = [case for group in rows for case in group[config["nested"]]]
    require(len(rows) == config["cases"], f"{name}: incomplete corpus")
    require(len({case[config["id"]] for case in rows}) == config["cases"], f"{name}: duplicate case")
    require(fixture["threads"] == "1", f"{name}: expected one BLAS thread")
    require(fixture["platform"] == config["platform"], f"{name}: unexpected platform")
    require(config["r_version"] in fixture["r_version"], f"{name}: unexpected R version")
    require(fixture["blas"] and fixture["session"], f"{name}: incomplete environment fingerprint")
    if config.get("serializer"):
        digest=sha256_file(ROOT/config["serializer"])
        require(fixture["serializer_sha256"]==digest,f"{name}: serializer changed without regeneration")
    if config.get("input"):
        digest=sha256_file(ROOT/config["input"])
        require(fixture["input_sha256"]==digest,f"{name}: source inputs changed without regeneration")
    if config.get("input_recipe"):
        inputs = json.loads((ROOT / config["input"]).read_text())
        require(inputs["generator"] == config["input_recipe"], f"{name}: unexpected input recipe")
        require(inputs["generator_sha256"] == sha256_file(ROOT / config["input_recipe"]),
                f"{name}: input recipe changed without regeneration")
    if config.get("native"):
        for key in ["CFLAGS","CXXFLAGS","FFLAGS"]:
            require("-mcpu=native" in fixture["build_config"][key],f"{name}: native compilation fingerprint missing")
    if config.get("exact_trace"):
        require([row["ordinal"] for row in rows] == list(range(64)), "active-set callback order changed")
        require(fixture["raw_calls"] == 1472 and len(fixture["raw_callback_fnv1a64"]) == 16, "active-set raw trace contract changed")
        int(fixture["raw_callback_fnv1a64"], 16)
        for source, expected in fixture["capture"]["sources"].items():
            require(sha256_file(ROOT / source) == expected, f"active-set capture source changed: {source}")
        require(fixture["capture"]["fixture_sha256"] == fixture["input_sha256"], "active-set source inputs disagree")
    return config["cases"]


ENVIRONMENT_SHA256 = "3595d4321826623da76b45c0e22d36029f1804f6d22e8a4f1defe925c07f61da"
BLAS_ENVIRONMENT_SHA256 = "f4b64c9f7920df6248236056aad9e455c8944131508fc0083677444e0e424555"
RUNTIME_SNAPSHOTS = {
    "r462-macos-arm-native.json": "b10117fb27d5ee4f8155068d742bb350a5ba946321598bc81f5ceaa1217a8966",
}


def verify_runtime_snapshots(directory=None):
    directory = Path(directory) if directory else ROOT / "fixtures/reference_environments"
    for name, expected in RUNTIME_SNAPSHOTS.items():
        payload = (directory / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, f"runtime snapshot changed: {name}")
        snapshot = json.loads(payload)
        recipe = ROOT / snapshot["generator"]
        require(sha256_file(recipe) == snapshot["generator_sha256"],
                f"runtime capture recipe changed: {name}")
        require(snapshot["cpu"] == "Apple Silicon", f"unexpected runtime CPU: {name}")
        require(snapshot["threads"] == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                f"unexpected runtime thread controls: {name}")
        require(snapshot["runtime"]["packages"]["DESeq2"]["version"] == "1.52.0",
                f"unexpected runtime DESeq2: {name}")
        require(len(snapshot["binary_sha256"]) >= 9, f"incomplete runtime binary hashes: {name}")
        require("4.6.2" in snapshot["runtime"]["r_version"], f"unexpected runtime R version: {name}")


STANDARD_MANIFEST_SHA256 = "01cda2ddbc7b822ba42d119d4bdf2c62a6a6dc7ebfb23e827f996db5473a558b"


def verify_standard_oracles(directory=None):
    directory = Path(directory) if directory else ROOT / "fixtures/r_optim_lbfgsb"
    payload = (directory / "manifest.json").read_bytes()
    require(hashlib.sha256(payload).hexdigest() == STANDARD_MANIFEST_SHA256, "standard fixture manifest changed")
    manifest = json.loads(payload)
    require("4.6.2" in manifest["r_version"], "unexpected standard R version")
    require(len(manifest["payloads"]) == 63, "incomplete standard fixtures")
    for name, expected in manifest["payloads"].items():
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, f"standard fixture payload changed: {name}")
        fixture = json.loads((directory / name).read_bytes())
        require("4.6.2" in fixture["r_version"], f"unexpected standard fixture R version: {name}")
    for name, expected in manifest["sources"].items():
        require(sha256_file(ROOT / name) == expected, f"standard fixture recipe changed: {name}")


def verify_blas_oracle(directory=None):
    directory = Path(directory) if directory else ROOT / "fixtures/arithmetic"
    payload = (directory / "arm_dot.csv").read_bytes()
    metadata = (directory / "environment.json").read_bytes()
    require(hashlib.sha256(metadata).hexdigest() == BLAS_ENVIRONMENT_SHA256,
            "BLAS environment fingerprint changed")
    config = json.loads(metadata)
    require(hashlib.sha256(payload).hexdigest() == config["payload_sha256"], "BLAS payload changed")
    require(config["cases"] == 200 and config["threads"] == 1, "BLAS corpus contract changed")
    require(config["library_config"] == "OpenBLAS 0.3.34 NO_AFFINITY USE_OPENMP VORTEX MAX_THREADS=128",
            "unexpected BLAS arithmetic environment")
    for key in ["generator", "source"]:
        require(sha256_file(ROOT / config[key]) == config[key + "_sha256"],
                f"BLAS {key} changed without regeneration")
    lines = payload.decode("ascii").splitlines()
    require(lines[0] == "length,seed,result_bits" and len(lines) == 201, "BLAS corpus is incomplete")
    require(len({line.rsplit(",", 1)[0] for line in lines[1:]}) == 200, "duplicate BLAS case")

def verify_reference_environment():
    path=ROOT/"fixtures/native_r462/environment.json"
    payload=path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==ENVIRONMENT_SHA256,"native reference environment fingerprint changed")
    environment=json.loads(payload)
    require("4.6.2" in environment["runtime"]["r_version"],"unexpected native R version")
    require(environment["runtime"]["deseq2_version"]=="1.52.0","unexpected native DESeq2 version")
    require(environment["cpu"]=="Apple Silicon","unexpected native reference CPU")
    for key in ["CFLAGS","CXXFLAGS","FFLAGS"]:
        require("-mcpu=native" in environment["build_config"][key],f"native reference {key} missing")
    require(environment["source_sha256"] == "c297040f6392b1dac6e5c0eb398422cfdaedf540ae2205edb9ad0d10cb10ee00",
            "native R source archive fingerprint changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=list(CONFIGS))
    parser.add_argument("--fixture", type=Path, help="alternative payload to check against the frozen fingerprint")
    parser.add_argument("--standard-directory", type=Path, help="alternative standard fixture directory to verify")
    parser.add_argument("--blas-directory", type=Path, help="alternative DDOT corpus directory to verify")
    parser.add_argument("--runtime-directory", type=Path, help="alternative installed-oracle snapshot directory to verify")
    args = parser.parse_args()
    if args.fixture and not args.corpus:
        parser.error("--fixture requires --corpus")
    try:
        verify_standard_oracles(args.standard_directory)
        print("standard fixtures: 63 current-R payloads and split recipes verified")
        verify_reference_environment()
        verify_runtime_snapshots(args.runtime_directory)
        verify_blas_oracle(args.blas_directory)
        print("runtime snapshots: R 4.6.2 binaries and package versions and capture recipe verified")
        print("arithmetic: 200 public DDOT calls; payload, recipes and reference environment verified")
        for name in [args.corpus] if args.corpus else CONFIGS:
            count = verify(name, args.fixture)
            print(f"{name}: {count} unique cases; payload, recipe SHA256 and reference metadata verified")
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
