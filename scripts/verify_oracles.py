#!/usr/bin/env python3
"""Verify frozen oracle payloads and generation fingerprints without running R."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {'mixed_bounds': {'cases': 37, 'recipe': 'scripts/generate_mixed_bound_fixtures.R', 'payload_sha256': '052c5bdeae0bd6ec1a18d96a55337c3ea23cd4daeae804408003b1b9e43e70c5', 'field': 'cases', 'id': 'id', 'r_version': '4.6.1', 'platform': 'aarch64-apple-darwin20.0.0'}, 'line_search': {'cases': 6, 'recipe': 'scripts/generate_line_search_fixtures.R', 'payload_sha256': 'a46c9cd154d6aa49f5610bfad8b186b31e73eb7e989f405eadeee357e5226a7e', 'field': 'cases', 'id': 'id', 'r_version': '4.6.1', 'platform': 'aarch64-apple-darwin20.0.0'}, 'synthetic_v1': {'path': 'fixtures/synthetic_v1/cases.json', 'cases': 512, 'recipe': 'scripts/generate_stress_fixtures.R', 'payload_sha256': '2f0f7495326be3e871887ebf168f3718c15caf340ccf83b87869f02b2bba4fd8', 'field': 'cases', 'id': 'id', 'r_version': '4.6.2', 'platform': 'aarch64-apple-darwin27.2.0', 'serializer': 'scripts/oracle_json.R', 'native': True}, 'native_r462_hard': {'path': 'fixtures/native_r462/hard_real_reference.json', 'cases': 48, 'recipe': 'scripts/baseline_deseq2_r461.R', 'payload_sha256': '0d5752dc76241ac6aaf84328da44d77f79a4ccdcf442ead013e52b3bdb41e497', 'field': 'results', 'id': 'fixture', 'r_version': '4.6.2', 'platform': 'aarch64-apple-darwin27.2.0', 'serializer': 'scripts/oracle_json.R', 'native': True, 'input': 'fixtures/deseq_hard_real_subset/optim_cases.json'}, 'native_r462_airway': {'path': 'fixtures/native_r462/airway_reference.json', 'cases': 28, 'recipe': 'scripts/baseline_deseq2_r461.R', 'payload_sha256': 'ff8152e773975a3d2c1258ab058a3c83b8dba42cd4adaf38c53fa5a10024ca36', 'field': 'results', 'id': 'fixture', 'r_version': '4.6.2', 'platform': 'aarch64-apple-darwin27.2.0', 'serializer': 'scripts/oracle_json.R', 'native': True, 'input': 'fixtures/deseq_airway/cases.json'}}
CONFIGS["active_set_trace"] = {
    "path": "fixtures/active_set/case_trace.json", "cases": 64,
    "recipe": "scripts/extract_active_set_trace.py",
    "payload_sha256": "25f2b5eff34114ab65f264105a3d5263878c215d29f97d7123aa16c223792fa7",
    "field": "base_calls", "id": "ordinal", "r_version": "4.6.2",
    "platform": "aarch64-apple-darwin27.2.0", "native": True,
    "input": "fixtures/synthetic_v1/cases.json", "top_result": True, "exact_trace": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(name, path=None):
    config = CONFIGS[name]
    path = Path(path) if path else ROOT / config.get("path", f"fixtures/{name}/cases.json")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == config["payload_sha256"],
            f"{name}: frozen oracle payload changed; review and regenerate its fingerprint")
    fixture = json.loads(payload)
    require(fixture["generator"] == config["recipe"], f"{name}: unexpected recipe")
    recipe_hash = hashlib.sha256((ROOT / config["recipe"]).read_bytes()).hexdigest()
    require(fixture["generator_sha256"] == recipe_hash, f"{name}: recipe changed without oracle regeneration")
    rows=fixture[config["field"]]
    require(len(rows) == config["cases"], f"{name}: incomplete corpus")
    require(len({case[config["id"]] for case in rows}) == config["cases"], f"{name}: duplicate case")
    require(fixture["threads"] == "1", f"{name}: expected one BLAS thread")
    require(fixture["platform"] == config["platform"], f"{name}: unexpected platform")
    require(config["r_version"] in fixture["r_version"], f"{name}: unexpected R version")
    require(fixture["blas"] and fixture["session"], f"{name}: incomplete environment fingerprint")
    if config.get("serializer"):
        digest=hashlib.sha256((ROOT/config["serializer"]).read_bytes()).hexdigest()
        require(fixture["serializer_sha256"]==digest,f"{name}: serializer changed without regeneration")
    if config.get("input"):
        digest=hashlib.sha256((ROOT/config["input"]).read_bytes()).hexdigest()
        require(fixture["input_sha256"]==digest,f"{name}: source inputs changed without regeneration")
    if config.get("native"):
        for key in ["CFLAGS","CXXFLAGS","FFLAGS"]:
            require("-mcpu=native" in fixture["build_config"][key],f"{name}: native compilation fingerprint missing")
    if config.get("exact_trace"):
        require([row["ordinal"] for row in rows] == list(range(64)), "active-set callback order changed")
        require(fixture["raw_calls"] == 1472 and len(fixture["raw_callback_fnv1a64"]) == 16, "active-set raw trace contract changed")
        int(fixture["raw_callback_fnv1a64"], 16)
        for source, expected in fixture["capture"]["sources"].items():
            require(hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == expected, f"active-set capture source changed: {source}")
        require(fixture["capture"]["fixture_sha256"] == fixture["input_sha256"], "active-set source inputs disagree")
    return config["cases"]


ENVIRONMENT_SHA256 = "3c5556074e9cb45cd3e1502261411e3d4c2307445f1dba4cde671f706aa3dd20"
BLAS_ENVIRONMENT_SHA256 = "f4b64c9f7920df6248236056aad9e455c8944131508fc0083677444e0e424555"
RUNTIME_SNAPSHOTS = {
    "r461-macos-arm.json": "0d3c8be44bf6dc66ba3e2eb77854d4e05533e860dd94cee39160461cc2f30c20",
    "r462-macos-arm-native.json": "af622ce8207ed3bc61a33e68c5218b7f9c5a586daad0cc7fedb3629e3acd1d8d",
}


def verify_runtime_snapshots(directory=None):
    directory = Path(directory) if directory else ROOT / "fixtures/reference_environments"
    for name, expected in RUNTIME_SNAPSHOTS.items():
        payload = (directory / name).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == expected, f"runtime snapshot changed: {name}")
        snapshot = json.loads(payload)
        recipe = ROOT / snapshot["generator"]
        require(hashlib.sha256(recipe.read_bytes()).hexdigest() == snapshot["generator_sha256"],
                f"runtime capture recipe changed: {name}")
        require(snapshot["cpu"] == "Apple Silicon", f"unexpected runtime CPU: {name}")
        require(snapshot["threads"] == {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                f"unexpected runtime thread controls: {name}")
        require(snapshot["runtime"]["packages"]["DESeq2"]["version"] == "1.52.0",
                f"unexpected runtime DESeq2: {name}")
        require(len(snapshot["binary_sha256"]) >= 9, f"incomplete runtime binary hashes: {name}")
        require(len(snapshot["bootstrap_distribution"]) >= 98, f"incomplete runtime distribution: {name}")


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
        require(hashlib.sha256((ROOT / config[key]).read_bytes()).hexdigest() == config[key + "_sha256"],
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
    for name,digest in environment["recipes"].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f"native build recipe changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=list(CONFIGS))
    parser.add_argument("--fixture", type=Path, help="alternative payload to check against the frozen fingerprint")
    parser.add_argument("--blas-directory", type=Path, help="alternative DDOT corpus directory to verify")
    parser.add_argument("--runtime-directory", type=Path, help="alternative installed-oracle snapshot directory to verify")
    args = parser.parse_args()
    if args.fixture and not args.corpus:
        parser.error("--fixture requires --corpus")
    try:
        verify_reference_environment()
        verify_runtime_snapshots(args.runtime_directory)
        verify_blas_oracle(args.blas_directory)
        print("runtime snapshots: R 4.6.1 / native 4.6.2 binaries, distribution archives and capture recipe verified")
        print("arithmetic: 200 public DDOT calls; payload, recipes and reference environment verified")
        for name in [args.corpus] if args.corpus else CONFIGS:
            count = verify(name, args.fixture)
            print(f"{name}: {count} unique cases; payload, recipe SHA256 and reference metadata verified")
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
