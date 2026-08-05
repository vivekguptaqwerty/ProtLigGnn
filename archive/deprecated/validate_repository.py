import sys
import subprocess
from pathlib import Path
import hashlib
import csv

# Expected dataset index hashes from BENCHMARK_SPECIFICATION.md
EXPECTED_HASHES = {
    "INDEX_general_PL.2020R1.lst": "35bacd30ce3731704f7adf60dff62df554ec9418f3f12d4411eb8baed3194b3d",
    "INDEX_general_NL.2020R1.lst": "3e84feb77ad1d7323dde41c8c72bfe1ec2fca11c3d200c4749c6b1715a19071c",
    "INDEX_general_PN.2020R1.lst": "42bd364b2da062a46184a86f3844ba29218c037393e9d5f50915b308be279c71",
    "INDEX_general_PP.2020R1.lst": "5a42fb9c2f8c637e9fb35ad7cf145be5bc87ee98e74c277208faa95d788ecfe5"
}

def check_file_sha256(path: Path, expected_sha: str) -> bool:
    if not path.exists():
        print(f"  -> FAIL: {path.name} does not exist.")
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    if digest == expected_sha:
        print(f"  -> PASS: {path.name} hash matches.")
        return True
    else:
        print(f"  -> FAIL: {path.name} hash mismatch (got {digest}, expected {expected_sha}).")
        return False

def main():
    print("==================================================")
    print("      ProtLigGNN Repository Pre-Run Validation    ")
    print("==================================================")
    
    passed = True
    
    # 1. Environment Verification
    print("[*] Verifying official software stack...")
    try:
        res = subprocess.run(
            [sys.executable, "verify_environment.py"],
            capture_output=True,
            text=True
        )
        if res.returncode == 0:
            print("  -> PASS: verify_environment.py completed successfully.")
        else:
            print("  -> FAIL: verify_environment.py failed:")
            print(res.stdout)
            print(res.stderr)
            passed = False
    except Exception as e:
        print(f"  -> FAIL: environment check process error: {e}")
        passed = False

    # 2. Dataset Hashes Verification
    print("[*] Verifying PDBbind dataset index hashes...")
    data_dir = Path("data/pdbbind2020")
    for name, expected_sha in EXPECTED_HASHES.items():
        path = data_dir / "index" / name
        if not check_file_sha256(path, expected_sha):
            passed = False

    # 3. Registry & Config Verification
    print("[*] Verifying configuration and registry...")
    config_path = Path("benchmark_config.yaml")
    if config_path.exists():
        print("  -> PASS: benchmark_config.yaml is present.")
    else:
        print("  -> FAIL: benchmark_config.yaml is missing.")
        passed = False
        
    registry_path = Path("experiments/registry.csv")
    if registry_path.exists():
        try:
            with registry_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                if "Run ID" in header and "Config Hash" in header:
                    print("  -> PASS: registry.csv exists and contains required fields.")
                else:
                    print("  -> FAIL: registry.csv header is missing required fields.")
                    passed = False
        except Exception as e:
            print(f"  -> FAIL: registry.csv parse error: {e}")
            passed = False
    else:
        print("  -> PASS: registry.csv does not exist yet (will be initialized).")

    # 4. Documentation Check
    print("[*] Verifying master documentation files...")
    required_docs = ["BRAIN.md", "ROADMAP.md", "BENCHMARK_SPECIFICATION.md", "OFFICIAL_ENVIRONMENT.md"]
    for doc in required_docs:
        if Path(doc).exists():
            print(f"  -> PASS: {doc} is present.")
        else:
            print(f"  -> FAIL: {doc} is missing.")
            passed = False

    print("==================================================")
    if passed:
        print("REPOSITORY VALIDATION: PASS")
        sys.exit(0)
    else:
        print("REPOSITORY VALIDATION: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
