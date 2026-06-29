#!/usr/bin/env python3
"""Run the authority suite at -O1, -O2, and -O3 and print a result matrix."""

import csv
import os
import pathlib
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OPT_LEVELS = ("O1", "O2", "O3")


def load_oracle_domains():
    domains = []
    with (SCRIPT_DIR / "oracle.tsv").open(encoding="utf-8") as oracle:
        for line in oracle:
            line = line.strip()
            if line and not line.startswith("#"):
                domains.append(line.split("\t", 1)[0])
    return domains


def load_opt_policy(known_domains):
    policy = {}
    with (SCRIPT_DIR / "opt-policy.tsv").open(encoding="utf-8") as policy_file:
        for line in policy_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            domain, reason = line.split("\t", 1)
            if domain not in known_domains:
                raise SystemExit(f"unknown domain in opt-policy.tsv: {domain}")
            if domain in policy:
                raise SystemExit(f"duplicate domain in opt-policy.tsv: {domain}")
            policy[domain] = reason
    return policy


def load_results(path):
    with path.open(encoding="utf-8", newline="") as result_file:
        return {
            row["domain"]: row["result"]
            for row in csv.DictReader(result_file, delimiter="\t")
        }


def main():
    domains = load_oracle_domains()
    policy = load_opt_policy(set(domains))
    eligible = [domain for domain in domains if domain not in policy]
    tmp_root = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
    matrix_root = tmp_root / "capstone-authority-opt-matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)

    matrix = {
        domain: {
            opt: ("SKIP" if domain in policy else "NOT-RUN")
            for opt in OPT_LEVELS
        }
        for domain in domains
    }
    level_logs = {}

    for opt in OPT_LEVELS:
        level_dir = matrix_root / opt
        level_dir.mkdir(parents=True, exist_ok=True)
        results_path = level_dir / "results.tsv"
        driver_log = level_dir / "runner.log"
        if results_path.exists():
            results_path.unlink()

        child_env = os.environ.copy()
        child_env.update(
            {
                "DOMAIN_OPT_LEVEL": f"-{opt}",
                "SHARE_DIR": str(level_dir / "share"),
                "AUTHORITY_LOG": str(level_dir / "qemu.log"),
                "AUTHORITY_RESULTS_TSV": str(results_path),
                "AUTHORITY_ONLY": ",".join(eligible),
                "AUTHORITY_BOOT_TIMEOUT_SECONDS": child_env.get(
                    "AUTHORITY_BOOT_TIMEOUT_SECONDS", "45"
                ),
            }
        )
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run-authority-suite.py")],
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        driver_log.write_text(completed.stdout, encoding="utf-8")
        level_logs[opt] = driver_log

        if results_path.exists():
            observed = load_results(results_path)
            missing_state = "INFRA" if completed.returncode == 75 else "ERROR"
            for domain in eligible:
                matrix[domain][opt] = observed.get(domain, missing_state)
        else:
            state = "INFRA" if completed.returncode == 75 else "ERROR"
            for domain in eligible:
                matrix[domain][opt] = state

    width = max(len(domain) for domain in domains)
    print(f"{'DOMAIN'.ljust(width)}  " + "  ".join(opt.ljust(8) for opt in OPT_LEVELS))
    print("-" * (width + 32))
    for domain in domains:
        cells = "  ".join(matrix[domain][opt].ljust(8) for opt in OPT_LEVELS)
        print(f"{domain.ljust(width)}  {cells}")

    if policy:
        print("\nO0-only probes:")
        for domain in domains:
            if domain in policy:
                print(f"  {domain}: {policy[domain]}")

    print(f"\nartifacts: {matrix_root}")
    for opt in OPT_LEVELS:
        print(f"  {opt}: {level_logs[opt]}")

    failed = any(
        matrix[domain][opt] != "PASS"
        for domain in eligible
        for opt in OPT_LEVELS
    )
    if failed:
        print("__CAPSTONE_AUTHORITY_OPT_MATRIX_FAILED__", file=sys.stderr)
        return 1
    print("__CAPSTONE_AUTHORITY_OPT_MATRIX_PASSED__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
