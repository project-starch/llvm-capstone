#!/usr/bin/env python3
"""Export complete A1 matrices; retain failed attempts without counting them."""

import argparse
import hashlib
import json
import pathlib
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(campaigns):
    accepted, attempts = {}, []
    for campaign in campaigns:
        for original in json.loads((campaign / "verdicts.json").read_text()):
            row = dict(original)
            mode, case = row["mode"], row["case"]
            if mode not in (0, 2) or case not in range(14, 36):
                raise ValueError("non-A1 case in matrix")
            run = campaign / f"mode-{mode}-case-{case}"
            text = (run / "serial.log").read_text(errors="replace")
            for key, path in (
                ("log_sha256", run / "serial.log"),
                ("domain_sha256", run / "share/security.dom"),
                ("trace_sha256", run / "share/trace.bin"),
            ):
                if row[key] != digest(path):
                    raise ValueError(f"changed evidence: {path}")
            row["log"] = f"{campaign.name}/{run.name}/serial.log"
            attempts.append(row)
            if not row["passed"]:
                continue
            if (mode, case) in accepted:
                raise ValueError("duplicate accepted case")
            stage = f"Print = Scalar(0x{0xff25000000000000 | case:x})"
            expected_fault = mode == 2 and 15 <= case <= 34
            faults = re.findall(
                r"domain halted by capability fault: cause = (\d+), pc = (0x[0-9a-f]+)",
                text,
            )
            if stage not in text:
                raise ValueError("missing setup marker")
            register = case in (23, 24, 33, 34)
            if (
                register
                and f"Print = Scalar(0x{0xff26000000000000 | case:x})" not in text
            ):
                raise ValueError("missing register transition marker")
            if expected_fault:
                sites = re.findall(
                    r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+),",
                    text.split(stage, 1)[1],
                )
                site = (2 if register else 0) + (case - 15) % 2
                if (
                    row["expected"] != "fault"
                    or len(faults) != 1
                    or int(faults[0][0]) not in (24, 25)
                    or len(sites) <= site
                    or int(faults[0][1], 16) != int(sites[site], 16)
                    or row["cause"] != int(faults[0][0])
                    or int(row["pc"], 16) != int(faults[0][1], 16)
                    or int(row["expected_pc"], 16) != int(sites[site], 16)
                    or "FF2 return=42044" in text
                ):
                    raise ValueError("fault oracle mismatch")
            elif (
                row["expected"] != "completed"
                or row["runner_exit"] != 0
                or faults
                or "FF2_SECURITY_DONE" not in text
            ):
                raise ValueError("completion oracle mismatch")
            accepted[mode, case] = row
    if set(accepted) != {(m, c) for m in (0, 2) for c in range(14, 36)}:
        raise ValueError("incomplete 44-execution A1 matrix")
    if len({r["domain_sha256"] for r in accepted.values()}) != 1:
        raise ValueError("mixed domain binaries")
    return {
        "schema": "capstone-alias-scatter-v1",
        "protected_stale_accesses": 20,
        "no_revoke_stale_accesses": 20,
        "valid_control_executions": 4,
        "accepted": [accepted[k] for k in sorted(accepted)],
        "failed_attempts": [r for r in attempts if not r["passed"]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("campaigns", type=pathlib.Path, nargs="+")
    args = parser.parse_args()
    result = collect(args.campaigns)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
