#!/usr/bin/env python3
"""Recheck and export compact CHERI/Capstone replay memory comparisons."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

PORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT / "host/memory"))
from measure import observations


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("campaign", type=Path, nargs="+")
    a = p.parse_args()
    reference = json.loads((a.reference / "manifest.json").read_text())
    if reference["status"] != "complete":
        raise ValueError("incomplete reference")
    workloads = [w[0] for w in reference["workloads"]]
    for workload in workloads:
        for name in ("recorded.bin", "commands.bin"):
            path = a.reference / "recordings" / workload / name
            if digest(path) != reference["fingerprints"][str(path)]:
                raise ValueError("native reference changed")
    rows = []
    records = []
    for arm in ("spatial", "sublet"):
        for workload in workloads:
            runs = [
                r
                for r in reference["results"]
                if r["arm"] == arm
                and r["workload"] == workload
                and r["status"] == "passed"
            ]
            if len(runs) != reference["repetitions"]:
                raise ValueError("incomplete Capstone repetitions")
            checksums = []
            for r in runs:
                path = a.reference / r["directory"] / "capstone.bin"
                checksum = digest(path)
                if checksum != r["output_sha256"]:
                    raise ValueError("Capstone output changed")
                checksums.append(checksum)
                stats, _ = observations(
                    path,
                    a.reference / "recordings" / workload / "recorded.bin",
                    0 if arm == "spatial" else 2,
                )
            if len(set(checksums)) != 1:
                raise ValueError("Capstone repetitions differ")
            rows.append(
                dict(
                    arm="capstone-" + arm,
                    workload=workload,
                    repetitions=len(runs),
                    payload_carved_bytes=stats["payload_carved_bytes"],
                    metadata_carved_bytes=stats["metadata_carved_bytes"],
                    peak_pool_payload_bytes=stats["combined"][
                        "peak_pool_backing_requested_bytes"
                    ],
                    max_bounds_slack_bytes=None,
                )
            )
    for campaign in a.campaign:
        manifest = json.loads((campaign / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("incomplete CHERI campaign")
        record = dict(
            label=manifest["label"],
            runtime_revocation=manifest["runtime_revocation"],
            scope=manifest["scope"],
            manifest_sha256=digest(campaign / "manifest.json"),
            artifacts=[],
            results=[],
        )
        for path, checksum in manifest["fingerprints"].items():
            f = Path(path)
            # The captured collector survives later formatting or documentation work.
            if f.name == "measure.py" and "host/cheribsd" in str(f):
                f = campaign / "collector.py"
            if digest(f) != checksum:
                raise ValueError(f"changed campaign input: {f.name}")
            name = Path(path).name
            if name in ("recorded.bin", "commands.bin"):
                name = Path(path).parent.name + "/" + name
            record["artifacts"].append(dict(file=name, sha256=checksum))
        for workload in workloads:
            runs = [r for r in manifest["results"] if r["workload"] == workload]
            if len(runs) != manifest["repetitions"] or any(
                r["status"] != "passed" for r in runs
            ):
                raise ValueError("incomplete CHERI repetitions")
            checksums = []
            for r in runs:
                path = campaign / f"{workload}-{r['repetition']}" / "output.bin"
                checksum = digest(path)
                if checksum != r["output_sha256"]:
                    raise ValueError("CHERI output changed")
                checksums.append(checksum)
                stats, _ = observations(
                    path, a.reference / "recordings" / workload / "recorded.bin", 0
                )
                if stats != r["measurements"]:
                    # JSON normalizes mapping keys to strings.
                    if json.dumps(stats, sort_keys=True) != json.dumps(
                        r["measurements"], sort_keys=True
                    ):
                        raise ValueError("stored accounting differs from binary")
                if r["cheri"]["pointer_bytes"] != 16:
                    raise ValueError("not confirmed purecap")
                if any(
                    not c["passed"] for c in r["controls"] + r.get("heap_controls", [])
                ):
                    raise ValueError("companion control failed")
                record["results"].append(r)
            if len(set(checksums)) != 1:
                raise ValueError("CHERI repetitions differ; do not collapse")
            rows.append(
                dict(
                    arm=manifest["label"],
                    workload=workload,
                    repetitions=len(runs),
                    payload_carved_bytes=stats["payload_carved_bytes"],
                    metadata_carved_bytes=stats["metadata_carved_bytes"],
                    peak_pool_payload_bytes=stats["combined"][
                        "peak_pool_backing_requested_bytes"
                    ],
                    max_bounds_slack_bytes=r["cheri"]["max_bounds_slack_bytes"],
                )
            )
        controls = [c for r in record["results"] for c in r["controls"]]
        if sorted(c["case"] for c in controls) != [0, 3, 5, 10, 11]:
            raise ValueError("missing or duplicate companion controls")
        for control in controls:
            expected = 162 if control["case"] in (10, 11) else 0
            if control["exit_code"] != expected or not control["passed"]:
                raise ValueError("incorrect companion verdict")
        if manifest.get("heap_probe_requested"):
            controls = [
                c for r in record["results"] for c in r.get("heap_controls", [])
            ]
            if sorted(c["stale"] for c in controls) != [False, True]:
                raise ValueError("missing or duplicate outer-heap controls")
            for control in controls:
                expected = (
                    162
                    if control["stale"] and manifest["runtime_revocation"] == "on"
                    else 0
                )
                if control["exit_code"] != expected or not control["passed"]:
                    raise ValueError("incorrect outer-heap verdict")
        records.append(record)
    for workload in workloads:
        peaks = {
            r["peak_pool_payload_bytes"] for r in rows if r["workload"] == workload
        }
        if len(peaks) != 1:
            raise ValueError("paired peak observations differ")
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    evidence = dict(
        schema="ffpool-cheri-comparison-v1",
        scope="allocator replay; no total-memory or timing comparison",
        workloads=reference["workloads"],
        summary=rows,
        campaigns=records,
    )
    (a.output / "measurements.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
