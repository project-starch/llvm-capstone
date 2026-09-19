#!/usr/bin/env python3
"""Export compact verified rows, retaining failures and the campaign's scope."""

import argparse
import csv
import json
from pathlib import Path

from measure import digest, observations, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    if manifest["status"] == "running":
        raise ValueError("cannot export a running campaign")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for entry in manifest["results"]:
        row = {k: entry[k] for k in ("workload", "arm", "repetition", "status")}
        if entry["status"] == "passed":
            output = campaign / entry["directory"] / "capstone.bin"
            if digest(output) != entry["output_sha256"]:
                raise ValueError("changed replay result")
            reference = campaign / "recordings" / entry["workload"] / "recorded.bin"
            result, _ = observations(
                output, reference, 0 if entry["arm"] == "spatial" else 2
            )
            if result != entry["measurement"]:
                raise ValueError("measurement differs on re-analysis")
            row.update({key: value for key, value in result.items() if key != "status"})
            row["replay_status"] = result["status"]
            row["output_sha256"] = entry["output_sha256"]
        rows.append(row)
    recordings = {}
    for label, seconds, dimensions in manifest["workloads"]:
        directory = campaign / "recordings" / label
        if not directory.exists():
            recordings[label] = {"status": "not-attempted"}
            continue
        workload = json.loads((directory / "workload.json").read_text())
        for path, expected in workload["sha256"].items():
            if digest(path) != expected:
                raise ValueError("changed recording artifact")
        recordings[label] = {
            "seconds": int(seconds),
            "dimensions": dimensions,
            "frames": workload["frames"],
            "output_identical": workload["output_identical"],
            "sha256": {
                name: digest(directory / name)
                for name in (
                    "recorded.bin",
                    "commands.bin",
                    "native.bin",
                    "stock.framemd5",
                )
            },
        }
    # Compact identities omit machine-specific paths and retain the full raw
    # manifest's digest. Caches/build flags remain in that external manifest.
    identities = {}
    for path, value in manifest["fingerprints"].items():
        name = Path(path).name
        if name in (
            "clang",
            "ld.lld",
            "qemu-system-riscv64",
            "Image",
            "fw_jump.elf",
            "rootfs.ext2",
            "replay.dom",
            "domain-loader",
            "measure.py",
            "analyze.py",
            "observe-pool-events.c",
            "pool-allocator.c",
            "metadata-allocator.c",
            "trace.h",
        ):
            identities[name] = value
    result = {
        "schema": manifest["schema"],
        "status": manifest["status"],
        "scope": manifest["scope"],
        "source_revision": manifest["source_revision"],
        "repetitions_requested": manifest["repetitions"],
        "node_capacity": manifest["node_capacity"],
        "raw_manifest_sha256": digest(campaign / "manifest.json"),
        "identities": identities,
        "recordings": recordings,
        "results": rows,
        "unknown": manifest["unknown"],
        "reservations_bytes": manifest.get("reservations_bytes"),
        "reservation_scope": "Four regions from trace.h; excludes static domain, driver and node/tag storage",
    }
    write_json(args.output / "measurements.json", result)
    with (args.output / "summary.csv").open("w") as stream:
        fields = [
            "workload",
            "arm",
            "repetition",
            "status",
            "events",
            "payload_carved_bytes",
            "metadata_carved_bytes",
            "peak_live_requested_bytes",
            "peak_pool_backing_requested_bytes",
            "revoke",
            "init_bytes",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flat = dict(row, **row.get("combined", {}), **row.get("primitives", {}))
            writer.writerow({key: flat.get(key, "") for key in fields})


if __name__ == "__main__":
    main()
