#!/usr/bin/env python3
"""Export reconciled QEMU memory data and plots without VM logs or host paths."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "common/host"))

from port_support import write_checksums
from plot import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--environment",
        type=Path,
        help="Additional verified build and workload provenance",
    )
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise SystemExit("campaign has not completed")
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    provenance = dict(
        scope=manifest["scope"],
        comparison=manifest["comparison"],
        repetitions=manifest["repetitions"],
        tool_sha256=manifest["tool_sha256"],
        instrumentation_counts_match=manifest["instrumentation_counts_match"],
        repeated_profiles_identical=manifest["repeated_profiles_identical"],
        runs=[],
    )
    if args.environment:
        provenance["environment"] = json.loads(args.environment.read_text())
    provenance["resume_tool_sha256"] = manifest.get("resume_tool_sha256", [])
    archive_rows = []
    for run in manifest["results"]:
        directory = args.campaign / run["artifacts"]
        identity = json.loads((directory / "manifest.json").read_text())
        provenance["runs"].append(
            dict(
                workload=run["workload"],
                program=run["program"],
                repetition=run["repetition"],
                profile=run["profile"],
                runner_exit=run["runner_exit"],
                failure_phase=run.get("failure_phase"),
                explicit_setup_retry=run.get("explicit_setup_retry", False),
                verdict=json.loads((directory / "verdict.json").read_text()),
                identity=identity,
                output_sha256={
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("serial.log", "share/payload.log")
                    if (directory / name).exists()
                },
                memory_csv_sha256=run.get("memory_csv_sha256"),
            )
        )
        if run["runner_exit"] != 0 or not run["profile"] or run["repetition"] != 1:
            continue
        memory = json.loads((directory / "memory-summary.json").read_text())
        rows = list(csv.DictReader((directory / "memory.csv").open()))
        for row in rows:
            archive_rows.append(
                dict(workload=run["workload"], program=run["program"], **row)
            )
        final, peaks = memory["final"], memory["peaks"]
        summaries.append(
            dict(
                workload=run["workload"],
                program=run["program"],
                events=memory["oracle"]["events"],
                requested_payload_peak_bytes=memory["oracle"]["payload_peak_bytes"],
                backing_peak_bytes=peaks["backing_peak"]["backing_bytes"],
                backing_final_bytes=final["backing_bytes"],
                assigned_backing_final_bytes=final["assigned_backing_bytes"],
                metadata_records_peak_bytes=peaks["metadata_peak"][
                    "metadata_records_live_bytes"
                ],
                metadata_records_reserved_bytes=final[
                    "metadata_records_reserved_bytes"
                ],
                nodes_created_cumulative=final["nodes_created_cumulative"],
                init_bytes_cumulative=final["init_bytes_cumulative"],
                arena_reserved_bytes=memory["regions"]["arena"],
                scratch_reserved_bytes=memory["regions"]["scratch"],
                report_reserved_bytes=memory["regions"]["payload"],
                trace_reserved_bytes=memory["regions"]["trace"],
                node_capacity=memory["node_capacity"],
            )
        )
    with (args.output / "summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out, fieldnames=summaries[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summaries)
    # Each repeated profile is identical; keep one copy per arm/workload and
    # preserve all repetition hashes in the provenance file.
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=archive_rows[0].keys())
    writer.writeheader()
    writer.writerows(archive_rows)
    (args.output / "profiles.csv.gz").write_bytes(
        gzip.compress(text.getvalue().encode(), mtime=0)
    )
    definitions = {
        "backing_bytes": {
            "replay-spatial": "Unfreed manager backing allocations, including first-fit headers; reusable arena holes excluded",
            "replay-sublet": "All subpool extents carved from the arena, including returned/cached pools; these extents do not coalesce into the arena",
        },
        "assigned_backing_bytes": "Backing assigned to managers and their caches; Sublet excludes pools returned to its backing free list",
        "block_bytes": "Backing held in PostgreSQL blocks; contains payload, headers, rounding and free capacity",
        "stranded_block_bytes": "Sublet carved block bytes whose bookkeeping was freed, unavailable until pool reset",
        "metadata_records_live_bytes": "Active Sublet pool/block/chunk records and context-header slots, including embedded capability slots",
        "metadata_records_reserved_bytes": "All static Sublet record arrays and header free-index array; live records are included, not additive",
        "nodes_created_cumulative": "Sublet split + mrev operations; cumulative identifier demand, excluding monitor/loader operations, not live-node occupancy",
        "init_bytes_cumulative": "Bytes initialized by Sublet pool revocation, excluding loader setup/cleanup",
        "arena_capacity_bytes": "Available manager arena; the loader reserves this full amount for both arms regardless of the backing occupancy curve",
        "observer_bytes": "Dynamic profiling context table, inside separate replay scratch",
        "peaks": "Exact maxima at replay operation boundaries; internal transient allocations during one operation are not separately sampled",
        "snapshots": "First 64 events, each 1024th event, every reset/delete, final event and exact peak witnesses",
        "scope": "Memory behaviour of allocator replays. No QEMU timing, FPGA result or security claim. Full application RSS and global node/tag storage are not measured.",
        "comparison": "Same trace and target ABI, with matching arena budgets. Spatial and Sublet use different backing policies and layouts; differences are complete-port effects, not isolated revocation-instruction overhead.",
    }
    (args.output / "metrics.json").write_text(json.dumps(definitions, indent=2) + "\n")
    provenance["files"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("summary.csv", "profiles.csv.gz", "metrics.json")
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    plot(args.output, args.output)
    write_checksums(args.output)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
