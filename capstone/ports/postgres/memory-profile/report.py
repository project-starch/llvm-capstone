#!/usr/bin/env python3
"""Export reconciled QEMU memory data and plots without VM logs or host paths."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    summaries, series = [], {}
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
                verdict=json.loads((directory / "verdict.json").read_text()),
                identity=identity,
                memory_csv_sha256=run.get("memory_csv_sha256"),
            )
        )
        if run["runner_exit"] != 0 or not run["profile"] or run["repetition"] != 1:
            continue
        memory = json.loads((directory / "memory-summary.json").read_text())
        rows = list(csv.DictReader((directory / "memory.csv").open()))
        events = [
            {k: int(v) if k != "kind" else v for k, v in row.items()}
            for row in rows
            if row["kind"] == "event"
        ]
        series[run["workload"], run["program"]] = events
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
        writer = csv.DictWriter(out, fieldnames=summaries[0].keys())
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
        "comparison": "Same trace, target ABI and 64-MiB arena budget. Spatial and Sublet use different backing policies and layouts; differences are complete-port effects, not isolated revocation-instruction overhead.",
    }
    (args.output / "metrics.json").write_text(json.dumps(definitions, indent=2) + "\n")
    provenance["files"] = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("summary.csv", "profiles.csv.gz", "metrics.json")
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    plt.rcParams.update({"font.size": 10, "svg.hashsalt": "postgres-memory-profile"})
    workloads = list(dict.fromkeys(s["workload"] for s in summaries))
    fig, axes = plt.subplots(
        len(workloads), 3, figsize=(13, 3.5 * len(workloads)), squeeze=False
    )
    for row, label in enumerate(workloads):
        off = series[label, "replay-spatial"]
        on = series[label, "replay-sublet"]
        for values, color, name in (
            (off, "#3274a1", "Temporal off: spatial port"),
            (on, "#e1812c", "Temporal on: Sublet port"),
        ):
            x = [r["event"] / 1e6 for r in values]
            axes[row, 0].plot(
                x,
                [r["backing_bytes"] / 2**20 for r in values],
                color=color,
                label=name,
                lw=1.2,
            )
        x = [r["event"] / 1e6 for r in on]
        axes[row, 0].plot(
            x,
            [r["live_payload_bytes"] / 2**20 for r in on],
            color="#333333",
            label="Requested live payload",
            lw=1,
        )
        axes[row, 0].set(ylabel=f"{label}: backing / payload (MiB)")
        axes[row, 1].plot(
            x,
            [r["metadata_records_live_bytes"] / 1024 for r in on],
            color="#e1812c",
            label="Records in use",
        )
        axes[row, 1].axhline(
            on[-1]["metadata_records_reserved_bytes"] / 1024,
            color="#777777",
            ls="--",
            label="Static record capacity",
        )
        axes[row, 1].set(ylabel="Sublet record storage (KiB)")
        axes[row, 2].plot(
            x, [r["nodes_created_cumulative"] / 1e6 for r in on], color="#e1812c"
        )
        axes[row, 2].set(ylabel="Sublet nodes created (millions)")
        for col in range(3):
            axes[row, col].set_xlabel("Replay event (millions)")
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_ylim(bottom=0)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
    fig.suptitle("PostgreSQL allocator memory behaviour in QEMU", fontsize=14)
    fig.text(
        0.5,
        0.01,
        f"{manifest['repetitions']} repetitions per arm; identical profiles. Backing definitions differ by port; arena reservation is 64 MiB in both.\nNode creation is cumulative demand. Replay buffers and global node/tag storage are outside these curves.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(args.output / "memory.svg", metadata={"Date": None})
    fig.savefig(
        args.output / "memory.pdf", metadata={"CreationDate": None, "ModDate": None}
    )
    fig.savefig(args.output / "memory.png", dpi=160)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
