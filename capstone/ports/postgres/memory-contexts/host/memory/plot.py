#!/usr/bin/env python3
"""Render PDF/PNG plots on the host from an exported result bundle, without QEMU."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(results, output):
    provenance = json.loads((results / "provenance.json").read_text())
    for name, expected in provenance["files"].items():
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Result checksum mismatch: {name}")
    with (results / "summary.csv").open(newline="") as source:
        summaries = list(csv.DictReader(source))
    if not summaries:
        raise ValueError("Result bundle contains no summaries")
    for summary in summaries:
        summary["arena_reserved_bytes"] = int(summary["arena_reserved_bytes"])
    series = {}
    with gzip.open(results / "profiles.csv.gz", "rt", newline="") as source:
        for row in csv.DictReader(source):
            if row["kind"] != "event":
                continue
            key = row.pop("workload"), row.pop("program")
            series.setdefault(key, []).append(
                {k: int(v) if k != "kind" else v for k, v in row.items()}
            )
    arena_sizes = {s["arena_reserved_bytes"] for s in summaries}
    if len(arena_sizes) != 1:
        raise SystemExit("expected a fixed arena budget across the campaign")
    arena_mib = arena_sizes.pop() / 2**20
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
        f"{provenance['repetitions']} repetitions per arm; identical profiles. Backing definitions differ by port; arena reservation is {arena_mib:g} MiB in both.\nNode creation is cumulative demand. Replay buffers and global node/tag storage are outside these curves.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output / "memory.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output / "memory.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Directory containing summary.csv and profiles.csv.gz",
    )
    parser.add_argument(
        "output", type=Path, help="New directory for the regenerated plots"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plot(args.results, args.output)


if __name__ == "__main__":
    main()
