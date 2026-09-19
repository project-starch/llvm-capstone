#!/usr/bin/env python3
"""Plot checked payload series against replay events; requires matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure import digest, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("plot requires a complete campaign")
    fig, axes = plt.subplots(
        1,
        len(manifest["workloads"]),
        squeeze=False,
        figsize=(12, 3.8),
        constrained_layout=True,
    )
    for ax, (label, seconds, dimensions) in zip(axes[0], manifest["workloads"]):
        reference = args.campaign / "recordings" / label / "recorded.bin"
        for arm, color, style in (
            ("spatial", "#286090", "-"),
            ("sublet", "#cb4b16", "--"),
        ):
            entries = [
                r
                for r in manifest["results"]
                if r["workload"] == label
                and r["arm"] == arm
                and r["status"] == "passed"
            ]
            if len(entries) != manifest["repetitions"] or any(
                r["status"] != "passed" for r in entries
            ):
                raise ValueError("incomplete repetition set")
            timelines = []
            for entry in entries:
                path = args.campaign / entry["directory"] / "capstone.bin"
                if digest(path) != entry["output_sha256"]:
                    raise ValueError("changed result")
                _, timeline = observations(
                    path, reference, 0 if arm == "spatial" else 2
                )
                timelines.append(timeline)
            if any(series != timelines[0] for series in timelines[1:]):
                raise ValueError("repetition curves differ; do not collapse them")
            x, live, backing, _ = zip(*timelines[0])
            ax.step(
                x,
                [v / 1024 for v in live],
                where="post",
                color=color,
                linestyle=style,
                label=f"{arm}: live",
            )
            ax.step(
                x,
                [v / 1024 for v in backing],
                where="post",
                color=color,
                linestyle=style,
                alpha=0.45,
                label=f"{arm}: pool backing",
            )
        ax.set_title(f"{label}: {seconds}s input, {dimensions}")
        ax.set_xlabel("Replay event (not time)")
        ax.set_ylabel("Requested pool payload (KiB)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        "FFmpeg allocator replay: paired payload behavior\n"
        "Three identical repetitions per arm; metadata and reservations excluded",
        fontsize=12,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "payload.png", dpi=180)
    fig.savefig(args.output / "payload.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
