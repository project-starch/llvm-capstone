#!/usr/bin/env python3
"""Plot verified paired pool memory measurements; requires matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from measure import digest, observations


def checked_series(campaign, manifest):
    """Keep all acceptance checks before summarizing or selecting an excerpt."""
    series = {}
    for label, _, _ in manifest["workloads"]:
        reference = campaign / "recordings" / label / "recorded.bin"
        series[label] = {}
        for arm, mode in (("spatial", 0), ("sublet", 2)):
            entries = [
                r
                for r in manifest["results"]
                if r["workload"] == label
                and r["arm"] == arm
                and r["status"] == "passed"
            ]
            if not entries or len(entries) != manifest["repetitions"]:
                raise ValueError("incomplete repetition set")
            timelines = []
            for entry in entries:
                path = campaign / entry["directory"] / "capstone.bin"
                if digest(path) != entry["output_sha256"]:
                    raise ValueError("changed result")
                _, timeline = observations(path, reference, mode)
                timelines.append(timeline)
            if any(timeline != timelines[0] for timeline in timelines[1:]):
                raise ValueError("repetition curves differ; do not collapse them")
            series[label][arm] = timelines[0]
        spatial, sublet = series[label].values()
        if [row[0] for row in spatial] != [row[0] for row in sublet]:
            raise ValueError("paired event sequences differ")
    return series


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(output / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=("en", "de"), default="en")
    parser.add_argument("--detail-workload", help="default: first workload")
    parser.add_argument("--detail-start", type=int, default=580)
    parser.add_argument("--detail-stop", type=int, default=800)
    args = parser.parse_args()
    manifest = json.loads((args.campaign / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("plot requires a complete campaign")
    series = checked_series(args.campaign, manifest)
    label = args.detail_workload or manifest["workloads"][0][0]
    if series[label]["spatial"] != series[label]["sublet"]:
        raise ValueError("detail needs separate curves: paired timelines differ")
    excerpt = [
        row
        for row in series[label]["spatial"]
        if args.detail_start <= row[0] <= args.detail_stop
    ]
    if len(excerpt) < 2:
        raise ValueError("detail range must include at least two events")
    args.output.mkdir(parents=True, exist_ok=False)

    def t(en, de):
        return de if args.language == "de" else en

    plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold"})
    colors = ("#42677E", "#D57A27")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.22, right=0.96, top=0.76, bottom=0.25)
    fig.text(
        0.04,
        0.94,
        t(
            "Does Sublet need more pool memory?",
            "Braucht Sublet mehr Speicher im Pool?",
        ),
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.04,
        0.88,
        t(
            f"FFmpeg replay · {manifest['repetitions']} runs per mode and recording · peak comparison",
            f"FFmpeg-Replay · {manifest['repetitions']} Läufe pro Modus und Aufzeichnung · Vergleich des Spitzenbedarfs",
        ),
        fontsize=14,
    )
    ticks, labels = [], []
    maximum = (
        max(
            row[2]
            for arms in series.values()
            for timeline in arms.values()
            for row in timeline
        )
        / 1024
    )
    for i, (workload, seconds, dimensions) in enumerate(manifest["workloads"]):
        for j, arm in enumerate(("spatial", "sublet")):
            y = i * 3 + j
            peak = max(row[2] for row in series[workload][arm]) / 1024
            ax.barh(y, peak, height=0.65, color=colors[j])
            ax.text(
                peak + maximum * 0.018,
                y,
                t(
                    f"{peak:,.1f} KiB",
                    f"{peak:,.1f}".translate(str.maketrans(",.", ".,")) + " KiB",
                ),
                va="center",
            )
            ticks.append(y)
            labels.append(
                f"{workload}  ·  {dimensions}  ·  {seconds}s\nSpatial"
                if j == 0
                else "Sublet"
            )
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.22)
    ax.set_xlabel(
        t(
            "Peak payload held by pools (KiB; used + idle)",
            "Maximal im Pool gehaltene Nutzdaten (KiB; benutzt + frei)",
        ),
        labelpad=12,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.xaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    delta = max(
        abs(a[column] - b[column])
        for arms in series.values()
        for a, b in zip(arms["spatial"], arms["sublet"])
        for column in (1, 2)
    )
    fig.text(
        0.04,
        0.085,
        t(
            f"Largest difference at any replay event: {delta:,} bytes (used and held payload).",
            f"Größter Unterschied über alle Replay-Schritte: {delta:,} Byte (benutzt und gehalten).",
        ),
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.04,
        0.035,
        t(
            "Requested payload only. Excludes total metadata, capability storage and runtime cost.",
            "Nur angeforderte Nutzdaten. Keine Aussage zu gesamten Metadaten, Capability-Speicher oder Laufzeit.",
        ),
        fontsize=10,
        color="#555555",
    )
    save(fig, args.output, "payload")

    x, live, retained, idle = zip(*excerpt)
    live = [v / 1024 for v in live]
    retained = [v / 1024 for v in retained]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.71, bottom=0.23)
    fig.text(
        0.04,
        0.94,
        t(
            "Returned memory stays in the pool",
            "Zurückgegebener Speicher bleibt im Pool",
        ),
        fontsize=23,
        weight="bold",
    )
    fig.text(
        0.04,
        0.88,
        t(
            f"Excerpt: {label}, events {x[0]}–{x[-1]}. Both modes have exactly the same curve.",
            f"Ausschnitt: {label}, Schritte {x[0]}–{x[-1]}. Beide Modi haben exakt denselben Verlauf.",
        ),
        fontsize=13,
    )
    ax.fill_between(x, 0, live, step="post", color="#347DA2")
    ax.fill_between(x, live, retained, step="post", color="#DFE9ED")
    ax.step(x, retained, where="post", color="#607D89", linewidth=1)
    ax.legend(
        handles=[
            Patch(color="#347DA2", label=t("Currently used", "Gerade benutzt")),
            Patch(
                color="#DFE9ED",
                label=t(
                    "Idle in pool, ready for reuse", "Frei im Pool, wiederverwendbar"
                ),
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncol=2,
        frameon=False,
    )
    ax.set_ylim(0, max(retained) * 1.05)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(t("Requested payload (KiB)", "Angeforderte Nutzdaten (KiB)"))
    ax.set_xlabel(
        t(
            "Replay step (event order, not elapsed time)",
            "Replay-Schritt (Reihenfolge, keine Zeitachse)",
        ),
        labelpad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.04,
        0.10,
        t(
            "Blue falls: memory is returned. Blue rises: memory is handed out again.",
            "Blau fällt: Speicher wird zurückgegeben. Blau steigt: Speicher wird wieder vergeben.",
        ),
        fontsize=13,
    )
    fig.text(
        0.04,
        0.045,
        t(
            "The total area is payload still held by pools. This excerpt shows every event, without smoothing.",
            "Die gesamte Fläche bleibt im Pool. Der Ausschnitt zeigt jeden Schritt, ohne Glättung.",
        ),
        fontsize=11,
        color="#555555",
    )
    save(fig, args.output, "payload-detail")


if __name__ == "__main__":
    main()
