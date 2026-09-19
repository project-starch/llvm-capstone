#!/usr/bin/env python3
"""Plot a verified CHERI comparison export with direct labels."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("measurements", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--language", choices=("en", "de"), default="en")
a = p.parse_args()
m = json.loads(a.measurements.read_text())
a.output.mkdir(parents=True, exist_ok=False)
t = lambda en, de: de if a.language == "de" else en
rows = m["summary"]
workloads = m["workloads"]
labels = {
    "capstone-spatial": "Capstone Spatial",
    "capstone-sublet": "Capstone Sublet",
    "cheri-spatial-arena": "CHERI Purecap",
}
arms = list(dict.fromkeys(r["arm"] for r in rows))
colors = ["#42677E", "#D57A27", "#38816A"]
plt.rcParams.update({"font.size": 12})


def number(value):
    text = f"{value:,.1f}"
    return text.translate(str.maketrans(",.", ".,")) if a.language == "de" else text


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(a.output / f"{name}.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(11, 8))
fig.subplots_adjust(left=0.23, right=0.97, top=0.81, bottom=0.16)
fig.text(
    0.04,
    0.94,
    t("The pools hold the same payload", "Die Pools halten gleich viele Nutzdaten"),
    fontsize=23,
    weight="bold",
)
fig.text(
    0.04,
    0.89,
    t(
        "Same FFmpeg recordings; peak requested bytes held by pools.",
        "Dieselben FFmpeg-Aufzeichnungen; Spitzenbedarf der Nutzdaten im Pool.",
    ),
    fontsize=13,
)
ticks, names = [], []
maximum = max(r["peak_pool_payload_bytes"] for r in rows) / 1024
for i, (workload, seconds, dimensions) in enumerate(workloads):
    for j, arm in enumerate(arms):
        row = next(r for r in rows if r["workload"] == workload and r["arm"] == arm)
        y = i * (len(arms) + 1) + j
        size = row["peak_pool_payload_bytes"] / 1024
        ax.barh(y, size, color=colors[j], height=0.68)
        ax.text(size + maximum * 0.015, y, number(size) + " KiB", va="center")
        ticks.append(y)
        names.append(
            (f"{dimensions} · {seconds}s\n" if j == 0 else "") + labels.get(arm, arm)
        )
ax.set_yticks(ticks, names)
ax.invert_yaxis()
ax.set_xlim(0, maximum * 1.24)
ax.set_xlabel(
    t("Peak requested pool payload (KiB)", "Maximal gehaltene Nutzdaten (KiB)")
)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=10)
fig.text(
    0.04,
    0.075,
    t(
        "Requested payload is equal at every event, not only at the peaks.",
        "Die Nutzdaten sind bei jedem Replay-Schritt gleich, auch zwischen den Spitzen.",
    ),
    weight="bold",
    fontsize=13,
)
fig.text(
    0.04,
    0.03,
    t(
        "These arms have different lifetime guarantees. This is not total memory overhead.",
        "Die Modi bieten unterschiedliche Lebensdauer-Garantien. Das ist kein Gesamt-Overhead.",
    ),
    fontsize=11,
    color="#555555",
)
save(fig, "payload-comparison")

fig, ax = plt.subplots(figsize=(11, 5.8))
fig.subplots_adjust(left=0.20, right=0.96, top=0.76, bottom=0.27)
fig.text(
    0.04,
    0.93,
    t("Extra space in the payload arena", "Zusätzlicher Platz in der Payload-Arena"),
    fontsize=23,
    weight="bold",
)
fig.text(
    0.04,
    0.87,
    t(
        "CHERI Purecap relative to Capstone Sublet; same recordings.",
        "CHERI Purecap gegenüber Capstone Sublet; dieselben Aufzeichnungen.",
    ),
    fontsize=13,
)
deltas = []
for i, (workload, seconds, dimensions) in enumerate(workloads):
    base = next(
        r for r in rows if r["workload"] == workload and r["arm"] == "capstone-sublet"
    )
    cheri = next(
        r
        for r in rows
        if r["workload"] == workload and r["arm"] == "cheri-spatial-arena"
    )
    delta = cheri["payload_carved_bytes"] - base["payload_carved_bytes"]
    deltas.append((delta, 100 * delta / base["payload_carved_bytes"]))
    ax.barh(i, delta, height=0.58, color="#38816A")
maximum = max(abs(delta) for delta, _ in deltas) or 1
for i, (delta, percent) in enumerate(deltas):
    label = f"{delta:+,} " + t("bytes", "Byte") + f" ({percent:+.3f}%)"
    if a.language == "de":
        label = label.replace(".", ",")
    ax.text(delta + maximum * 0.04, i, label, va="center", fontsize=14, weight="bold")
ax.set_yticks(range(len(workloads)), [f"{w[2]} · {w[1]}s" for w in workloads])
ax.invert_yaxis()
ax.set_xlim(min(0, min(v[0] for v in deltas) * 1.2), maximum * 1.8)
ax.set_xlabel(
    t(
        "Additional bytes carved from the arena",
        "Zusätzlich aus der Arena belegte Byte",
    ),
    labelpad=12,
)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=12)
fig.text(
    0.04,
    0.13,
    t(
        "Includes alignment and tail padding for CHERI compressed bounds.",
        "Enthält Ausrichtung und Auffüllung für CHERIs komprimierte Capability-Grenzen.",
    ),
    fontsize=12,
)
fig.text(
    0.04,
    0.055,
    t(
        "Excludes capability tags, OS, libc, static bookkeeping and protection metadata.",
        "Ohne Capability-Tags, Betriebssystem, libc, statische Buchhaltung und Schutzmetadaten.",
    ),
    fontsize=11,
    color="#555555",
)
save(fig, "arena-padding")
