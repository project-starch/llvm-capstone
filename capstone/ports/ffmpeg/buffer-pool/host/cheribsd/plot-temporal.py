#!/usr/bin/env python3
"""Plot counters separately: neither color IDs nor nodes are byte costs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("measurements", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--language", choices=("en", "de"), default="en")
    a = p.parse_args()
    data = json.loads(a.measurements.read_text())
    de = a.language == "de"
    runs = [r for r in data["churn"] if r["case"] == 13]
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [0.65, 1, 1]}
    )
    fig.suptitle(
        (
            "Gleicher Pool, viele Wiederverwendungen"
            if de
            else "Same pool, repeated reuse"
        ),
        fontsize=21,
        x=0.15,
        ha="left",
        y=0.98,
    )
    fig.text(
        0.15,
        0.935,
        (
            "QEMU · geschützte Ausleihen · Kennungen/Knoten sind keine Byte-Angaben"
            if de
            else "QEMU · protected leases · IDs/nodes are not byte measurements"
        ),
        fontsize=11,
    )
    axes[0].plot([0, data["rounds"]], [128, 128], color="#333333", linewidth=3)
    axes[0].set_ylim(0, 180)
    axes[0].set_yticks([0, 128], ["0", "128 B"])
    axes[0].set_title(
        (
            "Pool-Nutzspeicher: bei beiden konstant"
            if de
            else "Pool payload: constant in both arms"
        ),
        loc="left",
        fontweight="bold",
    )
    fmt = FuncFormatter(
        lambda n, pos: f"{int(n):,}".replace(",", ".") if de else f"{int(n):,}"
    )
    for ax, arm, field, color, title in (
        (
            axes[1],
            "picasso",
            "busy",
            "#146C94",
            (
                "PICASSO: reservierte Farbkennungen"
                if de
                else "PICASSO: reserved color IDs"
            ),
        ),
        (
            axes[2],
            "capstone",
            "not_on_free_list",
            "#87569B",
            (
                "Capstone-QEMU: Knoten außerhalb der Freiliste"
                if de
                else "Capstone QEMU: nodes outside the free list"
            ),
        ),
    ):
        for run in runs:
            if run["arm"] != arm:
                continue
            points = [r for r in run["checkpoints"] if r["round"]]
            ax.plot(
                [r["round"] for r in points],
                [r[field] for r in points],
                color=color,
                linewidth=2,
            )
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(fmt)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dedede", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(fmt)
    axes[-1].set_xlabel(
        (
            "Anzahl Wiederverwendungen desselben 64-Byte-Blocks"
            if de
            else "Number of reuses of the same 64-byte block"
        ),
        fontsize=12,
    )
    caption = (
        (
            f"{data['repetitions']} Wiederholung(en). Nutzspeicher ohne Verwaltungsdaten. "
            "PICASSO-Adapter zusätzlich: höchstens 128 B angeforderter Token-Speicher.\n"
            "Dieser Capstone-QEMU gibt im Lauf keine Knoten zurück; kein Vergleich mit dem aktuellen RTL-Reclaimer."
        )
        if de
        else (
            f"{data['repetitions']} repetition(s). Payload excludes bookkeeping. "
            "PICASSO adapter additionally requests at most 128 B of live token storage.\n"
            "This Capstone QEMU returns no nodes during the run; this does not compare the current RTL reclaimer."
        )
    )
    fig.text(0.15, 0.018, caption, fontsize=9, linespacing=1.5)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.865, bottom=0.14, hspace=0.52)
    a.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(a.output / f"temporal-reuse.{suffix}", dpi=160)


if __name__ == "__main__":
    main()
