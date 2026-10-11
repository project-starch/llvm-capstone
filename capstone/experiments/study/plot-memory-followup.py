#!/usr/bin/env python3
"""Validate full-application memory attempts and draw the follow-up figures.

Raw guest transcripts stay under /tmp/capstone. The committed data file is
enough to redraw the figures, while --inputs rechecks every source transcript.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIB = 1024 * 1024
ORACLE = re.compile(r"STUDY-ORACLE phase=(\d+) rows=(\d+) hash=([0-9a-f]{16})")
PHASES = (100, 110, 120, 130, 140, 142, 145, 150, 160, 161, 170, 180,
          190, 200, 210, 230, 240, 250, 260, 270, 280, 290, 300, 310,
          320, 400, 410, 500, 510, 520, 980, 990)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(line):
    return {k: int(v) if v.isdigit() else v
            for k, v in re.findall(r"([a-z_]+)=([\w-]+)", line)}


def oracle(text):
    return {int(p): (int(n), h) for p, n, h in ORACLE.findall(text)}


def sqlite_attempt(spec, expected):
    path = Path(spec["log"])
    text = path.read_text(errors="replace")
    stderr = path.with_suffix(".stderr")
    stderr_text = stderr.read_text(errors="replace") if stderr.exists() else ""
    observed = oracle(text)
    arm = spec["arm"]
    status = spec["status"]
    is_capstone = arm.startswith("capstone")
    memory = re.findall(r"STUDY-(?:CAP|MEM) phase=(\d+) (.+)", text)
    snapshots = {int(p): fields(tail) for p, tail in memory}
    result = dict(size=spec["size"], arm=arm, status=status,
                  pool=spec.get("pool"), tables=spec.get("tables"),
                  heap=spec.get("heap"), log_sha256=digest(path),
                  stderr_sha256=digest(stderr) if stderr.exists() else None,
                  oracle_phases=len(observed), last_oracle_phase=max(observed, default=None),
                  memory_phases=len(snapshots), final=snapshots.get(990))
    if status == "pass":
        if observed != expected or tuple(snapshots) != PHASES or "TOTAL" not in text:
            raise ValueError(f"{arm} size {spec['size']}: incomplete or wrong full workload")
        if is_capstone:
            if "__CAPSTONE_SPEEDTEST1_RAN__" not in text or "DROPPED 0" not in text:
                raise ValueError(f"{arm}: Capstone completion not observed")
            if snapshots[990]["oom"]:
                raise ValueError(f"{arm}: completed with allocator failures")
            result["total_visible_reservation"] = spec["pool"] + spec["tables"]
        else:
            if snapshots[990]["heap"] != spec["heap"] or snapshots[990]["revoke_errors"]:
                raise ValueError(f"{arm}: effective heap or revocation mismatch")
            if arm in ("poisoncap-corrected", "poisoncap-pressure") and snapshots[990]["revokes"] <= 0:
                raise ValueError(f"{arm}: temporal revocation inactive")
            if arm == "poisoncap-spatial" and snapshots[990]["revokes"]:
                raise ValueError("PoisonCap spatial unexpectedly revoked")
            result["total_visible_reservation"] = (spec["heap"] +
                ((snapshots[990]["links"] + 4095) // 4096) * 4096 +
                snapshots[990]["qtable"])
    elif status == "capability_fault":
        if "domain halted by capability fault" not in text or len(observed) == len(PHASES):
            raise ValueError(f"{arm}: missing capability-fault evidence")
    elif status == "sql_oom":
        if "SQL error: out of memory" not in text + stderr_text or len(observed) == len(PHASES):
            raise ValueError(f"{arm}: missing SQL OOM evidence")
    elif status == "kernel_panic":
        serial = Path(spec["serial"])
        if "panic: Poison probe missing page" not in serial.read_text(errors="replace"):
            raise ValueError(f"{arm}: missing kernel-panic evidence")
        result["serial_sha256"] = digest(serial)
    else:
        raise ValueError(f"unknown outcome {status}")
    return result


def ffmpeg_attempt(spec, expected):
    stdout = Path(spec["stdout"])
    stderr = Path(spec["stderr"])
    out_text, err_text = stdout.read_text(), stderr.read_text()
    if out_text != expected:
        raise ValueError(f"FFmpeg {spec}: decoded-frame oracle differs")
    releases = [fields(s) for s in err_text.splitlines()
                if s.startswith("EXP-CHERI phase=released-")]
    pool = [fields(s) for s in err_text.splitlines() if s.startswith("FFPOOL-MEM phase=released-")]
    adapter = [fields(s) for s in err_text.splitlines() if s.startswith("FF2_POISONCAP ")]
    if (len(releases) != spec["batches"] or len(pool) != spec["batches"] or
            len(adapter) != 2 * spec["batches"] or
            releases[-1]["revocation"] != 0 or
            pool[-1]["snapshots"] != adapter[-1]["snapshot_bytes"]):
        raise ValueError(f"FFmpeg {spec}: incomplete memory accounting")
    return dict(batches=spec["batches"], mode=spec["mode"],
                variant=spec["variant"], stdout_sha256=digest(stdout),
                stderr_sha256=digest(stderr),
                allocated_at_release=[r["allocated"] for r in releases],
                resident_at_release=[r["resident"] for r in releases],
                snapshots_at_release=[p["snapshots"] for p in pool],
                snapshot_peak=adapter[-1].get("snapshot_peak"),
                snapshot_final=adapter[-1]["snapshot_bytes"],
                sweeps=adapter[-1]["sweeps"])


def collect(manifest):
    expected = {}
    for key, path in manifest["oracles"].items():
        raw = Path(path)
        result = oracle(raw.read_text())
        if tuple(result) != PHASES:
            raise ValueError(f"size {key}: incomplete native oracle")
        expected[int(key)] = result
    sqlite = [sqlite_attempt(spec, expected[spec["size"]]) for spec in manifest["sqlite"]]
    ffmpeg = []
    for spec in manifest["ffmpeg"]:
        old = next(p for p in manifest["ffmpeg"]
                   if p["batches"] == spec["batches"] and p["mode"] == 0 and
                   p["variant"] == "original")
        ffmpeg.append(ffmpeg_attempt(spec, Path(old["stdout"]).read_text()))
    selected = [r for r in sqlite if r["size"] == 1 and r["status"] == "pass"]
    key = lambda arm: next(r for r in selected if r["arm"] == arm and r.get("heap", r.get("pool")) ==
                           (8388608 if arm.startswith("poisoncap-") and arm != "poisoncap-spatial" else
                            1179648 if arm == "poisoncap-spatial" else 1310720))
    if key("poisoncap-corrected")["final"] != key("poisoncap-pressure")["final"]:
        raise ValueError("8 MiB pressure control changed the observed SQLite allocator state")
    binaries = {name: digest(path) for name, path in manifest["binaries"].items()}
    return dict(schema=1, workload="full SQLite 3.22 speedtest1 main and FFmpeg 9.0.1 decoder",
                oracles={str(k): {"sha256": digest(manifest["oracles"][str(k)]),
                                   "rows": sum(n for n, _ in expected[k].values())}
                         for k in expected},
                binaries=binaries, clip_sha256=digest(manifest["clip"]),
                sqlite=sqlite, ffmpeg=ffmpeg)


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sqlite(data, out):
    records = [r for r in data["sqlite"] if r["size"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    left = [("Capstone spatial", "capstone-spatial"),
            ("Capstone + Sublet", "capstone-sublet"),
            ("PoisonCap spatial", "poisoncap-spatial")]
    right = [("PoisonCap corrected", "poisoncap-corrected"),
             ("PoisonCap pressure", "poisoncap-pressure")]
    styles = {"pass": ("#33795e", "o"),
              "capability_fault": ("#c18635", "x"),
              "sql_oom": ("#c18635", "x"),
              "kernel_panic": ("#9a5155", "X")}
    for ax, entries, limits in zip(axes, (left, right), ((.95, 1.38), (7.35, 8.22))):
        for y, (name, arm) in enumerate(entries):
            for r in records:
                if r["arm"] != arm:
                    continue
                budget = r.get("pool") if arm.startswith("capstone") else r.get("heap")
                color, marker = styles[r["status"]]
                ax.scatter(budget / MIB, y, marker=marker, color=color, s=82, zorder=3)
                if r["status"] == "pass" and (budget == 1310720 or budget == 1179648 or
                                               arm.startswith("poisoncap-") and budget == 8388608):
                    ax.annotate(f'{r["total_visible_reservation"]/MIB:.2f} MiB total',
                                (budget / MIB, y), xytext=(0, 10), textcoords="offset points",
                                ha="center", fontsize=7)
        ax.set_yticks(range(len(entries)), [x[0] for x in entries])
        ax.invert_yaxis()
        ax.set_ylim(len(entries) - .45, -.75)
        ax.set_xlim(*limits)
        ax.set_xlabel("Configured heap / pool (MiB)")
        ax.grid(axis="x", alpha=.2)
    axes[0].set_title("Complete SQLite size 1: spatial and Sublet")
    axes[1].set_title("Complete SQLite size 1: temporal policies")
    fig.subplots_adjust(left=.17, right=.98, bottom=.28, wspace=.47)
    fig.text(.17, .08, "Green: exact SQL oracle passed. Orange: SQL OOM or Capstone fault. Red: PoisonCap kernel panic.\n"
             "Totals include application allocator tables; kernel metadata is excluded. Panics do not bound required memory.", fontsize=8)
    save(fig, out, "sqlite-budget-trials")


def plot_ffmpeg(data, out):
    records = data["ffmpeg"]
    get = lambda b, m, v: next(r for r in records if r["batches"] == b and
                              r["mode"] == m and r["variant"] == v)
    batches = (1, 4, 16)
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    bars = ((0, "original", "Spatial", "#3f6b8a"),
            (2, "original", "Full-copy temporal", "#a75e60"),
            (2, "selective", "Selective temporal", "#4b8564"))
    for i, (mode, variant, label, color) in enumerate(bars):
        vals = [get(b, mode, variant)["allocated_at_release"][-1] / MIB for b in batches]
        axes[0].bar(x + (i-1)*.24, vals, .22, label=label, color=color)
    axes[0].set_ylabel("Jemalloc allocated after final release (MiB)")
    axes[0].set_ylim(0, 1.25)
    axes[0].legend(frameon=False, fontsize=8)
    for label, values, color, style in (
        ("Full-copy final", [get(b, 2, "original")["snapshot_final"] / 1024 for b in batches], "#a75e60", "o-"),
        ("Selective peak", [get(b, 2, "selective")["snapshot_peak"] / 1024 for b in batches], "#4b8564", "s-"),
        ("Selective final", [get(b, 2, "selective")["snapshot_final"] / 1024 for b in batches], "#3f6b8a", "^-")):
        axes[1].plot(x, values, style, label=label, color=color, lw=2)
    axes[1].set_ylabel("Snapshot backing (KiB)")
    axes[1].set_ylim(-12, 370)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, [str(b) for b in batches])
        ax.set_xlabel("Independent 30-frame decoder streams")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title("Same decoder oracle, after last stream")
    axes[1].set_title("Persistent-state copy policy")
    fig.subplots_adjust(bottom=.22, wspace=.28)
    fig.text(.12, .035, "PoisonCap within-platform policies. The full-copy adapter was conservative; the selective adapter restores only stateful pool entries.", fontsize=8)
    save(fig, out, "ffmpeg-selective-snapshots")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path)
    source.add_argument("--data", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = collect(json.loads(args.inputs.read_text())) if args.inputs else json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.inputs:
        (args.out / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    plot_sqlite(data, args.out)
    plot_ffmpeg(data, args.out)


if __name__ == "__main__":
    main()
