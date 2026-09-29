#!/usr/bin/env python3
"""Validate whole-decoder pool runs and plot only within-platform memory facts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BATCHES = (1, 4, 16)
MIB = 1024 * 1024


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def line(text, prefix):
    matches = [s for s in text.splitlines() if s.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {prefix!r} line, got {len(matches)}")
    return fields(matches[0])


def fields(text):
    return {k: int(v) if v.isdigit() else v
            for k, v in re.findall(r"([a-z_]+)=([\w-]+)", text)}


def phase_lines(text, prefix, batches):
    found = [fields(s) for s in text.splitlines() if s.startswith(prefix)]
    expected = [f"{name}-{i}" for i in range(batches)
                for name in ("before", "released")]
    if [x.get("phase") for x in found] != expected:
        raise ValueError(f"wrong {prefix} phase sequence")
    return found


def load(cap, poison, points):
    cap = Path(cap); poison = Path(poison)
    source = {p["id"]: p for p in json.loads(Path(points).read_text())}
    cap_manifest = json.loads((cap / "manifest.json").read_text())
    cap_image = source["ffmpeg-f30-b1-pool0"]["image"]
    if digest(cap_image) != cap_manifest["images"][cap_image]:
        raise ValueError("Capstone image changed since the recorded campaign")
    input_path = "/tmp/capstone/domain-process-runtime/share/experiments/clip-1.mkv"
    if digest(input_path) != cap_manifest["workload_inputs"]["/mnt/host/experiments/clip-1.mkv"]:
        raise ValueError("decoder input changed since the recorded campaign")
    poison_image = Path("/tmp/capstone/poisoncap-plots/ffmpeg-poisoncap-full/ffmpeg")
    poison_build = json.loads((poison_image.parent / "manifest.json").read_text())
    if digest(poison_image) != poison_build["image_sha256"] or poison_build["nested_pool"] != "poisoncap":
        raise ValueError("PoisonCap image differs from its build identity")
    attempts = [json.loads(s) for s in (cap / "runs.jsonl").read_text().splitlines()]
    poison_attempts = [json.loads(s) for s in (poison / "runs.jsonl").read_text().splitlines()]
    result = {}
    for batches in BATCHES:
        oracle = source[f"ffmpeg-f30-b{batches}-pool0"]["expected_stdout"]
        if source[f"ffmpeg-f30-b{batches}-pool2"]["expected_stdout"] != oracle:
            raise ValueError("Capstone policy oracle differs")
        for mode in (0, 2):
            label = f"b{batches}-m{mode}"
            cid = f"ffmpeg-f30-b{batches}-pool{mode}"
            crows = [x for x in attempts if x["point"]["id"] == cid]
            if len(crows) != 3 or any(x["status"] != "pass" for x in crows):
                raise ValueError(f"{cid}: need three passing Capstone attempts")
            cap_records = []
            for rep in range(3):
                folder = cap / f"{cid}-{rep}"
                stdout = (folder / "stdout").read_text()
                stderr = (folder / "stderr").read_text()
                if stdout != oracle:
                    raise ValueError(f"{cid}-{rep}: output oracle mismatch")
                outer = line(stderr, f"EXP-MEM phase=released-{batches-1} ")
                pool = line(stderr, "EXP-POOL ")
                if pool["mode"] != mode or outer["live"] or pool["payload"] <= 0:
                    raise ValueError(f"{cid}-{rep}: wrong policy or bad accounting")
                if mode == 0 and pool["revoke"] or mode == 2 and pool["revoke"] <= 0:
                    raise ValueError(f"{cid}-{rep}: Sublet path not observed")
                cap_records.append(dict(stdout_sha256=digest(folder / "stdout"),
                                        stderr_sha256=digest(folder / "stderr"),
                                        outer_heap=outer, pool=pool))
            prows = [x for x in poison_attempts
                     if x["batches"] == batches and x["mode"] == mode]
            if len(prows) != 1 or prows[0]["returncode"] != 0 or not prows[0]["oracle_match"]:
                raise ValueError(f"{label}: missing successful PoisonCap attempt")
            stdout_path = poison / (label + ".stdout")
            stderr_path = poison / (label + ".stderr")
            stdout = stdout_path.read_text()
            stderr = stderr_path.read_text()
            if stdout != oracle:
                raise ValueError(f"{label}: independent output oracle mismatch")
            policy = line(stderr, "FFPOOL-POLICY ")
            if policy != dict(mode=mode, payload_reservation=4194304):
                raise ValueError(f"{label}: wrong policy or payload reservation")
            pool_phases = phase_lines(stderr, "FFPOOL-MEM phase=", batches)
            cheri = line(stderr, f"EXP-CHERI phase=released-{batches-1} ")
            inner = pool_phases[-1]
            adapter = [fields(s) for s in stderr.splitlines()
                       if s.startswith("FF2_POISONCAP ")][-1]
            if (cheri["revocation"] != 0 or cheri["heap_error"] or cheri["shadow_error"]
                    or inner["payload_used"] <= 0 or
                    (mode == 0 and (inner["snapshots"] or inner["sweeps"])) or
                    (mode == 2 and (not inner["snapshots"] or not inner["sweeps"]))):
                raise ValueError(f"{label}: nested allocator accounting failed")
            if inner["snapshots"] != adapter["snapshot_bytes"] or inner["sweeps"] != adapter["sweeps"]:
                raise ValueError(f"{label}: inner snapshots and backend disagree")
            if any(r["pool"]["payload"] != inner["payload_used"] for r in cap_records):
                raise ValueError(f"{label}: different inner payload work")
            result[label] = dict(batches=batches, mode=mode,
                                 oracle_sha256=hashlib.sha256(oracle.encode()).hexdigest(),
                                 capstone=cap_records,
                                 poisoncap=dict(stdout_sha256=digest(stdout_path),
                                                stderr_sha256=digest(stderr_path),
                                                policy=policy, phases=pool_phases,
                                                jemalloc=cheri, adapter=adapter))
    return result


def plot(data, out):
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BATCHES))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for mode, color, offset, name in ((0, "#3f6b8a", -.18, "Spatial"),
                                      (2, "#ef9d4e", .18, "Temporal")):
        cap = [data[f"b{b}-m{mode}"]["capstone"][0]["outer_heap"]["peak"] / MIB
               for b in BATCHES]
        poi = [data[f"b{b}-m{mode}"]["poisoncap"]["jemalloc"]["allocated"] / MIB
               for b in BATCHES]
        axes[0].bar(x + offset, cap, .32, color=color, label=name)
        axes[1].bar(x + offset, poi, .32, color=color, label=name)
    axes[0].set_title("Capstone: outer application heap peak")
    axes[1].set_title("PoisonCap: jemalloc allocated after release")
    for ax in axes:
        ax.set_xticks(x, [str(b) for b in BATCHES])
        ax.set_xlabel("Independent 30-frame streams")
        ax.set_ylabel("MiB")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .15),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=.31, wspace=.27)
    fig.text(.08, .035, "Within-platform pairs only. Both pool modes use the same 4 MiB payload reservation.\n"
             "The two panels use different allocator ledgers; node metadata and kernel shadow are excluded.", fontsize=8)
    fig.savefig(out / "ffmpeg-pool-retention.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "ffmpeg-pool-retention.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    temporal = [data[f"b{b}-m2"]["poisoncap"] for b in BATCHES]
    axes[0].plot(BATCHES, [p["adapter"]["snapshot_bytes"] / MIB for p in temporal],
                 "o-", color="#3f6b8a", lw=2, label="Retained snapshot")
    axes[0].set_ylabel("MiB")
    axes[0].set_title("Snapshot backing persists across streams")
    for key, name, color in (("copied_bytes", "Copied", "#695798"),
                             ("poison_bytes", "Poisoned", "#ae5959"),
                             ("clear_bytes", "Cleared", "#a1b66b")):
        axes[1].plot(BATCHES, [p["adapter"][key] / MIB for p in temporal],
                     "o-", color=color, lw=2, label=name)
    axes[1].set_title("Explicit payload rewriting in temporal arm")
    axes[1].set_ylabel("Cumulative MiB")
    for ax in axes:
        ax.set_xticks(BATCHES)
        ax.set_xlabel("Independent 30-frame streams")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.subplots_adjust(bottom=.23, wspace=.27)
    fig.text(.08, .035, "Whole FFmpeg 9.0.1 decoder; exact frame oracle in all 24 runs.\n"
             "Rewriting counts are bytes touched, not elapsed time or memory bandwidth.", fontsize=8)
    fig.savefig(out / "ffmpeg-pool-snapshot-work.png", dpi=220, bbox_inches="tight")
    fig.savefig(out / "ffmpeg-pool-snapshot-work.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--capstone", type=Path)
    source.add_argument("--data", type=Path)
    p.add_argument("--poisoncap", type=Path)
    p.add_argument("--points", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.data:
        document = json.loads(args.data.read_text())
        if document.get("schema") != 1:
            raise ValueError("unknown study data")
        data = document["cells"]
    else:
        if not (args.poisoncap and args.points):
            p.error("--capstone requires --poisoncap and --points")
        data = load(args.capstone, args.poisoncap, args.points)
        document = dict(schema=1, workload="FFmpeg 9.0.1 MPEG-4 decoder, 30 frames per stream",
                        cells=data, capstone_binary_sha256=digest(
                            "/tmp/capstone/domain-process-runtime/share/experiments/ffmpeg-pool-fixed.dom"),
                        poisoncap_binary_sha256=digest(
                            "/tmp/capstone/poisoncap-plots/ffmpeg-poisoncap-full/ffmpeg"),
                        input_sha256=digest("/tmp/capstone/domain-process-runtime/share/experiments/clip-1.mkv"),
                        source_sha256={
                            "poisoncap_decoder_entry": digest(Path(__file__).with_name("ffmpeg-poisoncap-decode.c")),
                            "poisoncap_pool_backend": digest(Path(__file__).resolve().parents[2] /
                                                              "ports/ffmpeg/buffer-pool/src/cheribsd/poisoncap-payload.c"),
                            "shared_builder": digest(Path(__file__).resolve().parents[1] /
                                                     "applications/comparison-build.py")})
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "data.json").write_text(json.dumps(document, indent=2) + "\n")
    plot(data, args.out)
    print("Validated 3 workloads x 2 modes: PoisonCap 6/6, Capstone 18/18, exact frame oracle")


if __name__ == "__main__":
    main()
