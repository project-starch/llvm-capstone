#!/usr/bin/env python3
"""Validate SQLite 3.22 study logs and draw memory-only, application-level figures.

The input manifest names preserved raw transcripts and their selected backing
reservations. Figures deliberately exclude QEMU time and platform metadata.
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

PHASES = (100, 110, 120, 130, 140, 142, 145, 150, 160, 161, 170, 180,
          190, 200, 210, 230, 240, 250, 260, 270, 280, 290, 300, 310,
          320, 400, 410, 500, 510, 520, 980, 990)
MIB = 1024 * 1024
ORACLE = re.compile(r"STUDY-ORACLE phase=(\d+) rows=(\d+) hash=([0-9a-f]{16})")
CAP = re.compile(r"STUDY-CAP phase=(\d+) live=(\d+) peak=(\d+) oom=(\d+) blocks=(\d+) atom=(\d+)")
MEM = re.compile(r"STUDY-MEM phase=(\d+) (.+)")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oracle(text):
    matches = ORACLE.findall(text)
    if tuple(int(p) for p, _, _ in matches) != PHASES:
        raise ValueError("expected exactly 32 ordered SQL-result phases")
    return {int(p): {"rows": int(n), "hash": h} for p, n, h in matches}


def records(text, kind):
    out = {}
    if kind == "capstone":
        for p, live, peak, oom, blocks, atom in CAP.findall(text):
            out[int(p)] = dict(live=int(live), peak_live=int(peak), oom=int(oom),
                               blocks=int(blocks), atom=int(atom))
    else:
        for p, tail in MEM.findall(text):
            out[int(p)] = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", tail)}
    if tuple(out) != PHASES:
        raise ValueError(f"missing phase-level {kind} accounting: {tuple(out)}")
    return out


def read_arm(label, spec, expected):
    path = Path(spec["log"])
    text = path.read_text(errors="replace")
    result = oracle(text)
    if result != expected or "TOTAL" not in text:
        raise ValueError(f"{label}: result oracle or completion differs from native")
    kind = "capstone" if label.startswith("capstone") else "poisoncap"
    phases = records(text, kind)
    if "Successful lookasides:" not in text or not re.search(r"Successful lookasides:\s+0\b", text):
        raise ValueError(f"{label}: lookaside-off configuration was not observed")
    last = phases[990]
    if last["oom"] or (kind == "poisoncap" and last["revoke_errors"]):
        raise ValueError(f"{label}: allocator/revocation failure in completed run")
    if kind == "capstone":
        if "DROPPED 0" not in text or "__CAPSTONE_SPEEDTEST1_RAN__" not in text:
            raise ValueError(f"{label}: lost output or missing completion marker")
        if label == "capstone-sublet" and not re.search(r"sublet:.*revoke=\d+.*init=\d+", text):
            raise ValueError("Sublet path not observed")
        reservations = {k: spec[k] for k in ("pool", "tables") if k in spec}
    else:
        if last["heap"] != spec["heap"]:
            raise ValueError(f"{label}: effective heap differs from manifest")
        reservations = dict(heap=last["heap"], links=((last["links"] + 4095) // 4096) * 4096,
                            quarantine_table=last["qtable"])
    total = sum(reservations.values())
    return dict(log=str(path), log_sha256=digest(path), binary_sha256=digest(spec["binary"]),
                reservations=reservations, total_reservation=total,
                link_requested_bytes=last["links"] if kind == "poisoncap" else None,
                phases={str(p): phases[p] for p in PHASES},
                oracle={str(p): result[p] for p in PHASES})


def save(fig, out, stem):
    fig.savefig(out / (stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def backing_figure(arms, out):
    labels = ["Capstone\nspatial", "Capstone\n+ Sublet",
              "PoisonCap\nspatial", "PoisonCap\ntemporal†"]
    keys = ["capstone", "capstone-sublet", "poisoncap-spatial", "poisoncap-temporal"]
    components = [
        ("Heap / pool", [arms[k]["reservations"].get("pool", arms[k]["reservations"].get("heap", 0)) for k in keys], "#3f6b8a"),
        ("App allocator tables / links", [arms[k]["reservations"].get("tables", arms[k]["reservations"].get("links", 0)) for k in keys], "#ef9d4e"),
        ("Static quarantine table", [arms[k]["reservations"].get("quarantine_table", 0) for k in keys], "#a1b66b"),
    ]
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    x = np.arange(4)
    bottom = np.zeros(4)
    for name, vals, color in components:
        height = np.array(vals) / MIB
        ax.bar(x, height, bottom=bottom, color=color, label=name, width=.68)
        bottom += height
    for i, value in enumerate(bottom):
        ax.text(i, value + .10, f"{value:.2f}", ha="center", fontsize=9)
    cap_delta = (arms[keys[1]]["total_reservation"] - arms[keys[0]]["total_reservation"]) / MIB
    poi_delta = (arms[keys[3]]["total_reservation"] - arms[keys[2]]["total_reservation"]) / MIB
    ax.set_xticks(x, labels)
    ax.set_ylabel("Application-visible reserved backing (MiB)")
    ax.set_ylim(0, max(bottom) * 1.19)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.set_title("SQLite 3.22 speedtest1 main, size 1: 32 matching phases")
    fig.subplots_adjust(bottom=.31)
    fig.text(.12, .015, f"Within-platform increment: Sublet +{cap_delta:.2f} MiB; PoisonCap corrected +{poi_delta:.2f} MiB.\n"
             "† Corrected full-queue revocation; published policy drained without revocation.\n"
             "Heap includes inline control bytes. Kernel shadow and Capstone node metadata excluded.", fontsize=8)
    save(fig, out, "sqlite-backing")


def phase_figure(arms, out):
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.5), sharex=True)
    x = np.arange(len(PHASES))
    c = arms["capstone"]["phases"]
    s = arms["capstone-sublet"]["phases"]
    p = arms["poisoncap-spatial"]["phases"]
    t = arms["poisoncap-temporal"]["phases"]
    axes[0].plot(x, [c[str(k)]["live"] / MIB for k in PHASES], color="#3f6b8a", lw=2,
                 label="Capstone spatial live")
    axes[0].plot(x, [s[str(k)]["live"] / MIB for k in PHASES], color="#ef9d4e", lw=1.2,
                 ls="--", label="Capstone + Sublet live (overlaps)")
    axes[0].axhline(arms["capstone"]["reservations"]["pool"] / MIB,
                    color="#777", ls=":", lw=1, label="Pool reservation")
    axes[0].set_title("Capstone: rounded capacity checked out at each phase end")
    axes[1].plot(x, [p[str(k)]["live"] / MIB for k in PHASES], color="#3f6b8a", lw=2,
                 label="PoisonCap spatial live")
    axes[1].plot(x, [t[str(k)]["live"] / MIB for k in PHASES], color="#ef9d4e", ls="--", lw=1.3,
                 label="PoisonCap temporal live (overlaps)")
    axes[1].plot(x, [t[str(k)]["quarantine"] / MIB for k in PHASES], color="#ae5959", lw=2,
                 label="Temporal quarantine")
    axes[1].plot(x, [t[str(k)]["held"] / MIB for k in PHASES], color="#695798", lw=1,
                 label="Temporal live + quarantine")
    axes[1].set_title("PoisonCap: quarantine remains unavailable after SQL phases")
    axes[1].set_xticks(x, [str(k) for k in PHASES], rotation=65, fontsize=7)
    for ax in axes:
        ax.set_ylabel("MiB")
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.text(.12, .015, "Phase-end snapshots; intra-phase peaks are reported in the JSON. QEMU time is excluded.", fontsize=8)
    save(fig, out, "sqlite-release-refill")


def policy_figure(arms, published, out):
    final = lambda a: a["phases"]["990"]
    a, b = final(published), final(arms["poisoncap-temporal"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7))
    names = ["Published", "Corrected"]
    full = [a["full_drains"], b["full_drains"]]
    revoke = [a["revokes"], b["revokes"]]
    x = np.arange(2)
    axes[0].bar(x - .17, full, .32, color="#ae5959", label="Full drains")
    axes[0].bar(x + .17, revoke, .32, color="#3f6b8a", label="Revocations")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Completed operations")
    axes[0].set_title("Quarantine policy paths")
    axes[0].set_ylim(0, 8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].bar(x - .17, [a["poison_bytes"] / MIB, b["poison_bytes"] / MIB],
                .32, color="#ae5959", label="Poisoned")
    axes[1].bar(x + .17, [a["clear_bytes"] / MIB, b["clear_bytes"] / MIB],
                .32, color="#a1b66b", label="Cleared")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("Explicit payload bytes (MiB)")
    axes[1].set_title("Payload rewriting")
    axes[1].set_ylim(0, 25)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes: ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.20)
    fig.text(.12, .025, "Both use an 8 MiB heap and return the same 32 SQL result hashes; no timing inference.", fontsize=8)
    save(fig, out, "sqlite-poisoncap-policy")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path)
    source.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data:
        result = json.loads(args.data.read_text())
        if result.get("schema") != 1 or tuple(result.get("phase_ids", ())) != PHASES:
            raise ValueError("unrecognized study data or phase set")
        arms = result["arms"]
        published = result["published_policy"]
        backing_figure(arms, args.out)
        phase_figure(arms, args.out)
        policy_figure(arms, published, args.out)
        print("Redrew the three figures from curated study data")
        return
    manifest = json.loads(args.inputs.read_text())
    native_path = Path(manifest["native_oracle"])
    native = oracle(native_path.read_text())
    arms = {k: read_arm(k, v, native) for k, v in manifest["arms"].items()}
    published = read_arm("poisoncap-published", manifest["published"], native)
    assert set(arms) == {"capstone", "capstone-sublet", "poisoncap-spatial", "poisoncap-temporal"}
    assert published["phases"]["990"]["full_drains"] > 0
    assert published["phases"]["990"]["revokes"] == 0
    assert arms["poisoncap-temporal"]["phases"]["990"]["revokes"] > 0
    assert arms["poisoncap-temporal"]["phases"]["990"]["revoke_errors"] == 0
    result = dict(schema=1, workload="SQLite 3.22.0 speedtest1 main --size 1",
                  phase_ids=PHASES, native_oracle_sha256=digest(native_path),
                  native_oracle={str(p): native[p] for p in PHASES},
                  source_sha256={k: digest(v) for k, v in manifest["sources"].items()},
                  arms=arms, published_policy=published,
                  attempts=manifest["attempts"],
                  limits=["Application-visible reservations, not process RSS or total system memory",
                          "Capstone node and PoisonCap kernel shadow metadata are excluded",
                          "QEMU execution times are not comparable performance measurements",
                          "Exploratory single executions; no statistical uncertainty estimate"])
    (args.out / "data.json").write_text(json.dumps(result, indent=2) + "\n")
    backing_figure(arms, args.out)
    phase_figure(arms, args.out)
    policy_figure(arms, published, args.out)
    print(f"Validated {len(PHASES)} native-matched SQL phases in four arms and published-policy control")


if __name__ == "__main__":
    main()
