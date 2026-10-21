#!/usr/bin/env python3
"""Run the wmem lifetime controls on one arm, in one boot of the virtual platform, and record them.

    run-lifetime-controls.py --arm {protected,release} --build B --kit K --work NEW_DIR \
        --llvm-bin BIN --revision REV.json [--timeout S]

B is the wmem port configured with the capstone-application preset, -DWM_CORPUS_DIR=<this
directory>, WM_SUBLET=ON for protected and OFF for release; the runner refuses a build whose
WM_SUBLET does not match the arm. K holds the platform: qemu-system-riscv64, images/{Image,
fw_jump.elf,rootfs.ext2}, adapter/capstone-vexec, adapter/module/capstone_vm.ko. One boot runs
every program of this directory in order, each as its own process under capstone-vexec
(capstone/runtime/virtual/run-staged.py: one hart, exact bounds), so a fault in one cannot hide
another's result.

What is recorded per program: the image's sha256, every `WM_CTL` line it printed (the addresses and
bounds of A, B, C and the parent references, each access with its call site and probe, each
check), its exit status, and the launcher's fault line if any, resolved to the function and to the
labelled instruction from the image's own symbols. The probe labels resolve to source lines by
reading this snapshot's sources, never from memory.

This runner records and compares against the pre-registered prediction
(docs/plans/2026-10-11-nested-lifetime-controls.md); a program with no result is an ERROR, never a
match. Exit 0 when every program produced a reading (matching or not), 1 otherwise.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import sdk_builtins  # noqa: E402  (C-81: refuse a builtins build before anything starts)
import verdicts as v  # noqa: E402

KIT_FILES = {"qemu": "qemu-system-riscv64", "kernel": "images/Image", "firmware": "images/fw_jump.elf",
             "rootfs": "images/rootfs.ext2", "launcher": "adapter/capstone-vexec",
             "module": "adapter/module/capstone_vm.ko"}
LABELS = {"wm_defect_probe": ("bug-corpora/wireshark/wmem-repros/shared/driver.c", "wm_defect_probe:"),
          "wm_defect_write": ("bug-corpora/wireshark/wmem-repros/shared/driver.c", "wm_defect_write:"),
          "wm_ctl_load_insn": ("bug-corpora/wireshark/wmem-repros/controls/lifetime/lifetime.h",
                               "wm_ctl_load_insn:"),
          "wm_ctl_store_insn": ("bug-corpora/wireshark/wmem-repros/controls/lifetime/lifetime.h",
                                "wm_ctl_store_insn:")}
SUBLET = {"protected": "ON", "release": "OFF"}


def fsha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def programs():
    found = sorted(d for d in HERE.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    for n, d in enumerate(found):
        if int(d.name[:2]) != n:
            sys.exit(f"{d}: control numbers must be dense from zero")
    return [(n, d.name, f"{d.name[:2]}-{d.name.split('_', 2)[2].replace('_', '-')}") for n, d in enumerate(found)]


def label_lines():
    """Each probe label's file:line in this snapshot, by the line that defines it."""
    out = {}
    for label, (rel, needle) in LABELS.items():
        lines = (REPO / "capstone" / rel).read_text().splitlines()
        hits = [i + 1 for i, line in enumerate(lines) if needle in line]
        if len(hits) != 1:
            sys.exit(f"{label}: {len(hits)} definitions in {rel}")
        out[label] = f"capstone/{rel}:{hits[0]}"
    return out


def label_addresses(llvm_bin, image):
    """Link-time addresses of the probe labels in IMAGE (any symbol type: they are asm labels)."""
    nm = subprocess.run([str(llvm_bin / "llvm-nm"), "-n", "--defined-only", str(image)],
                        capture_output=True, text=True, check=True).stdout
    found = {}
    for line in nm.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in LABELS:
            found[parts[2]] = int(parts[0], 16)
    return found


GATE_HEAD = """#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }
chmod 666 /dev/capstone-vm
export CAPSTONE_EXEC_DIAGNOSTICS=1
"""
GATE_RUN = """echo "@@BEGIN {stem}"
cd /tmp && /mnt/vm/capstone-vexec /mnt/vm/{stem}.dom 0 {n} > /tmp/{stem}.out 2>&1
rc=$?
cd /mnt/vm
sed 's/^/@@OUT {stem} /' /tmp/{stem}.out
echo "@@RC {stem} rc=$rc"
"""


def parse(stem, log):
    lines = [m.group(1) for m in re.finditer(rf"^@@OUT {re.escape(stem)} (.*)$", log, re.M)]
    rc = re.search(rf"^@@RC {re.escape(stem)} rc=(\d+)$", log, re.M)
    rec = {"rc": int(rc.group(1)) if rc else None, "lines": lines}
    text = "\n".join(lines)
    refs = {}
    for m in re.finditer(r"^WM_CTL ref (\S+) addr=(\S+) base=(\S+) end=(\S+)$", text, re.M):
        refs[m.group(1)] = {"addr": m.group(2), "base": m.group(3), "end": m.group(4)}
    rec["refs"] = refs
    accesses = []
    for line in lines:
        m = re.match(r"WM_CTL access (read|write) (\S+) via=(\S+) addr=(\S+) len=(\d+) probe=(\S+) site=(\S+)$", line)
        if m:
            accesses.append({"op": m.group(1), "what": m.group(2), "via": m.group(3), "addr": m.group(4),
                             "len": int(m.group(5)), "probe": m.group(6), "site": m.group(7), "done": False})
            continue
        m = re.match(r"WM_CTL done (read|write) (\S+)$", line)
        if m and accesses and accesses[-1]["op"] == m.group(1) and accesses[-1]["what"] == m.group(2):
            accesses[-1]["done"] = True
    rec["accesses"] = accesses
    rec["checks"] = re.findall(r"^WM_CTL check (.*)$", text, re.M)
    rec["notes"] = [l[len("WM_CTL "):] for l in lines if l.startswith("WM_CTL ") and not
                    re.match(r"WM_CTL (ref|access|done|check) ", l)]
    rec["ready"] = bool(re.search(r"^WM_DEFECT case=\d+ ready$", text, re.M))
    rec["completed"] = bool(re.search(r"^WM_DEFECT case=\d+ mode=0 completed$", text, re.M))
    rec["control_failed"] = [l for l in lines if "CONTROL-FAILED" in l]
    rec["fault_line"] = next((l for l in lines if "domain fault cause=" in l), None)
    return rec


def check_ok(rec, name):
    return any(c.startswith(name + " ") and c.endswith(" ok") for c in rec["checks"])


def judge(n, stem, arm, rec):
    """The pre-registered prediction for (control, arm), against what the run printed."""
    if rec["rc"] is None:
        return "ERROR", "no result: the program's @@RC line is absent"
    if rec["control_failed"]:
        return "ERROR", "the control refused: " + rec["control_failed"][0]
    fault = rec.get("fault")
    arm_line = f"arm={arm}" in rec["notes"]
    if not arm_line:
        return "ERROR", f"the program did not report arm={arm}"
    every_done = rec["accesses"] and all(a["done"] for a in rec["accesses"])
    if n in (0, 1, 2) or (n == 3 and arm == "release"):
        want = {0: ["A-former-range-after-links", "A-former-range-written", "guard-intact"]
                   + (["A-former-range-via-generation", "A-former-range-written-via-generation"]
                      if arm == "protected" else []),
                1: ["B-intact-after-A-released", "B-written"],
                2: ["C-written"],
                3: ["C-written", "stale-A-reads-C"]}[n]
        missing = [c for c in want if not check_ok(rec, c)]
        extra = []
        if n == 0:
            extra = [x for x in ("ok: free, A's length", "ok: the recycler holds A's chunk alone")
                     if not any(x in note for note in rec["notes"])]
        if n in (2, 3) and not any(re.search(r"equal=yes", note) for note in rec["notes"]):
            extra.append("C's address equals A's")
        if fault or not rec["completed"] or rec["rc"] != 0 or not every_done or missing or extra:
            return "DIFFERS", (f"fault={fault} completed={rec['completed']} rc={rec['rc']} "
                               f"all_accesses_done={bool(every_done)} failed_checks={missing} missing={extra}")
        return "AS PREDICTED", "completed; every access done; " + ", ".join(want + extra or ["checks"]) + " ok"
    # n == 3, protected: the stale read must fault at the labelled read with cause 25,
    # after C's accesses completed and the ready mark printed.
    pre = [a for a in rec["accesses"] if a["what"] != "stale-A"]
    stale = [a for a in rec["accesses"] if a["what"] == "stale-A"]
    ok = (fault and fault["cause"] == 25 and fault["label"] == "wm_defect_probe" and rec["ready"]
          and not rec["completed"] and pre and all(a["done"] for a in pre) and check_ok(rec, "C-written")
          and stale and not stale[0]["done"] and any("equal=yes" in note for note in rec["notes"]))
    if ok:
        return "AS PREDICTED", "C's accesses completed; the stale read faulted at wm_defect_probe, cause 25"
    return "DIFFERS", (f"fault={fault} ready={rec['ready']} completed={rec['completed']} "
                       f"stale_done={[a['done'] for a in stale]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(SUBLET))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--revision", type=Path, required=True, help="JSON: the snapshot's revision record")
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    sdk_builtins.gate([a.build], who="wireshark:run-lifetime-controls.py")
    cache = (a.build / "CMakeCache.txt").read_text()
    sublet = re.search(r"^WM_SUBLET:BOOL=(\S+)", cache, re.M)
    sdk = re.search(r"^CAPSTONE_SDK:PATH=(\S+)", cache, re.M)
    corpus = re.search(r"^WM_CORPUS_DIR:PATH=(\S+)", cache, re.M)
    if (sublet.group(1) if sublet else "OFF") != SUBLET[a.arm] or not sdk:
        sys.exit(f"{a.build} is not the {a.arm} arm's build (WM_SUBLET={SUBLET[a.arm]}, a CAPSTONE_SDK)")
    if not corpus or Path(corpus.group(1)).resolve() != HERE:
        sys.exit(f"{a.build} was not configured with WM_CORPUS_DIR={HERE}")
    sdk_json = json.loads((Path(sdk.group(1)) / "sdk.json").read_text())
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / "stage"
    stage.mkdir()
    record = {"arm": a.arm, "revision": json.loads(a.revision.read_text()),
              "platform": {k: fsha(a.kit / f) for k, f in KIT_FILES.items()},
              "configuration": {"WM_SUBLET": SUBLET[a.arm], "profile": "virtual", "exact_bounds": True,
                                "harts": 1, "allocator": "WMEM_ALLOCATOR_BLOCK", "release": "wmem_free",
                                "build_type": (re.search(r"^CMAKE_BUILD_TYPE:STRING=(\S*)", cache, re.M)
                                               or [None, None])[1],
                                "sdk_json_sha256": fsha(Path(sdk.group(1)) / "sdk.json"),
                                "sdk_cc_sha256": fsha(sdk_json["cc"]) if Path(sdk_json.get("cc", "")).is_file()
                                else "unrecorded",
                                "wmem_variant": "sublet (patch 0001)" if a.arm == "protected" else "reference",
                                "patch_sha256": fsha(REPO / "capstone/ports/wireshark/wmem/patches/"
                                                     "wireshark-4.6.8-0001-wmem-sublet-lifetimes.patch")},
              "probe_lines": label_lines(), "runner_sha256": fsha(__file__),
              "run_staged_sha256": fsha(REPO / "capstone/runtime/virtual/run-staged.py"),
              "programs": {}}
    progs = programs()
    script = GATE_HEAD
    for n, name, stem in progs:
        image = a.build / "bin" / stem
        if not image.is_file():
            sys.exit(f"{image}: not built")
        shutil.copy2(image, stage / f"{stem}.dom")
        (stage / f"{stem}.dom").chmod(0o755)
        record["programs"][stem] = {"control": name, "image_sha256": fsha(stage / f"{stem}.dom"),
                                    "labels": {k: hex(x) for k, x in label_addresses(a.llvm_bin, image).items()}}
        script += GATE_RUN.format(stem=stem, n=n)
    script += "rmmod capstone_vm\necho VIRTUAL_STAGED_DONE\n"
    (stage / "gate.sh").write_text(script)
    record["gate_sha256"] = hashlib.sha256(script.encode()).hexdigest()
    shutil.copy2(a.kit / KIT_FILES["launcher"], stage / "capstone-vexec")
    (stage / "capstone-vexec").chmod(0o755)
    shutil.copy2(a.kit / KIT_FILES["module"], stage / "capstone_vm.ko")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(a.kit) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    import time
    started = time.time()
    record["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    run = subprocess.run([sys.executable, str(REPO / "capstone/runtime/virtual/run-staged.py"),
                          "--qemu", str(a.kit / KIT_FILES["qemu"]), "--images", str(a.kit / "images"),
                          "--stage", str(stage), "--work", str(a.work / "guest"), "--timeout", str(a.timeout)],
                         env=env)
    log = (a.work / "guest/serial.log").read_text(errors="replace").replace("\r", "")
    record["boot_completed"] = run.returncode == 0
    record["wall_seconds"] = round(time.time() - started, 1)
    all_read = True
    for n, name, stem in progs:
        rec = parse(stem, log)
        image = a.build / "bin" / stem
        if rec["fault_line"]:
            f = v.domain_fault(rec["fault_line"], v.Symbols(a.llvm_bin, image))
            m = v.FAULT_LINE.search(rec["fault_line"])
            link_pc = int(m.group(2), 16) - int(m.group(4), 16) + v.Symbols(a.llvm_bin, image).base
            labels = label_addresses(a.llvm_bin, image)
            label = next((k for k, x in labels.items() if x == link_pc), None)
            rec["fault"] = {"cause": f.cause, "pc": hex(f.pc), "address": m.group(3), "code_start": m.group(4),
                            "link_pc": hex(link_pc), "function": f.symbol, "offset": f.offset, "label": label,
                            "probe_line": record["probe_lines"].get(label)}
        else:
            rec["fault"] = None
        rec["prediction"], rec["evidence"] = judge(n, stem, a.arm, rec)
        all_read &= rec["prediction"] != "ERROR"
        record["programs"][stem].update(rec)
        print(f"{a.arm:9s} {stem:22s} {rec['prediction']:12s} {rec['evidence'][:150]}", flush=True)
    (a.work / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(a.work / "result.json")
    return 0 if all_read and record["boot_completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
