#!/usr/bin/env python3
"""The libc heap qualification of docs/plans/sublet-heap-qualification.md.

Runs the heap cases of contract.c through the Linux supervisor in a booted VM,
once on the HEAP=sublet image and once on the HEAP=level0 image, which is the
control. Every fault-* case must be a SIGSEGV on the sublet image and must run
to completion on level0; every heap-* case must complete on both. A case whose
verdict is the same on both images has not shown anything, so the level0 run
is not optional. The supervisor's own verdict lines are the evidence; this
script only checks them against the table and records the platform.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

FAULTS = ("fault-stale", "fault-reused", "fault-bounds", "fault-bounds-large",
          "fault-double-free", "fault-double-free-reused")
COMPLETES = ("healthy", "churn", "heap-bounds", "heap-neighbour", "heap-companion")
# What the supervisor must print per image: the sublet image passes everything,
# the control passes the completing cases and fails every fault case, because
# the access it is meant to catch simply succeeds there.
EXPECT = {"sublet": {m: "PASS" for m in FAULTS + COMPLETES},
          "level0": {**{m: "FAIL" for m in FAULTS}, **{m: "PASS" for m in COMPLETES}}}
HEAP_SYMBOLS = ("malloc", "free", "calloc", "realloc", "__libc_malloc", "__libc_free")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="capstone_vm state of a booted VM")
    parser.add_argument("--sublet-image", required=True, help="guest path of the HEAP=sublet image")
    parser.add_argument("--control-image", required=True, help="guest path of the HEAP=level0 image")
    parser.add_argument("--sublet-elf", type=Path, help="host ELF of the sublet image, for the symbol check")
    parser.add_argument("--control-elf", type=Path, help="host ELF of the control image")
    parser.add_argument("--nm", default=os.environ.get("CAPSTONE_LLVM_NM", "llvm-nm"))
    parser.add_argument("--platform", nargs="*", default=[], help="host files whose sha256 the report records")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]

    def call(*words, timeout=90, verdict=False):
        # The supervisor exits 1 on a FAIL verdict, and a FAIL is the expected
        # outcome of every fault case on the control image, so its calls are
        # judged by the verdict line, never by the exit status.
        result = subprocess.run([*cli, *words], text=True, capture_output=True, timeout=timeout, env=env)
        if result.returncode and not verdict:
            raise RuntimeError(f"{words[0]} exited {result.returncode}: {result.stderr}\n{result.stdout}")
        return result

    report = {"date": date.today().isoformat(), "platform": "QEMU supervised CALL, one hart",
              "verdicts": {}, "symbols": {}, "stats": {}, "sha256": {}}
    for path in args.platform:
        report["sha256"][Path(path).name] = sha256(path)

    # The linked heap: the image defines the public and musl-internal
    # allocation entry points once, and the object that defines them is the
    # selected heap's, which is the only heap object in the image's build tree.
    # Unreferenced markers such as the stats exports are dropped by the linker,
    # so the object file is the evidence, not a symbol name.
    for label, elf in (("sublet", args.sublet_elf), ("level0", args.control_elf)):
        if not elf:
            continue
        table = subprocess.run([args.nm, "--defined-only", str(elf)], text=True, capture_output=True, check=True).stdout
        defined = {line.split()[-1] for line in table.splitlines() if line.strip()}
        missing = [s for s in HEAP_SYMBOLS if s not in defined]
        build = elf.resolve().parent
        objects = {p.name for p in build.rglob("*.c.obj") if p.name in ("sublet_heap.c.obj", "level0.c.obj")}
        wanted = "sublet_heap.c.obj" if label == "sublet" else "level0.c.obj"
        report["symbols"][label] = {"missing": missing, "heap_objects": sorted(objects), "sha256": sha256(elf)}
        assert not missing and objects == {wanted}, report["symbols"][label]
        print(f"{label}: heap entry points defined, linked from {wanted} only: PASS")

    boot_id = call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout.strip()
    call("exec", "sh", "-c", "cp /mnt/host/application-supervisor /tmp/application-supervisor && "
         "chmod +x /tmp/application-supervisor")
    for label, image in (("sublet", args.sublet_image), ("level0", args.control_image)):
        # the integration smoke: the supervisor's own sequence on this image
        result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec", image, timeout=180, verdict=True)
        sequence = "application sequence: PASS" in result.stdout
        report["verdicts"][label] = {"sequence": "PASS" if sequence else "FAIL"}
        assert sequence, (label, result.stdout, result.stderr)
        print(f"{label}: application sequence: PASS")
        for mode, expected in EXPECT[label].items():
            before = json.loads(call("exec", "capstone-exec", "--stats").stdout)
            result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec",
                          image, "--mode", mode, timeout=200, verdict=True)
            after = json.loads(call("exec", "capstone-exec", "--stats").stdout)
            match = re.search(rf"application {re.escape(mode)}: (PASS|FAIL) \(wait status=(\d+)\)", result.stdout)
            assert match, (label, mode, result.stdout, result.stderr)
            verdict, status = match.group(1), int(match.group(2))
            report["verdicts"][label][mode] = {"supervisor": verdict, "wait_status": status,
                                              "expected": expected}
            report["stats"].setdefault(label, {})[mode] = {
                "nodes_allocated": after["nodes_allocated_total"] - before["nodes_allocated_total"],
                "live_after": {k: after[k] for k in ("live_domains", "live_regions", "live_bytes")}}
            # a faulted or exited domain leaves nothing behind either way
            assert after["live_domains"] == after["live_regions"] == after["live_bytes"] == 0, (label, mode, after)
            ok = verdict == expected
            print(f"{label} {mode}: supervisor {verdict}, expected {expected}: {'PASS' if ok else 'FAIL'}")
            assert ok, (label, mode, result.stdout)
    assert call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout.strip() == boot_id
    report["boot_id"] = boot_id
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print("heap qualification: PASS")


if __name__ == "__main__":
    main()
