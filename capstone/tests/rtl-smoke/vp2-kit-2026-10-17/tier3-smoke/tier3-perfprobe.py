#!/usr/bin/env python3
"""tier3-perfprobe.py (--kit <QEMU kit> | --board-kit <board kit>) <out dir>: the perf evaluation's first boot (2026-10-26),
the runtime lane's counter probe plus the Tier 1 contract of dev's new module revision (c2b137cdfee6: registry, v7 image,
CV_STEP 136) in its bench build. One boot through boardvm, steps in this order:
  params      the module's bench_* parameters (the default build has none)
  ok          contract.dom ok argument, CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 (vm-contract.sh's control)
  probe       probe.dom: 10 PROBE lines, then "PROBE done", exit 0
  stale       contract.dom stale (vm-contract.sh's identity reading: cause 25, exit 139)
After each step it prints the module's `capstone_vm: bench ...` dmesg lines that are new since the previous step.
Prints the raw step texts and one READ line per check; exit 0 when the boot completed and every check was read, 2
otherwise (no data is an error, not a zero). The verdict is the pre-registration's, not this script's."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

P = Path("/mnt/hdd") / Path.home().name / "capstone/perf/probe"
IMAGES = {"probe.dom": "0100285f9c66605d", "contract.dom": "072bc796396597a4"}
PARAMS = "/sys/module/capstone_vm/parameters"
BENCH = (f"dmesg | grep -a 'capstone_vm: bench ' | sed 's/^/BENCH /'; "
         f"for p in bench_cycles bench_instret bench_entries; do echo \"PARAM $p $(cat {PARAMS}/$p 2>&1)\"; done")


def s16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    k = ap.add_mutually_exclusive_group(required=True)
    k.add_argument("--kit", type=Path)
    k.add_argument("--board-kit", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    for name, want in IMAGES.items():
        if s16(P / name) != want:
            print(f"PERF STOP: {P / name} is {s16(P / name)}, not {want}"); sys.exit(2)
    Batch, kit = boardvm.select(a.kit, a.board_kit)
    a.out.mkdir(parents=True, exist_ok=False)
    b = Batch(a.out / "stage")
    probe = b.put("images/probe.dom", P / "probe.dom")
    contract = b.put("images/contract.dom", P / "contract.dom")
    root = str(Path(probe).parent.parent)
    vx = f"{root}/capstone-vexec"
    cenv = "export CAPSTONE_VM_CONTRACT=environment CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1;"
    b.step("params", f"ls {PARAMS} | grep bench | sed 's/^/BPARAM /'; {BENCH}")
    b.step("ok", f"{cenv} {vx} {contract} ok argument; echo \"RUN:ok:rc=$?\"; {BENCH}")
    b.step("probe", f"export CAPSTONE_EXEC_DIAGNOSTICS=1; {vx} {probe}; echo \"RUN:probe:rc=$?\"; {BENCH}")
    b.step("stale", f"{cenv} {vx} {contract} stale; echo \"RUN:stale:rc=$?\"; {BENCH}")
    b.bounds = {"params": 60, "ok": 300, "probe": 1200, "stale": 300}
    runs, transcript, completed = b.execute(kit, a.out / "guest", timeout=3600)
    print(f"PERF kit {kit} boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; steps {len(runs)}")
    rc = 0 if completed else 2
    seen = set()
    for name in ("params", "ok", "probe", "stale"):
        text, result = runs.get(name, ("", {"kind": "none"}))
        print(f"## step {name} result {json.dumps(result, sort_keys=True)}")
        print(text.rstrip())
        if re.search(r"Oops|Unable to handle|BUG:", text):
            print(f"READ {name}: KERNEL OOPS in the step's text"); rc = 2
        if result.get("kind") == "none":
            print(f"READ {name}: NO DATA"); rc = 2; continue
        new = [l for l in re.findall(r"^BENCH (.*)$", text, re.M) if l not in seen]
        seen.update(new)
        for l in new:
            print(f"READ {name}: bench {l.strip()}")
        if name == "params":
            bp = re.findall(r"^BPARAM (\S+)$", text, re.M)
            print(f"READ params: {len(bp)} bench parameters {bp}")
            rc = rc if len(bp) == 3 else 2
        elif name == "ok":
            r = re.search(r"^RUN:ok:rc=(\d+)$", text, re.M)
            okline = re.search(r"^VIRTUAL_APPLICATION_OK ", text, re.M)
            print(f"READ ok: rc {r.group(1) if r else None}, VIRTUAL_APPLICATION_OK {'present' if okline else 'ABSENT'}")
            rc = rc if r else 2
        elif name == "probe":
            r = re.search(r"^RUN:probe:rc=(\d+)$", text, re.M)
            lines = [l for l in re.findall(r"^PROBE (.*)$", text, re.M) if l.strip() != "done"]
            done = re.search(r"^PROBE done$", text, re.M)
            print(f"READ probe: rc {r.group(1) if r else None}, {len(lines)} PROBE lines, done {'present' if done else 'ABSENT'}")
            for l in lines:
                print(f"READ probe line: {l.strip()}")
            rc = rc if r else 2
        elif name == "stale":
            r = re.search(r"^RUN:stale:rc=(\d+)$", text, re.M)
            f = re.search(r"domain fault cause=(\d+) pc=0x([0-9a-f]+).*?code=0x([0-9a-f]+)-", text)
            print(f"READ stale: rc {r.group(1) if r else None}, fault "
                  + (f"cause {f.group(1)} at code+{int(f.group(2), 16) - int(f.group(3), 16):#x}" if f else "ABSENT"))
            rc = rc if r else 2
    sys.exit(rc)


if __name__ == "__main__":
    main()
