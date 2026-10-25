#!/usr/bin/env python3
"""tier3-census-board.py <board kit> <out dir>: the runtime lane's pin_pte_census gate (R-61's open question: why the corpus
read no cause 15) in ONE board boot through boardvm, the same calls as its QEMU gate
(<mnt-hdd>/capstone/modfix/census-gate/stage/gate.sh, module 836fd06d = the QEMU build of runtime/pin-pte-census
a06950b3ab1b). The kit's module must be that branch's BOARD build 0fa26564e9450bf0 (bitstream 2's 215a09be plus the
default-off census); the counters are cumulative across the steps (one insmod, by the batch):
  off-walk  pin.dom walk with the census OFF, then the counters   (negative control: all 0)
  on        echo 1 > /sys/module/capstone_vm/parameters/pin_pte_census
  walk      pin.dom walk, counters
  vma       pin.dom vma, counters
  uaf       controls.dom uaf-malloc (faults), its fault line, counters
CAPSTONE_EXEC_DIAGNOSTICS=1 as on QEMU. Prints one READ line per counter read and the raw step texts; exit 0 when every
step ran and every read parsed, 2 otherwise (no data is an error, not a zero). Written 2026-10-26."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/tmp/capstone/wt-newarms/capstone/bug-corpora/tools")
import boardvm  # noqa: E402

H = Path("/mnt/hdd") / Path.home().name / "capstone"
PIN, PIN_SHA = H / "modfix/pinstale/pin.dom", "cc2b05a1b9befa1d"
CTL, CTL_SHA = H / "t3-run-treeB/uaf-repeat/stage/images/controls.dom", "b1decb0612856121"
KO_SHA = "0fa26564e9450bf0"
P = "/sys/module/capstone_vm/parameters"
CTR = ("echo \"CTR:{0}: pages=$(cat {1}/census_pages) absent=$(cat {1}/census_absent) write=$(cat {1}/census_write) "
       "dirty=$(cat {1}/census_dirty) young=$(cat {1}/census_young) write_clean=$(cat {1}/census_write_clean)\"")


def s16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    kit, out = Path(sys.argv[1]), Path(sys.argv[2])
    for p, want in ((PIN, PIN_SHA), (CTL, CTL_SHA), (kit / "adapter/module/capstone_vm.ko", KO_SHA)):
        if s16(p) != want:
            print(f"CENSUS STOP: {p} is {s16(p)}, not {want}"); sys.exit(2)
    out.mkdir(parents=True, exist_ok=False)
    batch = boardvm.BoardBatch(out / "stage")
    pin = batch.put("images/pin.dom", PIN)
    ctl = batch.put("images/controls.dom", CTL)
    root = str(Path(pin).parent.parent)
    env = "export CAPSTONE_EXEC_DIAGNOSTICS=1;"
    batch.step("off-walk", f"{env} {root}/capstone-vexec {pin} walk > /tmp/o 2>&1; echo \"RUN:off-walk:rc=$?\"; {CTR.format('off', P)}")
    batch.step("on", f"echo 1 > {P}/pin_pte_census; echo \"ON:$(cat {P}/pin_pte_census)\"")
    batch.step("walk", f"{env} {root}/capstone-vexec {pin} walk > /tmp/o 2>&1; echo \"RUN:on-walk:rc=$?\"; {CTR.format('walk', P)}")
    batch.step("vma", f"{env} {root}/capstone-vexec {pin} vma > /tmp/o 2>&1; echo \"RUN:on-vma:rc=$?\"; {CTR.format('vma', P)}")
    batch.step("uaf", f"{env} {root}/capstone-vexec {ctl} uaf-malloc > /tmp/o 2>&1; echo \"RUN:on-uaf:rc=$?\"; "
                      f"grep -a 'domain fault' /tmp/o | sed -E 's/ entry=.*//; s/^/  /'; {CTR.format('uaf', P)}")
    batch.bounds = {"off-walk": 600, "on": 60, "walk": 600, "vma": 600, "uaf": 600}
    runs, transcript, completed = batch.execute(kit, out / "guest", timeout=3600)
    print(f"CENSUS boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; steps {sorted(runs)}")
    rc = 0 if completed else 2
    for name in ("off-walk", "on", "walk", "vma", "uaf"):
        text, result = runs.get(name, ("", {"kind": "none"}))
        print(f"## step {name} result {json.dumps(result, sort_keys=True)}")
        print(text.rstrip())
        for m in re.findall(r"^(RUN:\S+|ON:\S+|  capstone-exec: domain fault.*)$", text, re.M):
            print(f"READ {name}: {m.strip()[:200]}")
        c = re.findall(r"^CTR:(\w+): pages=(\d+) absent=(\d+) write=(\d+) dirty=(\d+) young=(\d+) write_clean=(\d+)$", text, re.M)
        if name != "on":
            if not c:
                print(f"READ {name}: counters NO DATA"); rc = 2
            else:
                print("READ {}: counters pages/absent/write/dirty/young/write_clean = {}".format(name, "/".join(c[-1][1:])))
    sys.exit(rc)


if __name__ == "__main__":
    main()
