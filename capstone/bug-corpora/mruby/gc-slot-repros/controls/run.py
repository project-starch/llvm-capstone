#!/usr/bin/env python3
"""One boot of the mruby GC-slot controls: one arm, the four modes, each its own process.

    run.py --arm release|protected --rep N --work NEW_DIR --image CONTROLS_DOM \
        --qemu QEMU --images IMAGES --adapter ADAPTER --run-staged RUN_STAGED --llvm-bin BIN

Stages the image (controls/build.sh), the launcher and its module into a fresh directory, writes
gate.sh and boots it once through runtime/virtual/run-staged.py (one hart, exact bounds;
CAPSTONE_QEMU_LOCK must name this VM's own lock). In the guest each mode runs as
`capstone-vexec ./controls.dom MODE`, in the order parent, sibling, successor, stale, each with
its stdout and stderr appended (O_APPEND) to a guest file that is sent back gzip+base64 between
markers, and CAPSTONE_EXEC_DIAGNOSTICS=1 so the launcher writes its domain fault line to stderr.
The driver's last line says how many lines it printed; rep.json records whether all arrived. NEW_DIR/rep.json holds, per mode: the exit
status, every PROBE line (name and file:line), the ADDR/REUSE lines (addresses of A, B, C), the
outcome (completed: exit 0 and "CTL <mode> completed"; fault: the launcher's domain fault line,
its cause and pc, and the pc's function and file:line from the image's own symbols and debug
information; anything else as it is), and the program's own output lines. The console log stays
in NEW_DIR/guest/ and is not a result.

Exit: 0 when the boot completed and every mode left a reading (completed or fault), 1 otherwise.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import sdk_builtins  # noqa: E402  (C-81: refuse a builtins build before anything starts)
import verdicts as v  # noqa: E402

MODES = ("parent", "sibling", "successor", "stale")

GATE = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
chmod 666 /dev/capstone-vm
export CAPSTONE_EXEC_DIAGNOSTICS=1
O=/tmp/ctl; mkdir -p $O; chmod 777 $O
emit() { [ -f "$2" ] || return 0; echo "CTL_FILE:$1"; gzip -c "$2" | base64; echo "CTL_FILE_END"; }
for m in MODES; do
  : > $O/$m.out; : > $O/$m.err
  ./capstone-vexec ./controls.dom $m >> $O/$m.out 2>> $O/$m.err
  echo "CTL_RC:$m:$?"
  emit $m.out $O/$m.out
  emit $m.err $O/$m.err
done
rmmod capstone_vm; echo "CTL_CLEANUP:$?"
echo VIRTUAL_STAGED_DONE
'''.replace("MODES", " ".join(MODES))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def symbolize(llvm_bin, image, symbols, pc, code):
    """Function and file:line of a runtime pc, from the image's symbols and DWARF."""
    fn, off = symbols.lookup(pc, code)
    va = pc - code + symbols.base
    out = subprocess.run([str(llvm_bin / "llvm-symbolizer"), f"--obj={image}", "--relativenames", hex(va)],
                         capture_output=True, text=True).stdout.split("\n")
    line = out[1].strip() if len(out) > 1 else ""
    return {"function": fn, "offset": off, "image_va": hex(va), "symbolizer": [l for l in out if l.strip()],
            "file_line": line}


def reading(mode, rc, out, err, llvm_bin, image, symbols):
    lines = out.splitlines()
    rec = {"exit": rc,
           "probes": [{"name": m.group(1), "at": m.group(2)}
                      for m in (re.match(r"PROBE (\S+) (\S+)$", l) for l in lines) if m],
           "addresses": [l for l in lines if l.startswith(("ADDR ", "REUSE "))],
           "lines": [l for l in lines if not l.startswith("PROBE ")]}
    # The driver's last line counts the lines it printed before it: a lossless transcript has
    # exactly that many. A run that faulted has no last line; its transcript is judged by its
    # probes instead.
    last = re.search(rf"^CTL {mode} exit (\d+) lines=(\d+) write_errors=(\d+)$", out, re.M)
    if last:
        before = out[:last.start()].splitlines()
        rec["transcript"] = {"printed": int(last.group(2)), "received": len(before),
                             "write_errors": int(last.group(3)),
                             "complete": len(before) == int(last.group(2)) and last.group(3) == "0"}
    fault = v.FAULT_LINE.search(err) or v.FAULT_LINE.search(out)
    if fault:
        cause, pc, code = int(fault.group(1)), int(fault.group(2), 16), int(fault.group(4), 16)
        line = next(l for l in (err + "\n" + out).splitlines() if "domain fault cause=" in l)
        rec.update(outcome="fault", cause=cause, pc=hex(pc), fault_line=line.strip(),
                   at=symbolize(llvm_bin, image, symbols, pc, code),
                   last_probe=rec["probes"][-1] if rec["probes"] else None)
    elif rc == 0 and f"CTL {mode} completed" in out:
        rec["outcome"] = "completed"
    else:
        rec["outcome"] = f"exit{rc}"
        rec["stderr"] = err.splitlines()[-5:]
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("release", "protected"), required=True)
    p.add_argument("--rep", type=int, required=True)
    for name in ("work", "image", "qemu", "images", "adapter", "run-staged", "llvm-bin"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    sdk_builtins.gate([a.image], who="mruby:run.py")
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / "stage"
    stage.mkdir()
    staged = {"capstone-vexec": a.adapter / "capstone-vexec", "capstone_vm.ko": a.adapter / "module/capstone_vm.ko",
              "controls.dom": a.image}
    for name, src in staged.items():
        shutil.copy2(src, stage / name)
    (stage / "gate.sh").write_text(GATE)
    t0 = time.time()
    run = subprocess.run([sys.executable, str(a.run_staged), "--qemu", str(a.qemu.resolve()),
                          "--images", str(a.images.resolve()), "--stage", str(stage),
                          "--work", str(a.work / "guest"), "--timeout", str(a.timeout)])
    wall = time.time() - t0
    log = (a.work / "guest/serial.log").read_text(errors="replace").replace("\r", "")
    files = {name: gzip.decompress(base64.b64decode(data)).decode(errors="replace")
             for name, data in re.findall(r"^CTL_FILE:(\S+)\n(.*?)^CTL_FILE_END$", log, re.M | re.S)}
    rcs = dict(re.findall(r"^CTL_RC:(\w+):(\d+)$", log, re.M))
    symbols = v.Symbols(a.llvm_bin, a.image)
    modes = {}
    for m in MODES:
        if m not in rcs:
            modes[m] = {"outcome": "no-reading"}
            continue
        modes[m] = reading(m, int(rcs[m]), files.get(f"{m}.out", ""), files.get(f"{m}.err", ""),
                           a.llvm_bin, a.image, symbols)
    rec = {"arm": a.arm, "rep": a.rep, "boot_completed": run.returncode == 0, "boot_wall_seconds": round(wall, 1),
           "modes": modes, "gate_sha256": hashlib.sha256(GATE.encode()).hexdigest(),
           "staged_sha256": {n: sha(stage / n) for n in sorted(staged)},
           "platform_sha256": {n: sha(a.images / n) for n in ("Image", "fw_jump.elf", "rootfs.ext2")}
           | {"qemu-system-riscv64": sha(a.qemu.resolve())},
           "runner_sha256": sha(HERE), "run_staged_sha256": sha(a.run_staged)}
    (a.work / "rep.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"{a.arm} rep{a.rep}: boot={'ok' if rec['boot_completed'] else 'FAILED'} {wall:.0f}s "
          + " ".join(f"{m}={modes[m]['outcome']}" + (f"/{modes[m]['cause']}" if "cause" in modes[m] else "")
                     for m in MODES))
    ok = rec["boot_completed"] and all(modes[m]["outcome"] in ("completed", "fault") for m in MODES)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
