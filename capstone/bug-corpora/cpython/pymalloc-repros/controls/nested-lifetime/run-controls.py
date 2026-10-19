#!/usr/bin/env python3
"""Run the nested-lifetime controls on the virtual Capstone profile and record what each showed.

    run-controls.py OUT --kit <kit-sublet> --llvm-bin <dir> --native <python3.13.7>
                    --arm protected=<build-arm.sh OUT> --arm release=<build-arm.sh OUT>
                    [--repeats N] [--timeout S]

One boot per arm and repeat, each with its own staged disk, work directory and QEMU lock, through
capstone/runtime/virtual/run-staged.py (one hart, exact bounds). In a boot every control is its own
interpreter process, `cpython.dom -S -c "import _pymctl; _pymctl.<control>()"`, so the stale read
is a separate run from control 3. The platform is checked by hash before anything is staged.

Outcome per control: `completed` when the process printed `CTL <control> completed=1`, exited 0
and no domain fault was reported; `fault` with cause and pc when capstone-vexec reported one (the
pc resolved against the image's own symbols and its probe labels); otherwise `error`.
OUT/results.json holds the result lines only, never the serial capture.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import sdk_builtins  # noqa: E402  (C-81: refuse a builtins build before anything starts)
import verdicts as v  # noqa: E402

PLATFORM = {  # docs/plans/2026-10-11-protected-same-answers.md, as the corpus runs of 2026-10-11
    "qemu-system-riscv64": "af37cc3269d1cfb47a72eb79cc3dfd2397848c613ca703e43625c68ab50ad27d",
    "adapter/capstone-vexec": "3975314a7a6da119c4bff6a0ab0c601fcd96d305f59a0a68fe5b36b9bd544ad4",
    "adapter/module/capstone_vm.ko": "a1c6cb6b79a98cf1f09e55a37adf2320e2b9787922b7e10bf4088c27e396e512",
}
IMAGES = ("Image", "fw_jump.elf", "rootfs.ext2")
CONTROLS = ("parent", "sibling", "successor", "stale")
LABELS = ("pymctl_read", "pymctl_write", "pymctl_stale_read")
NESTED = {"protected": "cpython", "release": "none"}

GATE = r'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko || exit 1
export CAPSTONE_EXEC_DIAGNOSTICS=1 HOME=/tmp TMPDIR=/tmp PYTHONHOME=/mnt/vm/@ARM@
cd /tmp
echo PORT_BEGIN:import
/mnt/vm/capstone-vexec /mnt/vm/@ARM@/cpython.dom -S -c "import sys, _pymctl; print('CTL import', _pymctl.__name__, sys.version.split()[0])"
echo PORT_END:import:$?
for c in @CONTROLS@; do
  echo PORT_BEGIN:$c
  /mnt/vm/capstone-vexec /mnt/vm/@ARM@/cpython.dom -S -c "import _pymctl; _pymctl.$c()"
  echo PORT_END:$c:$?
done
echo PORT_BEGIN:cleanup
cd /mnt/vm
./capstone-vexec --stats
rmmod capstone_vm
echo PORT_END:cleanup:$?
echo VIRTUAL_STAGED_DONE
'''


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(root):
    h = hashlib.sha256()
    for f in sorted(Path(root).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(root)).encode() + b"\0")
            h.update(bytes.fromhex(sha(f)))
    return h.hexdigest()


def probe_lines():
    """file:line of every labelled probe and of the pointer-sized parent read, from the source."""
    src = HERE.parent / "_pymctl.c"
    rel = src.relative_to(REPO)
    out = {}
    for n, line in enumerate(src.read_text().splitlines(), 1):
        for label in LABELS:
            if f"\\n{label}:\\n" in line:
                out[label] = f"{rel}:{n}"
        if "return (uintptr_t)*p;" in line:
            out["pymctl_link_probe"] = f"{rel}:{n}"
    return out


def label_addresses(llvm_bin, image):
    out = subprocess.run([str(llvm_bin / "llvm-nm"), "--defined-only", str(image)],
                         capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in LABELS + ("pymctl_link_probe", "pymctl_read_probe",
                                                     "pymctl_write_probe", "pymctl_stale_probe"):
            found[parts[2]] = int(parts[0], 16)
    return found


def stage(stage_dir, arm, build, kit, native):
    stage_dir.mkdir(parents=True)
    shutil.copy2(kit / "adapter/capstone-vexec", stage_dir / "capstone-vexec")
    shutil.copy2(kit / "adapter/module/capstone_vm.ko", stage_dir / "capstone_vm.ko")
    d = stage_dir / arm
    lib = d / "lib/python3.13"
    d.mkdir()
    shutil.copy2(build / "image/cpython.dom", d / "cpython.dom")
    shutil.copytree(build / "source/src/Python-3.13.7/Lib", lib,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "test", "idlelib", "tkinter"))
    bdir = (build / "source/build/pybuilddir.txt").read_text().strip()
    shutil.copy2(build / "source/build" / bdir / "_sysconfigdata__linux_.py", lib)
    (lib / "lib-dynload").mkdir(exist_ok=True)
    subprocess.run([str(native), "-m", "compileall", "-q", "-f", "-d", f"/mnt/vm/{arm}/lib/python3.13",
                    str(lib)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gate = GATE.replace("@ARM@", arm).replace("@CONTROLS@", " ".join(CONTROLS))
    (stage_dir / "gate.sh").write_text(gate)
    os.chmod(stage_dir / "gate.sh", 0o755)
    return {"gate_sha256": hashlib.sha256(gate.encode()).hexdigest(), "stage_tree_sha256": tree_sha(stage_dir)}


def boot(name, stage_dir, work, kit, timeout, lockdir):
    env = dict(os.environ, CAPSTONE_QEMU_LOCK=str(lockdir / f"qemu-{name}.lock"))
    t0 = time.time()
    r = subprocess.run([sys.executable, str(REPO / "capstone/runtime/virtual/run-staged.py"),
                        "--qemu", str(kit / "qemu-system-riscv64"), "--images", str(kit / "images"),
                        "--stage", str(stage_dir), "--work", str(work), "--timeout", str(timeout),
                        "--exact-bounds"], env=env, capture_output=True, text=True)
    return r.returncode, round(time.time() - t0, 1)


def segments(text):
    out = {}
    for m in re.finditer(r"PORT_BEGIN:(\w+)\r?\n(.*?)PORT_END:\1:(\d+)", text, re.S):
        out[m.group(1)] = (m.group(2), int(m.group(3)))
    return out


def observe(name, body, rc, symbols, labels):
    lines = [l.strip() for l in body.splitlines() if l.strip().startswith("CTL ")]
    fault = v.domain_fault(body, symbols)
    rec = {"rc": rc, "lines": lines}
    if fault:
        m = v.FAULT_LINE.search(body)
        image_pc = fault.pc - int(m.group(4), 16) + symbols.base
        at = [k for k, a in labels.items() if a == image_pc and k in LABELS]
        rec["outcome"] = "fault"
        rec["fault"] = {"cause": fault.cause, "pc": hex(fault.pc), "address": m.group(3),
                        "image_pc": hex(image_pc), "function": fault.symbol,
                        "offset": fault.offset, "at_label": at[0] if at else None,
                        "line": m.group(0)}
    elif rc == 0 and any(l.startswith(f"CTL {name} completed=1") for l in lines):
        rec["outcome"] = "completed"
    else:
        rec["outcome"] = "error"
    addr = {}
    for l in lines:
        for k, val in re.findall(r"\b(A|B|C)=(0x[0-9a-f]+)", l):
            addr.setdefault(k, val)
    rec["addresses"] = addr
    return rec


def judge(arm, name, rec):
    """The pre-registered prediction for this arm and control, and whether it held."""
    lines = " ".join(rec["lines"])
    if name == "parent":
        want, ok = "completes", rec["outcome"] == "completed"
    elif name == "sibling":
        want, ok = "completes, B intact", rec["outcome"] == "completed" and "intact=1" in lines
    elif name == "successor":
        want = "C's address equals A's; access through C completes"
        ok = rec["outcome"] == "completed" and "equal=1" in lines
    elif arm == "release":
        want = "C's address equals A's; the read through A completes and reads C's data"
        ok = rec["outcome"] == "completed" and "equal=1" in lines and "reads_c=1" in lines
    else:
        want = "C's address equals A's; the read through A faults with cause 25 at pymctl_stale_read"
        f = rec.get("fault") or {}
        ok = (rec["outcome"] == "fault" and "equal=1" in lines and f.get("cause") == 25
              and f.get("at_label") == "pymctl_stale_read")
    return {"predicted": want, "held": ok}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--kit", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--arm", action="append", required=True, help="protected=BUILD or release=BUILD")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1200)
    a = p.parse_args()
    sdk_builtins.gate([s.split("=", 1)[-1] for s in a.arm], who="cpython:run-controls.py")
    arms = dict(x.split("=", 1) for x in a.arm)

    platform = {}
    for rel, want in PLATFORM.items():
        got = sha(a.kit / rel)
        platform[rel] = got
        if got != want:
            print(f"CONTROL-FAILED {a.kit / rel} is {got}, the registered platform is {want}", file=sys.stderr)
            return 75
    for img in IMAGES:
        platform[f"images/{img}"] = sha(a.kit / "images" / img)

    a.output.mkdir(parents=True)
    builds, stages = {}, {}
    for arm, build in arms.items():
        build = Path(build)
        manifest = json.loads((build / "image/manifest.json").read_text())
        if manifest.get("profile") != "virtual" or manifest.get("nested") != NESTED[arm]:
            print(f"CONTROL-FAILED {build} is {manifest.get('profile')}/{manifest.get('nested')}, "
                  f"arm {arm} needs virtual/{NESTED[arm]}", file=sys.stderr)
            return 75
        image = build / "image/cpython.dom"
        builds[arm] = {"build": str(build), "image_sha256": sha(image),
                       "manifest": manifest,
                       "applied_patches": (build / "source/build/applied-patches.txt").read_text().split(),
                       "labels": {k: hex(x) for k, x in label_addresses(a.llvm_bin, image).items()}}
        stages[arm] = stage(a.output / f"stage-{arm}", arm, build, a.kit, a.native)
        builds[arm].update(stages[arm])

    lockdir = a.output / "locks"
    lockdir.mkdir()
    jobs = [(arm, r) for r in range(1, a.repeats + 1) for arm in arms]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {(arm, r): pool.submit(boot, f"{arm}-{r}", a.output / f"stage-{arm}",
                                         a.output / f"guest-{arm}-{r}", a.kit, a.timeout, lockdir)
                   for arm, r in jobs}
        done = {k: f.result() for k, f in futures.items()}

    probes = probe_lines()
    runs = []
    for (arm, r), (rc, seconds) in sorted(done.items()):
        image = Path(arms[arm]) / "image/cpython.dom"
        symbols = v.Symbols(a.llvm_bin, image)
        labels = label_addresses(a.llvm_bin, image)
        text = (a.output / f"guest-{arm}-{r}/serial.log").read_text(errors="replace")
        seg = segments(text)
        run = {"arm": arm, "repeat": r, "boot_rc": rc, "boot_seconds": seconds,
               "image_sha256": builds[arm]["image_sha256"],
               "staged_done": "VIRTUAL_STAGED_DONE" in text, "controls": {}}
        if "import" in seg:
            run["import"] = {"rc": seg["import"][1],
                             "lines": [l.strip() for l in seg["import"][0].splitlines() if "CTL" in l]}
        stats = re.search(r"live_domains\D+(\d+).*?live_regions\D+(\d+).*?live_bytes\D+(\d+)",
                          seg.get("cleanup", ("", 0))[0], re.S)
        run["stats_after"] = stats.group(0).replace("\r", "").split("\n") if stats else None
        for name in CONTROLS:
            if name not in seg:
                run["controls"][name] = {"outcome": "not-run"}
                continue
            rec = observe(name, *seg[name], symbols, labels)
            rec["prediction"] = judge(arm, name, rec)
            run["controls"][name] = rec
        runs.append(run)
        for name in CONTROLS:
            rec = run["controls"][name]
            f = rec.get("fault")
            print(f"{arm:<9} r{r} {name:<9} {rec['outcome']:<9} "
                  f"{('cause=%d pc=%s %s+%s label=%s' % (f['cause'], f['pc'], f['function'], f['offset'], f['at_label'])) if f else ''} "
                  f"{rec.get('addresses', '')} held={rec.get('prediction', {}).get('held')}", flush=True)

    record = {"plan": "capstone/docs/plans/2026-10-11-nested-lifetime-controls.md",
              "program": "CPython 3.13.7", "allocator": "pymalloc (PyObject_Malloc/PyObject_Free)",
              "platform": dict(platform, smp=1, exact_bounds=True, runner="capstone/runtime/virtual/run-staged.py"),
              "probes": probes, "builds": builds, "runs": runs,
              "all_predictions_held": all(c.get("prediction", {}).get("held") for run in runs
                                          for c in run["controls"].values())}
    (a.output / "results.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print("ALL-PREDICTIONS-HELD" if record["all_predictions_held"] else "SOME-PREDICTION-DID-NOT-HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
