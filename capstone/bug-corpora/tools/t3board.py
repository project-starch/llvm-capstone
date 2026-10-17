#!/usr/bin/env python3
"""Tier 3 on the board: bake staged corpus bundles into one boot, and slice the boot's transcript back.

A corpus runner is run twice around a board boot, unchanged (virtualvm.py, "THE BOARD BACKEND"):
with CAPSTONE_VIRTUAL_BOARD=stage:<dir> it saves its batch to <dir>/<bundle>/ and exits; this tool
bakes those bundles into a board image; after the boot it cuts each bundle's part of the transcript
into <dir>/<bundle>/serial.log; the runner is then run with judge:<dir> and judges as on QEMU.

    t3board.py plan  --dir D --order FILE [--payload BYTES]      boots, by compressed stage size
    t3board.py bake  --dir D --bundles a,b,c --kit KITDIR --out OVERLAY [--step-limit S]
    t3board.py slice --dir D --bundles a,b,c --transcript BOARD_OUT

THE BOARD SIDE (bake). OVERLAY/t3/ holds:
  kit/capstone-vexec, kit/capstone_vm.ko   the board kit's launcher and module, once per boot
  <bundle>/                                the bundle's staged files, its own copy, with
  <bundle>/capstone-vexec -> ../kit/...    the launcher where its steps call `./capstone-vexec`
  <bundle>/.t3/steps/NNN.sh                each step's command, verbatim
  <bundle>/.t3/run.sh                      the steps in order, each under the per-step limit
  t3run.sh                                 the controller: rung t3-<bundle> runs that bundle
The controller loads the module once per boot (the board kernel has no module unload), points
/mnt/vm at the bundle (every step runs in /mnt/vm, as on QEMU), prints a T3HB heartbeat every 30 s
(the baked-rung driver ends a rung after its idle limit) and brackets the bundle with
@@T3-BUNDLE-BEGIN / @@T3-BUNDLE-END. A step runs as `setsid sh step.sh` so the limit kills its whole
process tree (busybox here has no `timeout`; tested two-sided on busybox 1.36.1, the board's).

Nothing here judges, and nothing is skipped silently: a bundle missing from the transcript, or
present twice, is an error.
"""
import argparse
import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

KIT_FILES = ("capstone-vexec", "capstone_vm.ko")
NAME = re.compile(r"[A-Za-z0-9_.+-]+")

RUNSTEP = r'''runstep() { # runstep NAME FILE LIMIT_S: the step's whole process tree dies at the limit
  rm -f /tmp/t3wd.to
  setsid sh "$2" > /tmp/step.out 2>&1 &
  pid=$!
  ( s=0; while [ $s -lt $3 ]; do sleep 5; s=$((s+5)); kill -0 $pid 2>/dev/null || exit 0; done
    echo TO > /tmp/t3wd.to; kill -9 -$pid 2>/dev/null; kill -9 $pid 2>/dev/null ) &
  wd=$!
  wait $pid; rc=$?
  kill $wd 2>/dev/null; wait $wd 2>/dev/null
  cat /tmp/step.out
  [ -f /tmp/t3wd.to ] && echo "@@T3-TIMEOUT $1"
  echo "@@STEP-END $1 $rc"
}
'''

CONTROLLER = r'''#!/bin/sh
# Tier 3 controller (the baked-rung driver's BAKED_CTL). Rung t3-<bundle> runs that bundle; every
# other rung goes to the next controller unchanged. Written by t3board.py bake.
R=${T3_ROOT:-}
rung=$1
case $rung in
  t3-*)
    b=${rung#t3-}
    [ -f $R/t3/$b/.t3/run.sh ] || { echo "T3: no bundle $b"; echo "RESULT $rung retval=-1"; exit 1; }
    if [ ! -e /dev/capstone-vm ] && [ -z "$R" ]; then
      insmod $R/t3/kit/capstone_vm.ko || { echo "T3: insmod failed"; echo "RESULT $rung retval=-2"; exit 1; }
    fi
    [ -z "$R" ] && chmod 666 /dev/capstone-vm
    for l in vm host; do
      [ -L $R/mnt/$l ] && rm $R/mnt/$l
      [ -e $R/mnt/$l ] && { echo "T3: $R/mnt/$l exists and is not a T3 link"; echo "RESULT $rung retval=-3"; exit 1; }
    done
    # /mnt/vm is the bundle (virtualvm steps run there); /mnt/host/applications holds an appvm bundle's images at the
    # path the VM's run.py gives them, which the launcher's fault record names
    mkdir -p $R/mnt && ln -s $R/t3/$b $R/mnt/vm
    [ -d $R/t3/$b/host ] && ln -s $R/t3/$b/host $R/mnt/host
    ( i=0; while [ $i -lt 2000 ]; do sleep 30; i=$((i+1)); echo "T3HB $i"; done ) & hb=$!
    echo "@@T3-BUNDLE-BEGIN $b"
    T3_ROOT=$R sh $R/t3/$b/.t3/run.sh
    echo "@@T3-BUNDLE-END $b"
    kill $hb 2>/dev/null; wait $hb 2>/dev/null
    echo "RESULT $rung retval=0" ;;
  *) exec ${T3_NEXT_CTL:-/test-domains/b0run.sh} "$@" ;;
esac
'''


def stage_bytes(stage):
    """The stage's compressed size, as a tar.gz of its files (the planner's estimate; the bake's own
    preflight measures the real image)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            tar.add(stage, arcname=".")
    return buf.tell()


def bundle_home(d, b):
    home = Path(d) / b
    if not NAME.fullmatch(b) or not (home / "steps.json").is_file() or not (home / "stage").is_dir():
        sys.exit(f"ERROR: {home} is not a staged bundle (steps.json + stage/)")
    return home


def plan(a):
    """Order file: one bundle per line, `name [wedge-risk]`, in run order. A boot takes bundles in
    order while they fit the payload; a wedge-risk bundle closes its boot (it runs last)."""
    rows = []
    for line in Path(a.order).read_text().splitlines():
        f = line.split("#", 1)[0].split()
        if f:
            rows.append((f[0], len(f) > 1 and f[1] == "wedge-risk"))
    if not rows:
        sys.exit(f"ERROR: no bundles in {a.order}")
    boots, cur, size = [], [], 0
    for b, risky in rows:
        n = stage_bytes(bundle_home(a.dir, b) / "stage")
        if n > a.payload:
            print(f"NOT STAGEABLE {b}: {n} compressed bytes > payload {a.payload}")
            continue
        if cur and size + n > a.payload:
            boots.append(cur); cur, size = [], 0
        cur.append((b, n, risky)); size += n
        if risky:
            boots.append(cur); cur, size = [], 0
    if cur:
        boots.append(cur)
    for i, boot in enumerate(boots, 1):
        print(f"BOOT {i}: {sum(n for _, n, _ in boot)} bytes: " +
              " ".join(f"{b}({n}{',wedge-risk' if r else ''})" for b, n, r in boot))
    return 0


def bake(a):
    bundles = a.bundles.split(",")
    out = Path(a.out) / "t3"
    if out.exists():
        sys.exit(f"ERROR: {out} exists; a bake writes a fresh overlay")
    kit = Path(a.kit)
    (out / "kit").mkdir(parents=True)
    for f in KIT_FILES:
        shutil.copy2(kit / f, out / "kit" / f)
    (out / "kit/capstone-vexec").chmod(0o755)
    manifest = {"kit": {f: sha(out / "kit" / f) for f in KIT_FILES}, "step_limit_s": a.step_limit,
                "dry_run_root": a.dry_run_root, "bundles": {}}
    for b in bundles:
        home = bundle_home(a.dir, b)
        steps = json.loads((home / "steps.json").read_text())
        dest = out / b
        shutil.copytree(home / "stage", dest, symlinks=False)
        for f in KIT_FILES[:1]:
            if (dest / f).exists():
                sys.exit(f"ERROR: bundle {b} stages its own {f}")
            os.symlink(f"../kit/{f}", dest / f)
        (dest / ".t3/steps").mkdir(parents=True)
        # As virtualvm's gate: in /mnt/vm, diagnostics on (the launcher's fault line); T3_ROOT is empty on
        # the board and a scratch root in the host dry run.
        lines = ["#!/bin/sh", "cd ${T3_ROOT:-}/mnt/vm || exit 1", "dmesg -n 1 2>/dev/null",
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1", RUNSTEP]
        for i, st in enumerate(steps):
            if not NAME.fullmatch(st["name"]):
                sys.exit(f"ERROR: bundle {b} step {st['name']!r} is not shell-safe")
            step = dest / f".t3/steps/{i:03d}.sh"
            cmd = st["command"]
            if a.dry_run_root:   # host dry run only: no /mnt/vm or /mnt/host there; recorded in bake.json
                cmd = re.sub(r"(?<![\w/.])/mnt/(vm|host)\b", lambda m: f"{a.dry_run_root}/mnt/{m.group(1)}", cmd)
            step.write_text(cmd + "\n")
            lines += [f"echo '@@STEP-BEGIN {st['name']}'",
                      f"runstep {st['name']} ${{T3_ROOT:-}}/t3/{b}/.t3/steps/{i:03d}.sh {a.step_limit}"]
        (dest / ".t3/run.sh").write_text("\n".join(lines) + "\n")
        manifest["bundles"][b] = {"steps": len(steps),
                                  "stage_sha256sums": sha(home / "stage.SHA256SUMS"),
                                  "compressed_bytes": stage_bytes(home / "stage")}
    (out / "t3run.sh").write_text(CONTROLLER)
    (out / "t3run.sh").chmod(0o755)
    (out / "bake.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"BAKED {len(bundles)} bundles into {out}: " +
          ", ".join(f"{b} {m['steps']} steps" for b, m in manifest["bundles"].items()))
    return 0


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def slice_(a):
    text = Path(a.transcript).read_text(errors="replace").replace("\r\n", "\n")
    bad = 0
    for b in a.bundles.split(","):
        home = bundle_home(a.dir, b)
        begins = [m.start() for m in re.finditer(rf"^@@T3-BUNDLE-BEGIN {re.escape(b)}$", text, re.M)]
        if len(begins) != 1:
            print(f"ERROR {b}: {len(begins)} @@T3-BUNDLE-BEGIN lines in {a.transcript} (need exactly 1)")
            bad += 1
            continue
        end = re.compile(rf"^@@T3-BUNDLE-END {re.escape(b)}$", re.M).search(text, begins[0])
        nxt = re.compile(r"^@@T3-BUNDLE-BEGIN ", re.M).search(text, begins[0] + 1)
        stop = end.end() if end and (not nxt or end.start() < nxt.start()) else (nxt.start() if nxt else len(text))
        piece = text[begins[0]:stop] + ("\n" if not text[begins[0]:stop].endswith("\n") else "")
        if (home / "serial.log").exists():
            print(f"ERROR {b}: {home}/serial.log exists; slices are written once")
            bad += 1
            continue
        (home / "serial.log").write_text(piece)
        n_end = len(re.findall(r"^@@STEP-END ", piece, re.M))
        print(f"SLICED {b}: {len(piece)} bytes, {n_end} steps ended, "
              f"{'bundle ENDED' if end and stop == end.end() else 'bundle DID NOT END'}")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("plan"); q.add_argument("--dir", required=True); q.add_argument("--order", required=True)
    q.add_argument("--payload", type=int, default=8_000_000)
    q = sub.add_parser("bake"); q.add_argument("--dir", required=True); q.add_argument("--bundles", required=True)
    q.add_argument("--kit", required=True); q.add_argument("--out", required=True)
    q.add_argument("--step-limit", type=int, default=1200)
    q.add_argument("--dry-run-root", default=None, help="host dry runs only: prefix for /mnt/vm and /mnt/host")
    q = sub.add_parser("slice"); q.add_argument("--dir", required=True); q.add_argument("--bundles", required=True)
    q.add_argument("--transcript", required=True)
    a = p.parse_args()
    return {"plan": plan, "bake": bake, "slice": slice_}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
