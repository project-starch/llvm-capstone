"""Running corpus steps on the virtual Capstone platform, all in one boot.

The virtual platform (capstone/runtime/virtual) runs each application as an ordinary Linux process
under `capstone-vexec`: a capability fault ends that process and not the guest. So a corpus run is
one boot that executes every control and every case in turn, staged on a read-only disk, through
runtime/virtual/run-staged.py -- the same harness that qualified the platform.

A Batch collects files to stage and named shell steps. execute() returns, per step, the console
text and a result dict shaped like the persistent VM's (`kind`, `value`, `fault`), so a corpus's
observe() reads a virtual step exactly as it reads a physical run. Nothing here judges.

The platform is a kit directory holding exactly what the qualification recorded:
qemu-system-riscv64 (+ libslirp.so.0), images/{Image,fw_jump.elf,rootfs.ext2},
adapter/capstone-vexec, adapter/module/capstone_vm.ko.

THE BOARD BACKEND (Tier 3 on silicon, docs/plans/2026-10-17-vp2-corpus-on-silicon.md) is off unless
CAPSTONE_VIRTUAL_BOARD is set; unset, nothing below changes the QEMU path. A board boot runs several
bundles, so a runner is run twice around it, unchanged:
  stage:<dir>  execute() saves the batch instead of booting and exits 0: <dir>/<bundle>/stage/ (the
               staged files; the launcher and module come from the board kit, once per boot),
               steps.json (each step's name, command and limit) and stage.SHA256SUMS.
  judge:<dir>  execute() returns split() of <dir>/<bundle>/serial.log, the boot's slice for this
               bundle, after refusing unless this run staged exactly the bytes and steps that ran.
CAPSTONE_VIRTUAL_BOARD_BUNDLE names the bundle. In judge mode platform() is the boot's record,
<dir>/board-platform.json. A step the board's per-step limit killed reads kind 'none' with
`[runner] TIMEOUT`, as a QEMU step that never ended does; the board prints `@@T3-TIMEOUT <step>`.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from verdicts import sha256

REPO = Path(__file__).resolve().parents[3]
STAGED = REPO / "capstone/runtime/virtual/run-staged.py"
FILES = ("qemu-system-riscv64", "images/Image", "images/fw_jump.elf", "images/rootfs.ext2",
         "adapter/capstone-vexec", "adapter/module/capstone_vm.ko")
BEGIN, END = "@@STEP-BEGIN ", "@@STEP-END "
BOARD_KIT_FILES = ("capstone-vexec", "capstone_vm.ko")   # the board kit's, never a bundle's
BOARD_TIMEOUT, BOARD_HEARTBEAT = "@@T3-TIMEOUT ", re.compile(r"^T3HB \d+\n", re.M)


def board():
    """(mode, dir, bundle) when the board backend is selected, else None."""
    spec = os.environ.get("CAPSTONE_VIRTUAL_BOARD")
    if not spec:
        return None
    mode, _, where = spec.partition(":")
    bundle = os.environ.get("CAPSTONE_VIRTUAL_BOARD_BUNDLE", "")
    if mode not in ("stage", "judge") or not where or not re.fullmatch(r"[A-Za-z0-9_.+-]+", bundle):
        sys.exit("CAPSTONE_VIRTUAL_BOARD must be stage:<dir> or judge:<dir>, with "
                 "CAPSTONE_VIRTUAL_BOARD_BUNDLE a shell-safe name")
    return mode, Path(where), bundle


def stage_sums(stage):
    """{relative path: sha256} for every staged file except the board kit's own."""
    stage = Path(stage)
    return {str(f.relative_to(stage)): sha256(f) for f in sorted(stage.rglob("*"))
            if f.is_file() and str(f.relative_to(stage)) not in BOARD_KIT_FILES + ("gate.sh",)}


def platform(kit, compiler, runner):
    """Every platform file's sha256, plus the compiler, the runner and the judge."""
    b = board()
    if b and b[0] == "judge":
        record = json.loads((b[1] / "board-platform.json").read_text())
        record["compiler"] = sha256(compiler) if compiler and Path(compiler).is_file() else "unrecorded"
        record["runner"] = sha256(runner)
        record["judge"] = sha256(Path(__file__).with_name("verdicts.py"))
        record["profile"] = "virtual"
        return record
    kit = Path(kit)
    record = {"qemu": sha256(kit / FILES[0]), "kernel": sha256(kit / FILES[1]),
              "firmware": sha256(kit / FILES[2]), "rootfs": sha256(kit / FILES[3]),
              "launcher": sha256(kit / FILES[4]), "module": sha256(kit / FILES[5])}
    record["compiler"] = sha256(compiler) if compiler and Path(compiler).is_file() else "unrecorded"
    record["runner"] = sha256(runner)
    record["judge"] = sha256(Path(__file__).with_name("verdicts.py"))
    record["profile"] = "virtual"
    return record


class Batch:
    def __init__(self, stage):
        self.stage = Path(stage)
        if self.stage.exists():
            sys.exit(f"{self.stage} exists; a batch stages into a fresh directory")
        self.stage.mkdir(parents=True)
        self.steps = []

    def put(self, dest, source=None, text=None):
        """Stage a file, a directory tree, or text at /mnt/vm/<dest>."""
        target = self.stage / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if text is not None:
            target.write_text(text)
        elif Path(source).is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            shutil.copy2(source, target)
        return f"/mnt/vm/{dest}"

    def step(self, name, command):
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name) or name in {n for n, _ in self.steps}:
            raise ValueError(f"step name {name!r} must be unique and shell-safe")
        self.steps.append((name, command))

    def script(self):
        lines = ["#!/bin/sh", "cd /mnt/vm || exit 1", "dmesg -n 1",
                 "insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }",
                 "chmod 666 /dev/capstone-vm",
                 # The launcher prints its `capstone-exec: domain fault cause= pc= ... code=` line only
                 # with diagnostics on (as run-ports.py runs it); without it a fault is a bare
                 # SIGSEGV, which no observe() may read as the mechanism reporting.
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1"]
        for name, command in self.steps:
            lines += [f"echo '{BEGIN}{name}'", f"( {command} ) > /tmp/step.out 2>&1", "rc=$?",
                      "cat /tmp/step.out", f"echo \"{END}{name} $rc\""]
        lines += ["rmmod capstone_vm", "echo VIRTUAL_STAGED_DONE", ""]
        return "\n".join(lines)

    def execute(self, kit, work, timeout):
        """Boot once, run every step. Returns ({name: (text, result)}, serial path, completed)."""
        if board():
            return self.board_execute(*board())
        kit = Path(kit)
        shutil.copy2(kit / "adapter/capstone-vexec", self.stage / "capstone-vexec")
        (self.stage / "capstone-vexec").chmod(0o755)
        shutil.copy2(kit / "adapter/module/capstone_vm.ko", self.stage / "capstone_vm.ko")
        (self.stage / "gate.sh").write_text(self.script())
        size = sum(f.stat().st_size for f in self.stage.rglob("*") if f.is_file())
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(kit) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env.setdefault("CAPSTONE_QEMU_LOCK", str(Path.home() / ".capstone-locks/qemu.lock"))
        Path(env["CAPSTONE_QEMU_LOCK"]).parent.mkdir(parents=True, exist_ok=True)
        run = subprocess.run([sys.executable, str(STAGED), "--qemu", str(kit / FILES[0]),
                              "--images", str(kit / "images"), "--stage", str(self.stage),
                              "--work", str(work), "--timeout", str(timeout),
                              "--disk-mib", str(max(1024, (size >> 20) * 2 + 256))], env=env)
        serial = Path(work) / "serial.log"
        text = serial.read_text(errors="replace").replace("\r\n", "\n") if serial.exists() else ""
        return split(text), serial, run.returncode == 0


    def board_execute(self, mode, where, bundle):
        steps = [{"name": n, "command": c} for n, c in self.steps]
        sums = stage_sums(self.stage)
        home = where / bundle
        if mode == "stage":
            if home.exists():
                sys.exit(f"{home} exists; a bundle is staged once")
            shutil.copytree(self.stage, home / "stage", symlinks=False,
                            ignore=lambda d, names: [n for n in names if Path(d) == self.stage
                                                     and n in BOARD_KIT_FILES + ("gate.sh",)])
            (home / "steps.json").write_text(json.dumps(steps, indent=1) + "\n")
            (home / "stage.SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in sums.items()))
            print(f"BOARD-STAGED {bundle}: {len(steps)} steps, {len(sums)} files -> {home}", flush=True)
            sys.exit(0)
        # judge: refuse unless this run staged what the boot ran
        if json.loads((home / "steps.json").read_text()) != steps:
            sys.exit(f"BOARD JUDGE REFUSED {bundle}: this run's steps are not the staged steps.json")
        ran = dict(l.split("  ", 1)[::-1] for l in (home / "stage.SHA256SUMS").read_text().splitlines())
        if ran != sums:
            diff = sorted(set(ran.items()) ^ set(sums.items()))[:5]
            sys.exit(f"BOARD JUDGE REFUSED {bundle}: the staged bytes differ from the boot's ({diff})")
        serial = home / "serial.log"
        text = BOARD_HEARTBEAT.sub("", serial.read_text(errors="replace").replace("\r\n", "\n"))
        runs = split(text)
        for name, (out, result) in list(runs.items()):
            marker = f"{BOARD_TIMEOUT}{name}\n"
            if marker in out:
                kept = {k: v for k, v in result.items() if k == "fault"}   # as split() keeps it
                runs[name] = (out.replace(marker, "") + "\n[runner] TIMEOUT\n", {"kind": "none", **kept})
        return runs, serial, f"@@T3-BUNDLE-END {bundle}" in text


def split(serial):
    """{step: (text, result)} for every step that BEGAN. A step that began and never ended (the
    boot timed out or the guest died) gets kind 'none', which no observe() reads as a run."""
    steps = {}
    for m in re.finditer(rf"^{re.escape(BEGIN)}(\S+)\n(.*?)(?:^{re.escape(END)}\1 (\d+)$|\Z)",
                         serial, re.S | re.M):
        name, text, rc = m.group(1), m.group(2), m.group(3)
        fault = next((l.strip() for l in text.splitlines() if "domain fault cause=" in l), None)
        if rc is None:
            result = {"kind": "none"}
            text += "\n[runner] TIMEOUT\n"
        else:
            rc = int(rc)
            result = {"kind": "signal", "value": rc - 128} if rc > 128 else {"kind": "exit", "value": rc}
        if fault:
            result["fault"] = fault
        steps[name] = (text, result)
    return steps
