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
"""
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


def platform(kit, compiler, runner):
    """Every platform file's sha256, plus the compiler, the runner and the judge."""
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
                 # vp2's table geometry, so a QEMU prediction is made under the ceiling the
                 # board enforces. vp2 keeps the RTL's 16-bit node index, so a FLAT table holds
                 # 65,534 usable ids and the hardware saturates next-unused at 65,536; the
                 # module's node_max_records defaults to 0 = NO CAP, which is the pre-vp2
                 # behaviour. Left unset, a bundle that would exhaust the table on silicon
                 # allocates freely here and its silicon reading (node pressure 30) looks like a
                 # silicon finding rather than a kit mismatch.
                 # Tolerant but LOUD: every vp2-prep kit's module takes the parameter, but older
                 # kits (virt-stack, a2/r6-r9, dev-proof) predate it and insmod FAILS on an unknown
                 # parameter, which would break any re-run of those. So try with the geometry, and
                 # if that fails fall back WITH A MARKER rather than silently -- an uncapped run
                 # that looks capped is the misread this whole change exists to prevent.
                 # VP2_NODE_CAP= appears in every serial log; a reader can tell which it got.
                 # NAMED FOR WHAT IT MEASURES. It was VP2_GEOMETRY=flat65536, which was wrong and
                 # actively misleading: the marker records that the module accepted the 65,536
                 # RECORD CAP, not that the table FORMAT is flat. The format is chosen by the
                 # -DCAPSTONE_VM_FLAT_TABLE build define, not by this parameter, so a PAGED kit
                 # printed "flat65536" too -- observed on bundle 5's paged arm, whose node_capacity
                 # read 1024 while the marker claimed flat.
                 'if insmod capstone_vm.ko node_max_records=65536 2>/dev/null; then'
                 ' echo VP2_NODE_CAP=65536;'
                 ' elif insmod capstone_vm.ko; then'
                 ' echo VP2_NODE_CAP=UNCAPPED-module-predates-node_max_records;'
                 ' else echo VIRTUAL_STAGED_DONE; exit 1; fi',
                 "chmod 666 /dev/capstone-vm",
                 # The launcher prints its `capstone-exec: domain fault cause= pc= ... code=` line only
                 # with diagnostics on (as run-ports.py runs it); without it a fault is a bare
                 # SIGSEGV, which no observe() may read as the mechanism reporting.
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1"]
        # The node-table readout, per step, against vp2's 65,536-record ceiling. No module change
        # is needed: capstone-vexec --stats already prints nodes_high_water, nodes_live,
        # nodes_retired, node_capacity, collections and nodes_reclaimed as JSON through the
        # CV_GLOBAL ioctl (runtime/virtual/exec.c), which is what the ports' memory-behaviour runs
        # read. The corpus runner simply never called it, which is why every Tier 3 serial log was
        # blank for nodes_high_water. A baseline before the first step makes each step's delta
        # readable, and `|| true` keeps a missing or failing launcher from ending the batch -- the
        # absence then shows as no VP2_NODES line rather than a dead run.
        lines += ['echo "VP2_NODES:baseline:$(./capstone-vexec --stats 2>/dev/null || true)"']
        for name, command in self.steps:
            lines += [f"echo '{BEGIN}{name}'", f"( {command} ) > /tmp/step.out 2>&1", "rc=$?",
                      "cat /tmp/step.out", f"echo \"{END}{name} $rc\"",
                      f'echo "VP2_NODES:{name}:$(./capstone-vexec --stats 2>/dev/null || true)"']
        lines += ["rmmod capstone_vm", "echo VIRTUAL_STAGED_DONE", ""]
        return "\n".join(lines)

    def execute(self, kit, work, timeout):
        """Boot once, run every step. Returns ({name: (text, result)}, serial path, completed)."""
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
