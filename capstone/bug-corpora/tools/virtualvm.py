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
adapter/capstone-vexec, adapter/module/capstone_vm.ko, and optionally adapter/module/params (see module_params()).
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
PARAMS = "adapter/module/params"
SYSFS = "/sys/module/capstone_vm/parameters"


def module_params(kit):
    """The kit's module parameters: adapter/module/params, one `name=value` per line (# comments allowed). The guest
    script writes each through sysfs right after insmod and prints its read-back as `@@PARAM name=<value>`; the run is
    refused unless every one reads back as set (check_params). No file: no parameters, and the script is unchanged."""
    f = Path(kit) / PARAMS
    if not f.is_file():
        return []
    params = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"([a-z_][a-z0-9_]*)=([A-Za-z0-9]+)", line)
        if not m:
            raise ValueError(f"{f}: {line!r} is not name=value")
        params.append((m.group(1), m.group(2)))
    return params


def param_lines(params):
    """The guest lines that set the parameters and print their read-back; a parameter the module does not have, or a
    write that fails, ends the batch."""
    return [f"[ -f {SYSFS}/{n} ] && echo {v} > {SYSFS}/{n} && echo \"@@PARAM {n}=$(cat {SYSFS}/{n})\" || "
            f"{{ echo '@@PARAM {n} REFUSED'; echo VIRTUAL_STAGED_DONE; exit 1; }}" for n, v in params]


def check_params(params, text):
    """Raise unless every parameter was read back exactly once, as set: a bool reads Y for 1/y/Y and N for 0/n/N, any
    other parameter its own value. A boot whose module did not run the kit's configuration is not a record of it."""
    bools = {"1": "Y", "y": "Y", "Y": "Y", "0": "N", "n": "N", "N": "N"}
    for n, v in params:
        got = re.findall(rf"^@@PARAM {n}=(\S+)$", text, re.M)
        if len(got) != 1 or got[0] not in (v, bools.get(v, v)):
            raise RuntimeError(f"module parameter {n}={v}: read back {got or 'nothing'}; the boot did not run the "
                               f"kit's configuration (or never reached the batch)")


def record_params(record, kit):
    """Add the kit's module parameters to a platform record, only when it has any (records without stay as before)."""
    params = module_params(kit)
    if params:
        record["module_params"] = " ".join(f"{n}={v}" for n, v in params)
    return record


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
    return record_params(record, kit)


class Batch:
    def __init__(self, stage):
        self.stage = Path(stage)
        if self.stage.exists():
            sys.exit(f"{self.stage} exists; a batch stages into a fresh directory")
        self.stage.mkdir(parents=True)
        self.steps = []
        self.params = []

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
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1"] + param_lines(self.params)
        for name, command in self.steps:
            lines += [f"echo '{BEGIN}{name}'", f"( {command} ) > /tmp/step.out 2>&1", "rc=$?",
                      "cat /tmp/step.out", f"echo \"{END}{name} $rc\""]
        lines += ["rmmod capstone_vm", "echo VIRTUAL_STAGED_DONE", ""]
        return "\n".join(lines)

    def execute(self, kit, work, timeout):
        """Boot once, run every step. Returns ({name: (text, result)}, serial path, completed)."""
        kit = Path(kit)
        shutil.copy2(kit / "adapter/capstone-vexec", self.stage / "capstone-vexec")
        (self.stage / "capstone-vexec").chmod(0o755)
        shutil.copy2(kit / "adapter/module/capstone_vm.ko", self.stage / "capstone_vm.ko")
        self.params = module_params(kit)
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
        check_params(self.params, text)
        return split(text), serial, run.returncode == 0


def split(serial):
    """{step: (text, result)} for every step that BEGAN. A step that began and never ended (the
    boot timed out or the guest died) gets kind 'none', which no observe() reads as a run.
    The END marker is matched wherever it starts on its line: the script echoes it straight after `cat /tmp/step.out`,
    so a step whose output lacks a final newline puts it mid-line ("(unknown):0: @@STEP-END ..."). Anchored at the line
    start, such a step never closed, ran to the end of the transcript, and swallowed every later step."""
    steps = {}
    for m in re.finditer(rf"^{re.escape(BEGIN)}(\S+)\n(.*?)(?:{re.escape(END)}\1 (\d+)$|\Z)",
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
