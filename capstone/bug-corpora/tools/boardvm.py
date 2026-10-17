"""Running corpus steps on the vp2 board in the virtual profile, all in one boot: the Tier 3 board backend.

BoardBatch is virtualvm.Batch's board twin. It has the same put()/step() API and the same @@STEP-BEGIN/@@STEP-END
markers, so a corpus runner's observe() reads a board step exactly as it reads a QEMU one. The comparison rule and its
seven amendments are in docs/plans/2026-10-17-vp2-corpus-on-silicon.md (Tier 3).

execute():
- writes the guest script for /test-domains/<BATCH>/;
- hands the stage to the board kit's recipe, which bakes it into a private board image (the Tier 1 recipe: a manifest
  of explicit names, the shared buildroot restored in an EXIT trap, the initramfs checked by hash), builds the
  firmware, runs the preflight, and boots through the baked-rung driver with the watchdog;
- reads the batch rung's own segment from the driver's per-rung transcript, never the raw capture (the console replays
  the previous boot on connect; amendment 1).

A board kit is a directory holding:
- board.env: shell assignments the recipe sources (the bitstream, the monitor and wrapper worktrees and their
  commits, the firmware defines, the payload cap);
- adapter/capstone-vexec and adapter/module/capstone_vm.ko;
- board-run.sh, the recipe: board-run.sh <stage dir> <work dir> <timeout s>.
The recipe writes <work>/board.txt (the driver's per-rung transcript) and <work>/SEQ, and exits 0 when the board ran.

BOARDVM_STUB_TRANSCRIPT=<file> skips the recipe and takes <file> as board.txt: the dry run.

Nothing here judges.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import virtualvm
from verdicts import sha256

BATCH = "vbatch"
RUNG = "vbatch"
FILES = ("board.env", "board-run.sh", "adapter/capstone-vexec", "adapter/module/capstone_vm.ko")
HEARTBEAT_S = 30


def platform(kit, compiler, runner):
    """The board arm's identity. The `qemu` key is written board:<bitstream>, so PLATFORM_KEYS holds and the arm cannot
    be read as an emulator run (amendment 7). The firmware and kernel hashes are the ones the recipe built, read from
    its record after the run (<work>/firmware.txt), so they are filled in by the caller via record_firmware()."""
    kit = Path(kit)
    env = read_env(kit / "board.env")
    record = {"qemu": f"board:{env.get('BOARD_BITSTREAM', 'unrecorded')}",
              "firmware": "unrecorded", "kernel": "unrecorded", "rootfs": "board initramfs (baked; see firmware)",
              "launcher": sha256(kit / FILES[2]), "module": sha256(kit / FILES[3]),
              "monitor": env.get("BOARD_MONITOR_COMMIT", "unrecorded"),
              "monitor_defines": env.get("BOARD_FW_DEFINES", "unrecorded")}
    record["compiler"] = sha256(compiler) if compiler and Path(compiler).is_file() else "unrecorded"
    record["runner"] = sha256(runner)
    record["judge"] = sha256(Path(__file__).with_name("verdicts.py"))
    record["profile"] = "virtual"
    record["machine"] = "board"
    return record


def record_firmware(record, work):
    """Fill the firmware and kernel hashes from the recipe's record (<work>/firmware.txt: `firmware <sha256>` and
    `kernel <sha256>` lines). A missing record stays "unrecorded" -- said, not omitted."""
    f = Path(work) / "firmware.txt"
    if f.is_file():
        for line in f.read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in ("firmware", "kernel") and v:
                record[k] = v.strip()
    return record


def read_env(path):
    """KEY=value lines (no shell expansion: the recipe sources the same file, so keep the values literal)."""
    env = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip('"')
    return env


def rung_segment(transcript, rung=RUNG):
    """The text between the rung's own start and END markers in the driver's per-rung transcript. The markers are
    anchored at line start with a numeric rc, because the driver's echoed command line carries both literally. A rung
    that did not return has no END marker: its segment runs to the next rung's start or to the end, and completed is
    False."""
    t = transcript.replace("\r", "")
    starts = list(re.finditer(rf"^### RUNG \d+/\d+ {re.escape(rung)} ###$", t, re.M))
    if len(starts) != 1:
        return None, False
    rest = t[starts[0].end():]
    end = re.search(rf"^### RUNG \d+ {re.escape(rung)} END rc=-?\d+ ###$", rest, re.M)
    nxt = re.search(r"^### RUNG \d+/\d+ \S+ ###$", rest, re.M)
    if end and (not nxt or end.start() < nxt.start()):
        return rest[:end.start()], True
    return (rest[:nxt.start()] if nxt else rest), False


class BoardBatch(virtualvm.Batch):
    """put() and step() as virtualvm.Batch; put() returns the guest path under /test-domains/<BATCH>/."""

    def put(self, dest, source=None, text=None):
        super().put(dest, source, text)
        return f"/test-domains/{BATCH}/{dest}"

    def script(self):
        """The guest script, board-shaped (amendments 2 and 4).
        - It runs in /test-domains/<BATCH>.
        - insmod is guarded by /proc/modules, because an earlier rung may have loaded the module and the board kernel
          cannot unload it.
        - A heartbeat that no observe() reads keeps the driver's idle limit from cutting a long silent step.
        - The final rmmod fails on the board, harmlessly, after the last step."""
        lines = ["#!/bin/sh", f"cd /test-domains/{BATCH} || exit 1", "dmesg -n 1",
                 'grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }',
                 "chmod 666 /dev/capstone-vm",
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1",
                 f"( i=0; while [ $i -lt 2000 ]; do sleep {HEARTBEAT_S}; i=$((i+1)); echo \"@@HB $i\"; done ) & hb=$!"]
        for name, command in self.steps:
            lines += [f"echo '{virtualvm.BEGIN}{name}'", f"( {command} ) > /tmp/step.out 2>&1", "rc=$?",
                      "cat /tmp/step.out", f"echo \"{virtualvm.END}{name} $rc\""]
        lines += ["kill $hb 2>/dev/null", "rmmod capstone_vm 2>/dev/null", "echo VIRTUAL_STAGED_DONE", ""]
        return "\n".join(lines)

    def controller(self):
        """The baked-rung controller (BAKED_CTL): the batch rung runs the guest script and prints the RESULT line the
        driver waits for; every other rung goes to the image's own controller (the board controls)."""
        return "\n".join([
            "#!/bin/sh",
            f"if [ \"$1\" = {RUNG} ]; then",
            f"  sh /test-domains/{BATCH}/gate.sh 2>&1; rc=$?",
            f"  echo \"RESULT {RUNG} retval=$rc\"",
            "else",
            "  exec /test-domains/b0run.sh \"$@\"",
            "fi", ""])

    def execute(self, kit, work, timeout):
        """Boot once, run every step. Returns ({name: (text, result)}, transcript path, completed)."""
        kit = Path(kit)
        missing = [f for f in FILES if not (kit / f).is_file()]
        if missing:
            raise FileNotFoundError(f"board kit {kit} lacks {missing}")
        shutil.copy2(kit / "adapter/capstone-vexec", self.stage / "capstone-vexec")
        (self.stage / "capstone-vexec").chmod(0o755)
        shutil.copy2(kit / "adapter/module/capstone_vm.ko", self.stage / "capstone_vm.ko")
        (self.stage / "gate.sh").write_text(self.script())
        (self.stage / "ctl.sh").write_text(self.controller())
        (self.stage / "ctl.sh").chmod(0o755)
        work = Path(work)
        work.mkdir(parents=True, exist_ok=False)
        transcript = work / "board.txt"
        stub = os.environ.get("BOARDVM_STUB_TRANSCRIPT")
        if stub:
            shutil.copy2(stub, transcript)
            ran = True
        else:
            run = subprocess.run(["bash", str(kit / "board-run.sh"), str(self.stage), str(work), str(timeout)])
            ran = run.returncode == 0
        if not transcript.exists():
            return {}, transcript, False
        seg, ended = rung_segment(transcript.read_text(errors="replace"))
        if seg is None:
            return {}, transcript, False
        steps = virtualvm.split(seg.replace("\r\n", "\n"))
        return steps, transcript, ran and ended and "VIRTUAL_STAGED_DONE" in seg
