"""Running corpus steps on the vp2 board in the virtual profile, all in one boot: the Tier 3 board backend.

BoardBatch is virtualvm.Batch's board twin. It has the same put()/step() API and the same @@STEP-BEGIN/@@STEP-END
markers, so a corpus runner's observe() reads a board step exactly as it reads a QEMU one. The comparison rule and its
seven amendments are in docs/plans/2026-10-17-vp2-corpus-on-silicon.md (Tier 3).

execute():
- writes the guest script for /test-domains/<BATCH>/;
- hands the stage to the board kit's recipe, which bakes it into a private board image (the Tier 1 recipe: a manifest
  of explicit names, the shared buildroot restored in an EXIT trap, the initramfs checked by hash), builds the
  firmware (the dom_stack gate and the payload cap), and boots through the baked-rung driver with the watchdog
  (the driver's own preflight is off, as in every vp2 Tier 1/2 boot);
- reads the batch rung's own segment from the driver's per-rung transcript, never the raw capture (the console replays
  the previous boot on connect; amendment 1).

A board kit is a directory holding:
- board.env: shell assignments the recipe sources (the bitstream, the monitor and wrapper worktrees and their
  commits, the firmware defines, the payload cap);
- adapter/capstone-vexec and adapter/module/capstone_vm.ko, and optionally adapter/module/params (virtualvm's
  module_params(): set through sysfs after insmod, read back, and the run refused unless every one reads as set);
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
STEP_MULT = 20     # a board step's bound: its runner's limit times this, unless board.env sets BOARD_STEP_MULT
RUNG = "vbatch"
FILES = ("board.env", "board-run.sh", "adapter/capstone-vexec", "adapter/module/capstone_vm.ko")
HEARTBEAT_S = 30
TAIL_B = 240       # the running step's last bytes each heartbeat carries, base64 (decoded into a step with no END)
BOOT_MARGIN_S = 300  # the boot timeout minus this is the script's own budget: a step's bound never outlives the boot
CLAMP_MARGIN_S = 60  # a clamped bound ends this long before the budget, so the step's output and END still print
CLAMP_MIN_S = 30     # and never falls below this, so a late step still gets a run


def last_tail(text):
    """The decoded output tail of the LAST heartbeat in a step's text, or None if no heartbeat carried one."""
    import base64
    import re
    tails = re.findall(r"^@@HB \d+ out=\S* tail=([A-Za-z0-9+/=]*)\s*$", text, re.M)
    if not tails or not tails[-1]:
        return None
    try:
        return base64.b64decode(tails[-1]).decode("utf-8", "replace")
    except ValueError:
        return None


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
    return virtualvm.record_params(record, kit)


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


def record_arm(arm, board):
    """The arm a run records: a board run records under vp2-<arm without "virtual-">, which corpus.json maps to the
    same configuration, so the QEMU bundle stays the corpus's current record of the virtual arm."""
    if not board:
        return arm
    return "vp2-" + (arm[len("virtual-"):] if arm.startswith("virtual-") else arm)


def select(virtual_kit, board_kit):
    """For a corpus runner with --virtual-kit | --board-kit: (Batch class, kit). Both share put()/step()/execute()."""
    return (BoardBatch, board_kit) if board_kit else (virtualvm.Batch, virtual_kit)


def run_platform(virtual_kit, board_kit, compiler, runner, work):
    """The platform record for either kit; a board record takes its firmware and kernel from the recipe's record in
    <work> (the guest directory the batch executed in)."""
    if board_kit:
        return record_firmware(platform(board_kit, compiler, runner), work)
    return virtualvm.platform(virtual_kit, compiler, runner)


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


# the module's .text base, printed once after insmod (script())
MODTEXT_LINE = 'echo "@@MODTEXT $(cat /sys/module/capstone_vm/sections/.text 2>/dev/null || echo ABSENT)"'

# What a step looks like at its bound, taken BEFORE the kill (script()): a killed step otherwise reads as one bit. For the
# step's process and every descendant, twice, 5 s apart: /proc stat state, utime and stime (spinning or sleeping, and on
# which side), wchan and the syscall number; the module's resume counters; and, between the two samples, the module's
# own escape trace switched on for 5 s (diag_escapes: kind, cause, pc and tval of each escape, read from dmesg after
# END). Its print budget (16) is per module load, so only a boot's FIRST bounded step gets escape lines. Written to
# /tmp/stall.out, removed before every bounded step, and printed after END as @@STALL, outside every step's text.
STALL_FN = (
    "stall() { S=/sys/module/capstone_vm/parameters; pids=$1; "
    "for r in 1 2 3 4; do for f in /proc/[0-9]*/stat; do read -r l < $f 2>/dev/null || continue; "
    "p=${f#/proc/}; p=${p%/stat}; set -- ${l##*) }; case \" $pids \" in *\" $p \"*) ;; "
    "*) case \" $pids \" in *\" $2 \"*) pids=\"$pids $p\";; esac;; esac; done; done; "
    "for k in 1 2; do for p in $pids; do read -r l < /proc/$p/stat 2>/dev/null || continue; set -- ${l##*) }; "
    "echo \"proc $k $p $(cat /proc/$p/comm 2>/dev/null) state=$1 ppid=$2 utime=${12} stime=${13} "
    "wchan=$(cat /proc/$p/wchan 2>/dev/null) syscall=$(cut -d' ' -f1 /proc/$p/syscall 2>/dev/null)\"; done; "
    "echo \"mod $k sfences=$(cat $S/resume_sfences 2>/dev/null) satp=$(cat $S/resume_satp_refreshes 2>/dev/null)/"
    "$(cat $S/resume_satp_changed 2>/dev/null) busy=$(cat $S/busy_frame_refusals 2>/dev/null)\"; "
    "if [ $k = 1 ]; then [ -w $S/diag_escapes ] && echo 1 > $S/diag_escapes && echo 'escapes on' || echo 'escapes UNSET'; "
    "sleep 5; [ -w $S/diag_escapes ] && echo 0 > $S/diag_escapes; fi; done; }")


class BoardBatch(virtualvm.Batch):
    """put() and step() as virtualvm.Batch; put() returns the guest path under /test-domains/<BATCH>/.
    other_ctl is the controller every rung other than the batch's goes to: the board controls' b0run.sh by default,
    or a kit's own controller (the vp2 Tier 1 vmrun.sh, which runs vm-contract and hands b0-* to b0run.sh)."""

    def __init__(self, stage, other_ctl="/test-domains/b0run.sh"):
        super().__init__(stage)
        self.other_ctl = other_ctl

    def put(self, dest, source=None, text=None):
        super().put(dest, source, text)
        return f"/test-domains/{BATCH}/{dest}"

    def script(self):
        """The guest script, board-shaped (amendments 2 and 4).
        - It runs in /test-domains/<BATCH>.
        - insmod is guarded by /proc/modules, because an earlier rung may have loaded the module and the board kernel
          cannot unload it.
        - The module's .text base is printed once (`@@MODTEXT <addr>`, ABSENT if unreadable): .text is page-aligned, so
          a monitor line carrying a module pc (FRTY, DRT1) decodes exactly as pc minus this base, instead of by its page
          offset alone (S-23's fetch face, 2026-10-20).
        - A heartbeat that no observe() reads keeps the driver's idle limit from cutting a long silent step.
        - The final rmmod fails on the board, harmlessly, after the last step."""
        mult = getattr(self, "step_mult", STEP_MULT)
        lines = ["#!/bin/sh", f"cd /test-domains/{BATCH} || exit 1", "dmesg -n 1",
                 'grep -q "^capstone_vm " /proc/modules || insmod capstone_vm.ko || { echo VIRTUAL_STAGED_DONE; exit 1; }',
                 "chmod 666 /dev/capstone-vm",
                 MODTEXT_LINE, STALL_FN,
                 "export CAPSTONE_EXEC_DIAGNOSTICS=1", "t0=$(date +%s)"] + virtualvm.param_lines(self.params) + [
                 # The heartbeat carries the running step's output size, bytes and lines: a step's output is printed only
                 # when it ends, so without it a step that runs for hours shows nothing of where it is (mruby's smoke,
                 # 2026-10-21, 153 min). Sizes, plus the output's last TAIL_B bytes in base64: base64 so no observe() can
                 # match or count a line of it twice, and the tail so a step the boot ends before its END still shows
                 # where it was (cpython lt1 step 21, 2026-10-22: 458 lines never reached the transcript). execute()
                 # decodes the last one into such a step's text.
                 f"( i=0; while [ $i -lt 2000 ]; do sleep {HEARTBEAT_S}; i=$((i+1)); "
                 "echo \"@@HB $i out=$(wc -c < /tmp/step.out 2>/dev/null)B/$(wc -l < /tmp/step.out 2>/dev/null)L "
                 f"tail=$(tail -c {TAIL_B} /tmp/step.out 2>/dev/null | base64 | tr -d '\\n')\"; "
                 "done ) & hb=$!"]
        for name, command in self.steps:
            bound = self.bounds.get(name)
            if bound:
                # A bounded step: the board image's busybox has no timeout applet, so a watcher kills the step at its
                # bound and says so (@@STEP-BOUND, which split() reads as a timeout). The step runs in the background,
                # so its stdin is /dev/null.
                b = int(bound * mult)
                # The bound never outlives the boot: a step still running when the boot times out never prints its
                # output (cat runs after the kill), so the bound is clamped to what is left of the script's budget
                # (the boot timeout minus BOOT_MARGIN_S, counted from the script's start), never below 30 s.
                budget = getattr(self, "budget", None)
                clamp = ([f"b=$(( {budget} - ($(date +%s) - t0) - {CLAMP_MARGIN_S} ))", f"[ $b -gt {b} ] && b={b}",
                          f"[ $b -lt {CLAMP_MIN_S} ] && b={CLAMP_MIN_S}"] if budget else [f"b={b}"])
                lines += ["rm -f /tmp/stall.out"] + clamp + [f"echo '{virtualvm.BEGIN}{name}'",
                          f"( {command} ) > /tmp/step.out 2>&1 & sp=$!",
                          f"( sleep $b; stall $sp > /tmp/stall.out 2>&1; "
                          f"kill -9 $sp 2>/dev/null && echo \"{virtualvm.BOUND}{name} $b\" ) & wp=$!",
                          "wait $sp; rc=$?", "kill $wp 2>/dev/null",
                          "cat /tmp/step.out", f"echo \"{virtualvm.END}{name} $rc\"",
                          # After a kill at the bound: what the step looked like before the kill (STALL_FN), then the
                          # module's last kernel lines (KERN_INFO, hidden on the board console at printk 1: "collect
                          # skipped ... available 511" is one, and the escape trace another). Printed after END, so
                          # they are outside every step's text.
                          f"[ $rc = 137 ] && {{ sed 's/^/@@STALL {name} /' /tmp/stall.out 2>/dev/null; "
                          f"dmesg | grep -a capstone_vm | tail -n 30 | sed 's/^/@@DMESG {name} /'; }}"]
            else:
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
            f"  exec {self.other_ctl} \"$@\"",
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
        self.params = virtualvm.module_params(kit)
        self.step_mult = float(read_env(kit / "board.env").get("BOARD_STEP_MULT", STEP_MULT))
        self.budget = int(timeout) - BOOT_MARGIN_S if timeout and int(timeout) > BOOT_MARGIN_S + 60 else None
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
        virtualvm.check_params(self.params, seg.replace("\r\n", "\n"))
        steps = virtualvm.split(seg.replace("\r\n", "\n"))
        for name, (text, result) in list(steps.items()):
            if result.get("kind") == "none":
                tail = last_tail(text)
                if tail is not None:
                    steps[name] = (text + "[runner] LAST-HEARTBEAT-TAIL (the step's last output, decoded):\n" + tail
                                   + ("" if tail.endswith("\n") else "\n"), result)
        return steps, transcript, ran and ended and "VIRTUAL_STAGED_DONE" in seg
