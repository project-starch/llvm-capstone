#!/usr/bin/env python3
"""Qualify hush through the shared native-task QEMU runner and exact shell grant."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
KERNEL_ARGS = (
    "capstone_vm.native_task=1 capstone_vm.native_io=1 capstone_vm.dom_binfmt=1 "
    "capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1 "
    "capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1 "
    "capstone_vm.supervisor_slots=1 capstone_vm.native_refused_trace=1 "
    "capstone_vm.dom_null=1 capstone_vm.dom_shell=/tmp/native-shell/hush.dom"
)
CPU_EXTRA = (
    "x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,"
    "x-capstone-test-csrs=false,x-capstone-checked-copy=true,"
    "x-capstone-context-transport=true"
)
GATE = """#!/bin/sh
set -u
trap 'echo HUSH_GATE_EXIT:$?; dmesg; echo VIRTUAL_STAGED_DONE' EXIT
mkdir -p /tmp/native-shell
cp /mnt/vm/hush.dom /tmp/native-shell/hush.dom
chmod +x /tmp/native-shell/hush.dom
cp -a /mnt/vm /tmp/hush-test
cd /tmp/hush-test
chmod +x driver child.dom
[ ! -w /proc/sys/fs/binfmt_misc/status ] || echo 0 > /proc/sys/fs/binfmt_misc/status
echo 0 > /proc/sys/kernel/randomize_va_space
result=0
./driver /tmp/native-shell/hush.dom -c 'echo HUSH_SIMPLE_PASS' || result=1
./driver /tmp/native-shell/hush.dom -c './child.dom args one "two three"' || result=1
./driver /tmp/native-shell/hush.dom semantics.sh ./child.dom || result=1
./driver /tmp/native-shell/hush.dom large-heredoc.sh ./child.dom || result=1
./driver /tmp/native-shell/hush.dom rollback.sh ./child.dom || result=1
cat /sys/module/capstone_vm/parameters/dom_last_base
exit "$result"
"""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("shell", "child", "driver", "qemu", "images", "firmware", "work"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--runner", type=Path,
                   default=REPO / "capstone/runtime/virtual/run-staged.py",
                   help="shared run-staged.py, optionally from a qualified tool snapshot")
    p.add_argument("--timeout", type=int, default=240)
    args = p.parse_args()
    work = args.work.resolve()
    tmp = Path(os.environ["CAPSTONE_TMP_ROOT"]).resolve()
    if tmp not in work.parents or work.exists():
        p.error("work must be a fresh directory below CAPSTONE_TMP_ROOT")
    stage = work / "stage"
    stage.mkdir(parents=True)
    for name, source in (("hush.dom", args.shell), ("child.dom", args.child),
                         ("driver", args.driver), ("semantics.sh", HERE / "semantics.sh")):
        shutil.copy2(source, stage / name)
    (stage / "gate.sh").write_text(GATE)
    (stage / "gate.sh").chmod(0o755)
    (stage / "large-heredoc.sh").write_text(
        'set -e\ncat <<EOF | ./child.dom count > heredoc-count\n' + "A" * 100000 +
        '\nEOF\ntest "$(cat heredoc-count)" = 100001\necho "HUSH_LARGE_HEREDOC_PASS"\n')
    (stage / "rollback.sh").write_text(
        'n=0; while test "$n" -lt 100; do cat <missing-input 2>errors; '
        'n=$((n+1)); done\n'
        'test "$(./child.dom fds)" = "0 1 2 " || exit 1\n'
        'echo HUSH_REDIRECTION_ROLLBACK_PASS\n')
    files = {
        "shell": args.shell, "child": args.child, "driver": args.driver,
        "kernel": args.images / "Image", "rootfs": args.images / "rootfs.ext2",
        "qemu": args.qemu, "firmware": args.firmware,
        "semantics": HERE / "semantics.sh", "testRunner": Path(__file__),
        "sharedRunner": args.runner,
    }
    identities = {key: {"path": str(value.resolve()), "sha256": digest(value)}
                  for key, value in files.items()}
    env = dict(os.environ, CAPSTONE_QEMU_CPU_EXTRA=CPU_EXTRA)
    subprocess.run([
        "python3", str(args.runner),
        "--qemu", str(args.qemu), "--images", str(args.images),
        "--firmware", str(args.firmware), "--stage", str(stage),
        "--work", str(work / "vm"), "--memory", "8G", "--disk-mib", "64",
        "--timeout", str(args.timeout), "--icount", "0", "--kernel-arg", KERNEL_ARGS,
    ], check=True, env=env)
    log_path = work / "vm/serial.log"
    log = log_path.read_text()
    expected = ["HUSH_SIMPLE_PASS", "[one]", "[two three]",
                "HUSH PASS: shell semantics", "HUSH_LARGE_HEREDOC_PASS",
                "HUSH_REDIRECTION_ROLLBACK_PASS", "HUSH_GATE_EXIT:0"]
    passed = all(marker in log for marker in expected) and "HUSH FAIL" not in log
    result = {"status": "PASS" if passed else "FAIL", "identities": identities,
              "kernelArgs": KERNEL_ARGS, "cpuExtra": CPU_EXTRA,
              "serialSha256": digest(log_path), "expected": expected,
              "scope": "native shell service tests; not SPEC qualification"}
    (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], work / "result.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
