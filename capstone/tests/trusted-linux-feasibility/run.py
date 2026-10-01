#!/usr/bin/env python3
"""Run an ordinary Linux control on the M1 QEMU binary.

This gate establishes guest and harness readiness. It cannot claim a protected
Linux process until the kernel/user capability ABI exists.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
MARKER = re.compile(rb"^CAPSTONE_FEASIBILITY_BASELINE_OK same_address=([01])$")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def command_result(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def run_guest(qemu, images, disk, log_path, timeout, kernel_module=False):
    command = [str(qemu), "-M", "virt-capstone", "-m", "2G", "-smp", "1",
               "-nographic", "-monitor", "none", "-serial", "stdio",
               "-bios", str(images / "fw_jump.elf"),
               "-kernel", str(images / "Image"), "-append", "root=/dev/vda ro",
               "-snapshot", "-drive",
               f"file={images / 'rootfs.ext2'},format=raw,id=hd0",
               "-device", "virtio-blk-device,drive=hd0",
               "-drive", f"file={disk},format=raw,id=probe,readonly=on",
               "-device", "virtio-blk-device,drive=probe",
               "-cpu", "rv64,sstc=false,h=false,x-capstone-u-mode=true"]
    lock_path = os.environ.get("CAPSTONE_QEMU_LOCK")
    lock = None
    if lock_path and os.environ.get("CAPSTONE_QEMU_LOCK_HELD") != "1":
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        lock = open(lock_path, "a+b")
        fcntl.flock(lock, fcntl.LOCK_EX)

    guest = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=0)
    logged_in = False
    command_sent = False
    complete = False
    output = b""
    deadline = time.monotonic() + timeout
    try:
        with log_path.open("wb") as log:
            while time.monotonic() < deadline:
                ready, _, _ = select.select([guest.stdout], [], [], 1)
                if ready:
                    chunk = os.read(guest.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    output = (output + chunk)[-65536:]
                    if b"domain halted" in output:
                        break
                    if not logged_in and b"login:" in output:
                        guest.stdin.write(b"root\n")
                        guest.stdin.flush()
                        logged_in = True
                        output = b""
                    if logged_in and not command_sent and b"# " in output:
                        script = (b"mkdir -p /mnt/probe; "
                                  b"mount -t ext4 -o ro /dev/vdb /mnt/probe && ")
                        if kernel_module:
                            script += (b"insmod /mnt/probe/capstone_s_context.ko; "
                                       b"module_rc=$?; "
                                       b"printf 'CAPSTONE_MODULE_EXIT:%d\\n' "
                                       b"\"$module_rc\"; ")
                        script += (b"/mnt/probe/probe; rc=$?; "
                                   b"printf 'CAPSTONE_PROBE_EXIT:%d\\n' "
                                   b"\"$rc\"\n")
                        guest.stdin.write(script)
                        guest.stdin.flush()
                        command_sent = True
                        output = b""
                    if command_sent:
                        exit_lines = [line.strip() for line in output.splitlines()
                                      if re.fullmatch(rb"CAPSTONE_PROBE_EXIT:[0-9]+",
                                                      line.strip())]
                        if exit_lines:
                            complete = exit_lines[-1] == b"CAPSTONE_PROBE_EXIT:0"
                            break
                if guest.poll() is not None:
                    break
    finally:
        guest.terminate()
        try:
            guest.wait(timeout=5)
        except subprocess.TimeoutExpired:
            guest.kill()
            guest.wait()
        if lock:
            lock.close()

    markers = [MARKER.fullmatch(line.strip()) for line in output.splitlines()]
    valid = [match for match in markers if match]
    module_ok = not kernel_module or (
        sum(line.strip() == b"CAPSTONE_MODULE_EXIT:0"
            for line in output.splitlines()) == 1 and
        sum(line.strip().endswith(b"CAPSTONE_S_CONTEXT_PRESELECT_OK")
            for line in output.splitlines()) == 1)
    if (not complete or len(valid) != 1 or not module_ok or
            b"CAPSTONE_FEASIBILITY_FAIL:" in output):
        raise RuntimeError(f"guest did not complete the ordinary Linux control; see {log_path}")
    return int(valid[0].group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--kernel-module", type=Path,
                        help="run the S-mode context candidate in the actual Linux kernel")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--control-missing-probe", action="store_true",
                        help="omit the guest binary; the run must fail")
    parser.add_argument("--control-missing-module", action="store_true",
                        help="omit the requested kernel module; the run must fail")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.control_missing_module and not args.kernel_module:
        parser.error("--control-missing-module requires --kernel-module")
    inputs = {name: args.image_dir / name for name in
              ("fw_jump.elf", "Image", "rootfs.ext2")}
    inputs.update(qemu=args.qemu, compiler=args.cc, source=HERE / "probe.c")
    if args.kernel_module:
        inputs["kernel_module"] = args.kernel_module
    for name, path in inputs.items():
        if not path.is_file():
            parser.error(f"missing {name}: {path}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trusted-linux-feasibility.",
                                     dir=os.environ.get("CAPSTONE_TMP_ROOT", "/tmp")) as scratch:
        scratch = Path(scratch)
        staging = scratch / "staging"
        staging.mkdir()
        probe = staging / "probe"
        command_result([str(args.cc), "-O2", "-static", "-Wall", "-Wextra",
                        "-o", str(probe), str(HERE / "probe.c")])
        probe_hash = digest(probe)
        if args.control_missing_probe:
            probe.unlink()
        if args.kernel_module and not args.control_missing_module:
            shutil.copyfile(args.kernel_module,
                            staging / "capstone_s_context.ko")
        disk = scratch / "probe.ext4"
        with disk.open("wb") as stream:
            stream.truncate(32 * 1024 * 1024)
        command_result(["/sbin/mkfs.ext4", "-F", "-q", "-d", str(staging), str(disk)])
        same_address = run_guest(args.qemu, args.image_dir, disk, args.log,
                                 args.timeout, bool(args.kernel_module))
        if args.record:
            record = {"schema": 1, "status": "PASS",
                      "scope": "linux_s_context_preselect" if args.kernel_module else "ordinary_linux_only",
                      "protected_process": False,
                      "same_address_observed": bool(same_address),
                      "sha256": {name: digest(path) for name, path in inputs.items()},
                      "probe_binary_sha256": probe_hash}
            if args.kernel_module:
                record["kernel_context_module"] = "PASS: S-mode STC/LDC executed before protected U selection"
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("PASS ordinary Linux mmap/mprotect, malloc/free, fork, read, munmap"
          f" (same_address={same_address}; protection not enabled)")
    if args.kernel_module:
        print("PASS Linux S-mode tagged context instructions before U selection")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        sys.exit(1)
