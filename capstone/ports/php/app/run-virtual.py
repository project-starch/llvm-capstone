#!/usr/bin/env python3
"""Run a virtual-Capstone application image in one guest boot, and report what each run did.

WHY THIS EXISTS RATHER THAN runtime/virtual/run.py. That script is the PLATFORM gate: it boots
a fixed battery (contract.dom, sqlite3.dom, mruby.dom) and scores 46 named checks. It is not
parameterised by an arbitrary image, and it should not be -- a gate whose subject can change is
not a gate. This runner borrows its mechanism verbatim (virt-capstone, -serial stdio, a staged
read-only ext4 disk, console driving, VM_EXIT markers) and points it at our image instead.

AND RATHER THAN capstone_vm. The persistent-VM tool drives the guest over SSH, and its guest
setup runs dropbearkey unconditionally. This project's buildroot has never enabled dropbear --
BR2_PACKAGE_DROPBEAR is not set, in our pinned commit and in dev's -- so `capstone_vm up
--profile virtual` cannot provision a guest here at all. The console route needs nothing that
is not already in the rootfs.

ONE BOOT, MANY RUNS. A boot costs ~30 s and the lock is shared, so every invocation this script
is asked for happens in the same guest, each as its own `capstone-vexec` process. A capability
fault kills that child, not the guest, so later runs still report. Runs expected to RETURN are
still ordered first: if the guest dies for an unrelated reason, the cheap readings survive.

THE ROOTFS IS NEVER WRITTEN. -snapshot discards guest disk writes, so a run cannot corrupt the
2 GB image that every other suite shares.

THIS SCRIPT TAKES THE QEMU LOCK ITSELF, so DO NOT wrap it in capstone_with_qemu_lock -- that
deadlocks. The wrapper's flock and the fcntl.flock below are two processes holding two fds on
the same file, so the inner one blocks until the outer timeout kills it, which on the console
looks exactly like a guest that never booted. runtime/host/capstone_vm/cli.py has the same
property and at least refuses out loud; this one just waits, hence the warning here and the
check below.
"""
import argparse, fcntl, json, os, re, select, shutil, subprocess, sys, tempfile, time
from pathlib import Path

MARK = "VM_EXIT"
# "[   17.332387] ..." -- a kernel console line, not application output.
KERNEL_LINE = re.compile(r"^\[\s*\d+\.\d+\]")


def guest_script(runs, env=()):
    """runs: list of (name, argv-string). One capstone-vexec invocation each.

    env: NAME=VALUE pairs exported before the runs. This is how an arm that is a RUNTIME mode
    reaches the application when the application is real PHP: Phase A's SAPI took the seam mode
    as argv[3], but sapi/cli owns argv, so zend_cap_mode()'s other source -- PHP_CAP_MODE in the
    environment -- is the only route. One guest per arm, so the exports cannot leak across arms.
    """
    lines = ["#!/bin/sh", "cd /mnt/vm || exit 1", "insmod capstone_vm.ko || exit 1",
             "export CAPSTONE_VM_STATS=1 CAPSTONE_EXEC_DIAGNOSTICS=1"]
    for pair in env:
        lines.append(f"export {pair}")
    for name, argv in runs:
        # Leading NAME=VALUE tokens in the argv become a per-RUN environment, the way they
        # would in any shell. Needed because an arm can be a runtime mode (PHP_CAP_MODE) while
        # the image is shared, and without this every arm would need its own 30 s boot.
        assignments = []
        words = argv.split()
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=[^ ]*", words[0]):
            assignments.append(words.pop(0))
        prefix = ("env " + " ".join(assignments) + " ") if assignments else ""
        lines += [f"echo {MARK}_BEGIN:{name}",
                  f"{prefix}./capstone-vexec {' '.join(words)} 2>&1",
                  f"echo {MARK}:{name}:$?"]
    lines += ["echo PHP_VIRTUAL_RUNNER_DONE", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qemu", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="dir with Image, fw_jump.elf, rootfs.ext2")
    p.add_argument("--adapter", type=Path, required=True, help="dir with capstone-vexec and module/capstone_vm.ko")
    p.add_argument("--image", type=Path, action="append", required=True,
                   help="application .dom to stage (repeatable)")
    p.add_argument("--run", action="append", required=True, metavar="NAME=ARGV",
                   help="a run: NAME=<argv passed to capstone-vexec>, repeatable, in order")
    p.add_argument("--extra", type=Path, action="append", default=[],
                   help="extra file to stage into /mnt/vm (e.g. a .php trigger)")
    p.add_argument("-e", "--env", action="append", default=[], metavar="NAME=VALUE",
                   help="exported in the guest before every run (e.g. PHP_CAP_MODE=6)")
    p.add_argument("--memory", default="2G")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--record", type=Path)
    p.add_argument("--serial-log", type=Path)
    a = p.parse_args()

    runs = []
    for spec in a.run:
        if "=" not in spec:
            sys.exit(f"--run wants NAME=ARGV, got {spec!r}")
        name, argv = spec.split("=", 1)
        runs.append((name, argv))

    for n in ("Image", "fw_jump.elf", "rootfs.ext2"):
        if not (a.images / n).is_file():
            sys.exit(f"missing boot image: {a.images / n}")
    launcher = a.adapter / "capstone-vexec"
    module = a.adapter / "module" / "capstone_vm.ko"
    for f in (launcher, module, a.qemu):
        if not f.is_file():
            sys.exit(f"missing: {f}")

    work = Path(tempfile.mkdtemp(prefix="php-virtual-"))
    stage = work / "stage"
    stage.mkdir()
    shutil.copyfile(launcher, stage / "capstone-vexec")
    (stage / "capstone-vexec").chmod(0o755)
    shutil.copyfile(module, stage / "capstone_vm.ko")
    for img in a.image:
        if not img.is_file():
            sys.exit(f"missing application image: {img}")
        shutil.copyfile(img, stage / img.name)
        (stage / img.name).chmod(0o755)
    for extra in a.extra:
        shutil.copyfile(extra, stage / extra.name)
    for pair in a.env:
        if "=" not in pair:
            sys.exit(f"--env wants NAME=VALUE, got {pair!r}")
    (stage / "gate.sh").write_text(guest_script(runs, a.env))

    disk = work / "stage.ext4"
    with disk.open("wb") as f:
        f.truncate(64 << 20)
    subprocess.run(["/sbin/mkfs.ext4", "-F", "-q", "-d", str(stage), str(disk)], check=True)

    cmd = [str(a.qemu), "-M", "virt-capstone", "-m", a.memory, "-smp", "1",
           "-nographic", "-monitor", "none", "-serial", "stdio",
           "-bios", str(a.images / "fw_jump.elf"), "-kernel", str(a.images / "Image"),
           "-append", "root=/dev/vda ro", "-snapshot",
           "-drive", f"file={a.images / 'rootfs.ext2'},format=raw,id=hd0",
           "-device", "virtio-blk-device,drive=hd0",
           "-drive", f"file={disk},format=raw,id=gate,readonly=on",
           "-device", "virtio-blk-device,drive=gate",
           "-cpu", "rv64,sstc=false,h=false,sv48=false,sv57=false,"
                   "x-capstone-u-mode=true,x-capstone-exact-bounds=true"]

    serial = a.serial_log or (work / "serial.log")
    lock_path = os.environ.get("CAPSTONE_QEMU_LOCK") or str(Path.home() / ".capstone-locks/qemu.lock")
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    out = b""
    with open(lock_path, "a+b") as lock, open(serial, "wb") as log:
        # Non-blocking first, so being wrapped in capstone_with_qemu_lock (or racing another
        # suite) reports itself instead of hanging for the whole timeout.
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"{lock_path} is already held -- do NOT wrap this script in "
                     f"capstone_with_qemu_lock, it takes the lock itself; if another suite "
                     f"holds it, wait for that one to finish")
        guest = subprocess.Popen(cmd, env=dict(os.environ, TMPDIR=str(work)),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        login = sent = False
        deadline = time.monotonic() + a.timeout
        try:
            while time.monotonic() < deadline:
                if not select.select([guest.stdout], [], [], 1)[0]:
                    if guest.poll() is not None:
                        break
                    continue
                data = os.read(guest.stdout.fileno(), 65536)
                if not data:
                    break
                log.write(data); log.flush(); out += data
                if not login and b"login:" in out:
                    guest.stdin.write(b"root\n"); guest.stdin.flush(); login = True
                elif login and not sent and b"# " in out:
                    guest.stdin.write(b"mkdir -p /mnt/vm; mount -t ext4 -o ro /dev/vdb /mnt/vm"
                                      b" && sh /mnt/vm/gate.sh\n")
                    guest.stdin.flush(); sent = True
                if b"PHP_VIRTUAL_RUNNER_DONE" in out:
                    break
                # A guest that panics must END the wait, not consume the whole timeout: silence
                # and a panic look identical from here otherwise.
                if b"Kernel panic" in out or b"Oops" in out:
                    break
        finally:
            guest.terminate()
            try:
                guest.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guest.kill(); guest.wait()

    text = out.decode(errors="replace")
    # rstrip, NOT strip: the leading whitespace is DATA. var_dump and print_r indent their
    # output, and app/run-realphp.py establishes that a case's trigger ran by comparing the
    # control arm's output against the corpus's recorded pristine stdout -- a comparison that
    # silently fails for every indented case if the console path eats the indentation. Only the
    # trailing \r of the serial line is noise. The MARK and kernel-line matches below are
    # anchored at the start of a line and are unaffected, because both are emitted at column 0.
    lines = [s.rstrip() for s in text.splitlines()]
    results = {}
    for name, _ in runs:
        hits = [s for s in lines if s.startswith(f"{MARK}:{name}:")]
        results[name] = int(hits[-1].split(":")[2]) if hits else None

    # Per-run output, so a fault's last printed line is attributable to the run that printed it.
    #
    # KERNEL LINES ARE DROPPED. This guest emits "remote fence extension is not available in
    # SBI v1.0" on nearly every TLB shootdown, hundreds of lines per run, which buried the
    # probe's own output and would bury a fault report just as well. The raw stream is still in
    # the serial log; this filter only affects what is attributed to a run and printed. Matching
    # on the timestamp prefix rather than on the message keeps it general.
    chunks, cur = {}, None
    for s in lines:
        if s.startswith(f"{MARK}_BEGIN:"):
            cur = s.split(":", 1)[1]; chunks[cur] = []
        elif s.startswith(f"{MARK}:") and cur:
            cur = None
        elif cur and not KERNEL_LINE.match(s):
            chunks[cur].append(s)

    booted = "login:" in text
    record = {"booted": booted, "reached_script": sent,
              "completed": "PHP_VIRTUAL_RUNNER_DONE" in text,
              "panic": "Kernel panic" in text or "Oops" in text,
              "exits": results, "output": chunks, "serial_log": str(serial)}
    if a.record:
        a.record.write_text(json.dumps(record, indent=2) + "\n")

    print(f"booted={booted} script={sent} completed={record['completed']} panic={record['panic']}")
    for name, _ in runs:
        rc = results[name]
        tag = {0: "completed", 139: "FAULT(SIGSEGV)", 143: "SIGTERM", 126: "ENOEXEC"}.get(rc, rc)
        print(f"  {name:22s} exit={rc if rc is not None else '<none>'}  {tag}")
        for s in chunks.get(name, [])[-14:]:
            print(f"      | {s}")
    print(f"serial log: {serial}")
    if not record["completed"]:
        return 75          # no verdict: the runner did not get through its own script
    return 0


if __name__ == "__main__":
    sys.exit(main())
