#!/usr/bin/env python3
"""Boot CheriBSD purecap under CHERI-QEMU and run a list of commands, one record out.

It is the CheriBSD half of what ports/php/app/run-virtual.py is for the virtual profile, and it
deliberately speaks the same contract -- `--run NAME=ARGV`, one boot for all of them, a record of
{exits, output} per name -- so bug-corpora/php/tools/run-php-corpus.py can judge either backend
without knowing which machine produced the transcript.

THE GUEST SEES THE FILES OVER 9p, AND THE SHARED IMAGE IS NEVER OPENED FOR WRITING. The stage
directory is exported as a virtio-9p share and mounted in the guest. The disk is a qcow2 OVERLAY
whose backing file is the 9.5 GB CheriBSD image, so that image is opened read-only and our writes
land in a few hundred KB of overlay that is thrown away. `-snapshot` is not enough: it still takes
a shared write lock, and on this host a peer lane's guest has held the image for days -- the
second QEMU dies with "Failed to get shared write lock" before the kernel starts. An overlay also
lets two of these run at once.

ONE SCRIPT, NOT N CONSOLE COMMANDS. The runs are written into a shell script inside the stage
directory and the console is told to run that one file. Driving 56 cases by sending 56 lines to a
serial console costs a round trip each and leaves the transcript interleaved with the shell's own
echo; a script writes its own markers and the parse is unambiguous.
"""
import argparse, glob, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

MARK = "CHERI_RUN"
DONE = "CHERI_RUNNER_DONE"
MOUNT_TAG = "phpcorpus"
GUEST_DIR = "/mnt/php"


def guest_script(runs, env=(), run_timeout=60, copy=()):
    """The one script the console runs. `copy` names files to take off 9p before running them.

    A binary is copied to /tmp first: it is executed on every case, and executing straight off a
    9p share makes every run pay the transport. The trigger scripts stay on the share -- they are
    read once each.
    """
    lines = ["#!/bin/sh",
             f"set -u",
             f"cd {GUEST_DIR} || exit 1"]
    for pair in env:
        lines.append(f"export {pair}")
    for name in copy:
        lines += [f"cp {GUEST_DIR}/{name} /tmp/{name}", f"chmod +x /tmp/{name}"]
    for name, argv in runs:
        lines.append(f"echo {MARK}_BEGIN:{name}")
        lines.append(f"timeout -k 5 {int(run_timeout)} {argv} 2>&1")
        lines.append(f"echo {MARK}_EXIT:{name}:$?")
    lines += [f"echo {DONE}", ""]
    return "\n".join(lines)


def disk(a, own_copy):
    """A fresh qcow2 overlay, backed by the shared image or by our own copy of it.

    The shared image is preferred: a copy is 9.5 GB on a host whose disk other lanes share. But
    QEMU locks the BACKING file too, so while a peer lane's guest holds it even an overlay cannot
    open -- and then, and only then, we fall back to a copy. The caller decides, because the lock
    is only discoverable by trying.
    """
    import shutil, subprocess
    a.work.mkdir(parents=True, exist_ok=True)
    backing = a.img
    if own_copy:
        backing = a.work / "cheribsd-purecap.img"
        if not backing.is_file():
            print(f"[cheri-run] copying {a.img} -> {backing} (once per work dir)")
            shutil.copyfile(a.img, backing)
    overlay = a.work / "overlay.qcow2"
    overlay.unlink(missing_ok=True)
    subprocess.run([str(a.sdk / "bin/qemu-img"), "create", "-f", "qcow2",
                    "-b", str(backing), "-F", "raw", str(overlay)],
                   check=True, stdout=subprocess.DEVNULL)
    return overlay


def qemu_firmware(a):
    fw = sorted(glob.glob(str(a.sdk / "share/qemu/bbl-riscv64*cheri*-virt-fw_jump.bin"))) \
        or sorted(glob.glob(str(a.sdk / "share/qemu/bbl-riscv64*fw_jump*.bin")))
    if not fw:
        sys.exit(f"no bbl purecap firmware under {a.sdk}/share/qemu")
    return fw[0]


def qemu_binary(a):
    qemu = a.sdk / "bin/qemu-system-riscv64xcheri"
    return qemu if qemu.is_file() else a.sdk / "bin/qemu-system-riscv64cheri"


def vehicle(a):
    """What this boot ran ON, in the detail another table needs to be compared with ours.

    capstone/tests/cheri-php-blind pins its CheriBSD release, CHERI-LLVM, CHERI-QEMU and boot
    flag in RESULTS.md. Without the same pins here nobody can say whether that measurement and
    this one share a vehicle, which is the first thing a reader would ask -- and they do NOT:
    qemu_argv's own comment records that this QEMU is 7.1.0 and has no sstc property, while that
    recipe passes -cpu rv64,sstc=false to a 7.2.0 fork. A difference that matters is worth
    recording mechanically rather than in a comment, so these are READ from the tools at run
    time; a pin nobody re-reads drifts in silence. The image is identified by size and mtime,
    not by hash: it is ~8 GB and hashing it would cost more than the boot it describes.
    """
    def version(exe):
        try:
            out = subprocess.run([str(exe), "--version"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, timeout=120,
                                 universal_newlines=True).stdout
        except (OSError, subprocess.SubprocessError) as e:
            return f"unreadable: {e}"
        return out.strip().splitlines()[0] if out.strip() else "no output"
    fw, img = Path(qemu_firmware(a)), Path(a.img)
    st = img.stat() if img.is_file() else None
    return {
        "qemu": version(qemu_binary(a)),
        "clang": version(a.sdk / "bin/clang"),
        "firmware": {"file": fw.name, "bytes": fw.stat().st_size if fw.is_file() else None},
        "kernel": str(a.rootfs / "boot/kernel/kernel"),
        "image": {"path": str(img), "bytes": st.st_size if st else None,
                  "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
                           if st else None},
        "cpu_flags": "the machine default: no -cpu override (see qemu_argv)",
    }


def qemu_argv(a, overlay):
    fw = [qemu_firmware(a)]
    qemu = qemu_binary(a)
    kernel = a.rootfs / "boot/kernel/kernel"
    for p in (qemu, Path(fw[0]), kernel, a.img):
        if not Path(p).exists():
            sys.exit(f"missing: {p}")
    # NO -cpu rv64,sstc=false here: this QEMU (7.1.0) has no sstc property and refuses to start
    # with it. The sibling cheri-php-blind recipe passes it for a 7.2.0 fork.
    return str(qemu), [
        "-M", "virt", "-m", str(a.memory), "-nographic",
        "-bios", fw[0], "-kernel", str(kernel),
        "-drive", f"if=none,file={overlay},id=drv,format=qcow2",
        "-device", "virtio-blk-device,drive=drv",
        "-fsdev", f"local,id=fs0,path={a.stage},security_model=none",
        "-device", f"virtio-9p-device,fsdev=fs0,mount_tag={MOUNT_TAG}",
        "-append", "vfs.root.mountfrom=ufs:/dev/vtbd0",
    ]


def parse(text, runs):
    """{name: rc} and {name: [lines]}, from the serial transcript."""
    exits, output, cur = {}, {}, None
    exit_re = re.compile(rf"^{MARK}_EXIT:([^:]+):(\d+)")
    for raw in text.splitlines():
        s = raw.strip("\r\n ")
        if s.startswith(f"{MARK}_BEGIN:"):
            cur = s.split(":", 1)[1]
            output[cur] = []
            continue
        m = exit_re.match(s)
        if m:
            exits[m.group(1)] = int(m.group(2))
            cur = None
            continue
        if cur is not None:
            output[cur].append(s)
    return exits, output


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home()
    p.add_argument("--stage", type=Path, required=True,
                   help="directory exported over 9p: the php binary, the triggers, the controls")
    p.add_argument("--run", action="append", required=True, metavar="NAME=ARGV")
    p.add_argument("--copy", action="append", default=["php"],
                   help="files to copy out of the share into /tmp before running (default: php)")
    p.add_argument("-e", "--env", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--sysctl", action="append", default=[], metavar="NAME=VALUE",
                   help="set in the guest before the runs. The revocation knobs live here: "
                        "security.cheri.runtime_revocation_default and "
                        "security.cheri.runtime_revocation_every_free_default")
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True, help="the serial transcript")
    p.add_argument("--sdk", type=Path, default=home / "cheriBSD/cheri/output/sdk")
    p.add_argument("--rootfs", type=Path, default=home / "cheriBSD/cheri/output/rootfs-riscv64-purecap")
    p.add_argument("--img", type=Path, default=home / "cheriBSD/cheri/output/cheribsd-riscv64-purecap.img",
                   help="the shared image; it is only ever READ, as the backing of our own copy")
    p.add_argument("--work", type=Path, default=Path("/tmp/capstone/php-cheribsd-run"),
                   help="where the per-run overlay (and, only if the shared image is locked, our "
                        "own 9.5 GB copy of it) lives")
    p.add_argument("--own-copy", action="store_true",
                   help="skip the attempt on the shared image and copy it straight away")
    p.add_argument("--memory", default="2048")
    p.add_argument("--timeout", type=int, default=5400, help="the whole boot-to-done budget")
    p.add_argument("--run-timeout", type=int, default=60, help="guest seconds per run")
    a = p.parse_args()
    import pexpect

    runs = []
    for spec in a.run:
        name, _, argv = spec.partition("=")
        if not argv:
            sys.exit(f"--run wants NAME=ARGV, got {spec!r}")
        runs.append((name, argv))
    # The script name carries this run's pid. sh reads a script INCREMENTALLY, so a second run
    # that writes the same path while the first is executing it makes the first continue from the
    # same byte offset into the NEW file -- the guest then runs a mixture of the two. That is not
    # hypothetical: it corrupted the eager boot of a 30-case arm on 2026-10-29, and the transcript
    # shows the seam as garbage ("UN_EXUN_EXUN_EX..."). One file per run, never a shared name.
    script = a.stage / f"run-cases-{os.getpid()}.sh"
    script.write_text(guest_script(runs, a.env, a.run_timeout, a.copy))
    script.chmod(0o755)

    log = open(a.log, "w")
    own_copy = bool(a.own_copy)
    cmd, argv = qemu_argv(a, disk(a, own_copy))
    q = pexpect.spawn(cmd, argv, encoding="utf-8", timeout=a.timeout, codec_errors="replace")
    q.logfile = log
    booted = mounted = completed = False
    applied = []
    try:
        # The lock is only discoverable by trying: QEMU prints this and exits before the kernel
        # starts. One retry, over our own copy of the image.
        if q.expect([r"Failed to get .*lock", r"login:"], timeout=a.timeout) == 0 and not own_copy:
            print("[cheri-run] the shared image is held by another guest; using our own copy")
            q.terminate(force=True)
            own_copy = True
            cmd, argv = qemu_argv(a, disk(a, True))
            q = pexpect.spawn(cmd, argv, encoding="utf-8", timeout=a.timeout,
                              codec_errors="replace")
            q.logfile = log
            q.expect(r"login:", timeout=a.timeout)
        booted = True
        q.sendline("root")
        if q.expect([r"Password:", r"# "], timeout=240) == 0:
            q.sendline("")
            q.expect(r"# ", timeout=120)
        q.sendline("stty -echo 2>/dev/null; true")
        q.expect(r"# ", timeout=60)
        for pair in a.sysctl:
            # Read back what the kernel accepted: a knob that did not take is a run whose
            # configuration is not what its record says, and that is worse than no run. The
            # read-back is MARKED rather than scraped off the last line -- the last line of what
            # pexpect returns is the next shell prompt, and the first version of this recorded
            # that prompt string as the value of every knob.
            name = pair.split("=")[0]
            # The marker is SPLIT in the command and whole in the output, because this console
            # echoes what it is sent: an unsplit marker matches the command line first and the
            # record then holds a fragment of our own shell instead of the kernel's answer.
            q.sendline(f'sysctl {pair} >/dev/null 2>&1; V=$(sysctl -n {name}); '
                       f'echo "SYS""CTL:{name}=$V"')
            q.expect(rf"SYSCTL:{re.escape(name)}=(\S*)", timeout=60)
            applied.append((pair, q.match.group(1)))
            q.expect(r"# ", timeout=60)
        q.sendline(f"kldload virtio_p9fs 2>/dev/null; mkdir -p {GUEST_DIR}; "
                   f"mount -t p9fs -o trans=virtio {MOUNT_TAG} {GUEST_DIR} "
                   f"&& echo MOUNT_OK || echo MOUNT_FAIL")
        q.expect(r"MOUNT_OK|MOUNT_FAIL", timeout=180)
        mounted = q.after.strip() == "MOUNT_OK"
        q.expect(r"# ", timeout=60)
        if mounted:
            q.sendline(f"sh {GUEST_DIR}/{script.name}")
            q.expect(DONE, timeout=a.timeout)
            completed = True
            q.expect(r"# ", timeout=120)
        q.sendline("poweroff 2>/dev/null || halt -p 2>/dev/null || shutdown -p now")
        q.expect(pexpect.EOF, timeout=300)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        print(f"[cheri-run] stopped: {type(exc).__name__}", file=sys.stderr)
        try:
            q.terminate(force=True)
        except Exception:
            pass
    finally:
        log.flush()
        log.close()

    text = Path(a.log).read_text(errors="replace")
    exits, output = parse(text, runs)
    record = {"booted": booted, "mounted": mounted, "completed": completed,
              "panic": "panic:" in text, "exits": exits, "output": output,
              "serial_log": str(a.log), "env": a.env, "run_timeout": a.run_timeout,
              "sysctl": [{"set": k, "read_back": v} for k, v in applied],
              "vehicle": vehicle(a),
              "disk": "our own copy of the image" if own_copy else "an overlay over the shared image"}
    a.record.write_text(json.dumps(record, indent=2) + "\n")
    print(f"booted={booted} mounted={mounted} completed={completed} panic={record['panic']}")
    for name, _ in runs:
        print(f"  {name:28s} exit={exits.get(name, '<none>')}")
    return 0 if completed else 75


if __name__ == "__main__":
    sys.exit(main())
