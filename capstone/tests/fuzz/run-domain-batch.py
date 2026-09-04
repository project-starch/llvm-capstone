#!/usr/bin/env python3
"""Run a manifest of domains in ONE QEMU boot and record each one's 32-bit result.

    run-domain-batch.py --manifest FILE --share DIR --log FILE --out results.tsv

The manifest has one item per line: `<name>\t<path to .dom>`.  Every item gets a
row `name\tverdict\tvalue\tnote`:

  RET      the loader printed "Called dom (1-th time) retval = N": value = N
  FAULT    QEMU died while the domain ran (a capability fault inside a domain aborts
           the emulator); the guest is rebooted and the run continues with the NEXT
  WEDGE    the shell prompt never came back although QEMU is alive (the domain hung
           the guest); rebooted like FAULT.  Until 2026-09-05 this was an ERROR with
           no reboot, and every later item errored in seconds against the dead guest.
           item, so one bad program does not void the batch
  TIMEOUT  the domain did not return within --per-item-timeout; QEMU is killed and
           rebooted, the run continues
  ERROR    the loader returned without the retval line (create_dom failed, ...);
           the note carries the output tail

A missing row is impossible by construction: every manifest item is written, and a
manifest with no items is an error.  The boot sequence is the same as
tests/runtime-qemu/run-domain-smoke.py, whose helpers this imports; that script is
left as it is because every suite depends on it.
"""
import argparse
import importlib.util
import os
import pathlib
import re
import sys
import time

import pexpect

HERE = pathlib.Path(__file__).resolve().parent
SMOKE = HERE.parent / "runtime-qemu" / "run-domain-smoke.py"
spec = importlib.util.spec_from_file_location("run_domain_smoke", SMOKE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)

RETVAL = re.compile(r"Called dom \(1-th time\) retval = (\d+)")


def qemu_command(qemu_binary, buildroot_dir, share_dir):
    image_dir = buildroot_dir / "build" / "images"
    for required in ("fw_jump.elf", "Image", "rootfs.ext2"):
        if not (image_dir / required).exists():
            raise SystemExit(f"missing buildroot image: {image_dir / required}")
    return [
        str(qemu_binary), "-M", "virt-capstone", "-m", "8G", "-smp", "1", "-nographic",
        "-bios", str(image_dir / "fw_jump.elf"), "-kernel", str(image_dir / "Image"),
        "-append", "root=/dev/vda ro", "-snapshot",
        "-drive", f"file={image_dir / 'rootfs.ext2'},format=raw,id=hd0",
        "-device", "virtio-blk-device,drive=hd0",
        "-virtfs", f"local,path={share_dir},mount_tag=hostshare,security_model=none,id=hostshare",
        "-chardev", "stdio,mux=on,id=ch0,signal=on", "-mon", "chardev=ch0,mode=readline",
        "-serial", "chardev:ch0", "-cpu", "rv64,sstc=false,h=false",
    ]


def boot(qemu_cmd, buildroot_dir, log_writer, mult):
    """Boot, log in, mount the share, load the module.  Raises InfraFlakeError on a boot problem."""
    qemu = pexpect.spawn(qemu_cmd[0], qemu_cmd[1:], timeout=max(5.0, 5.0 * mult), encoding="utf-8",
                         codec_errors="replace", cwd=str(buildroot_dir), env={"QEMU_AUDIO_DRV": "none", **os.environ})
    qemu.logfile_read = log_writer
    try:
        qemu.expect("buildroot login:", timeout=smoke.env_float_or_default("CAPSTONE_QEMU_LOGIN_TIMEOUT", 120 * mult))
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        qemu.terminate(force=True)
        raise smoke.InfraFlakeError("boot-login", "QEMU stopped before the login prompt") from exc
    qemu.sendline("root")
    smoke.expect_prompt(qemu, timeout=30 * mult, action="logging in", infra_phase="guest-login")
    for cmd in ("dmesg -n 1", "stty columns 29999", "mkdir -p /mnt/host",
                "mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host", "insmod /capstone.ko"):
        smoke.run_guest_command(qemu, cmd, timeout=20 * mult, infra_phase="guest-setup")
    return qemu


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--share", required=True, help="host directory exported to the guest (domains are copied in)")
    ap.add_argument("--log", required=True, help="full serial log (never commit it)")
    ap.add_argument("--out", required=True, help="results TSV")
    ap.add_argument("--per-item-timeout", type=float, default=90.0)
    ap.add_argument("--max-reboots", type=int, default=8)
    ap.add_argument("--timeout-multiplier", type=float, default=1.0)
    ap.add_argument("--loader", default="/capstone-test.user")
    a = ap.parse_args()

    tmp_root = pathlib.Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
    repo_root = HERE.parent.parent.parent
    buildroot_dir = pathlib.Path(os.environ.get("CAPSTONE_BUILDROOT_DIR", repo_root / "capstone" / "caplifive-buildroot")).resolve()
    qemu_binary = pathlib.Path(os.environ.get("CAPSTONE_QEMU_BINARY",
                               repo_root / "capstone" / "capstone-qemu" / "build" / "qemu-system-riscv64")).resolve()
    if not qemu_binary.exists():
        raise SystemExit(f"missing qemu binary: {qemu_binary}")

    items = []
    for line in open(a.manifest):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        name, path = line.split("\t", 1)
        if not os.path.exists(path):
            raise SystemExit(f"manifest item {name}: no such file {path}")
        items.append((name, path))
    if not items:
        raise SystemExit("ERROR: the manifest has no items -- nothing to run")

    share = pathlib.Path(a.share).resolve()
    share.mkdir(parents=True, exist_ok=True)
    guest_names = {}
    for name, path in items:
        dst = share / f"{name}.dom"
        with open(path, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())
        guest_names[name] = dst.name

    qemu_cmd = qemu_command(qemu_binary, buildroot_dir, share)
    log_path = pathlib.Path(a.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    reboots = 0
    with log_path.open("w", encoding="utf-8") as log:
        writer = smoke.NormalizedLogWriter(log)
        log.write("> " + " ".join(qemu_cmd) + "\n\n")
        qemu = boot(qemu_cmd, buildroot_dir, writer, a.timeout_multiplier)
        try:
            for name, _ in items:
                cmd = f"{a.loader} /mnt/host/{guest_names[name]}"
                log.write(f"\n> BEGIN {name}\n")
                t0 = time.time()
                try:
                    out = smoke.run_guest_command(qemu, cmd, timeout=a.per_item_timeout)
                    m = RETVAL.search(out)
                    if m:
                        rows.append((name, "RET", m.group(1), f"{time.time() - t0:.1f}s"))
                    else:
                        tail = " | ".join(l.strip() for l in out.strip().splitlines()[-3:])
                        rows.append((name, "ERROR", "", f"no retval line: {tail[:200]}"))
                except RuntimeError as exc:
                    # The loader returned non-zero, or the prompt never came back but QEMU is alive.
                    text = str(exc)
                    m = RETVAL.search(text)
                    if m:
                        rows.append((name, "RET", m.group(1), "loader exit non-zero"))
                    elif qemu.isalive():
                        rows.append((name, "WEDGE", "", text.strip().splitlines()[0][:200]))
                    else:
                        rows.append((name, "FAULT", "", "QEMU exited during the domain"))
                except pexpect.EOF:
                    rows.append((name, "FAULT", "", "QEMU exited during the domain"))
                except pexpect.TIMEOUT:
                    rows.append((name, "TIMEOUT", "", f"> {a.per_item_timeout}s"))
                log.write(f"\n> END {name} {rows[-1][1]} {rows[-1][2]}\n")
                if rows[-1][1] in ("FAULT", "TIMEOUT", "WEDGE") or not qemu.isalive():
                    qemu.terminate(force=True)
                    reboots += 1
                    if reboots > a.max_reboots:
                        for rest, _ in items[len(rows):]:
                            rows.append((rest, "ERROR", "", "reboot budget exhausted"))
                        break
                    log.write(f"\n> REBOOT {reboots}\n")
                    qemu = boot(qemu_cmd, buildroot_dir, writer, a.timeout_multiplier)
            if qemu.isalive():
                smoke.power_off(qemu)
        finally:
            if qemu.isalive():
                qemu.terminate(force=True)

    with open(a.out, "w") as f:
        for r in rows:
            f.write("\t".join(r) + "\n")
    n = {}
    for r in rows:
        n[r[1]] = n.get(r[1], 0) + 1
    print(f"run-domain-batch: {len(rows)} items, " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())) +
          f", reboots {reboots}; results {a.out}")
    assert len(rows) == len(items), "every manifest item must have a row"
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except smoke.InfraFlakeError as exc:
        print(f"{smoke.INFRA_FLAKE_MARKER} phase={exc.phase}", file=sys.stderr)
        print(exc, file=sys.stderr)
        sys.exit(smoke.INFRA_FLAKE_EXIT_CODE)
