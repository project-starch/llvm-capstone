#!/usr/bin/env python3

import argparse
import os
import pathlib
import re
import shutil
import sys
from typing import Iterable

import pexpect

SUCCESS_MARKERS = (
    "Ok, good file.",
    "Loadable executable segment found.",
    "Created domain ID = 0",
    "Called dom (1-th time) retval = 0",
)


def parse_args() -> argparse.Namespace:
    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    buildroot_dir = repo_root / "capstone" / "caplifive-buildroot"
    qemu_bin = repo_root / "capstone" / "capstone-qemu" / "build" / "qemu-system-riscv64"

    parser = argparse.ArgumentParser(
        description="Boot Capstone QEMU once, mount a host-shared 9p directory, and run domain smoke tests."
    )
    parser.add_argument("domains", nargs="+", help="Domain ELF files to run inside the guest.")
    parser.add_argument(
        "--share-dir",
        default="/tmp/alexey/capstone-runtime-qemu-share",
        help="Host directory exported into the guest over 9p.",
    )
    parser.add_argument(
        "--log-file",
        default="/tmp/alexey/capstone-runtime-qemu-smoke.log",
        help="Full serial/QEMU log file.",
    )
    parser.add_argument(
        "--buildroot-dir",
        default=str(buildroot_dir),
        help="Path to capstone/caplifive-buildroot.",
    )
    parser.add_argument(
        "--qemu-binary",
        default=str(qemu_bin),
        help="Path to qemu-system-riscv64 built from capstone-qemu.",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help="Multiply the default expect timeouts by this factor.",
    )
    return parser.parse_args()


def last_exit_code(text: str) -> int:
    for line in reversed(text.replace("\r\r", "\r").splitlines()):
        stripped = line.strip()
        if stripped.startswith("__EXIT_CODE__"):
            return int(stripped.removeprefix("__EXIT_CODE__"))
    raise RuntimeError(f"could not parse exit code from: {text!r}")


def copy_domains_into_share(domains: Iterable[str], share_dir: pathlib.Path) -> list[str]:
    share_dir.mkdir(parents=True, exist_ok=True)
    guest_names = []
    for domain in domains:
        src = pathlib.Path(domain).resolve()
        dst = share_dir / src.name
        if src != dst:
            shutil.copy2(src, dst)
        guest_names.append(src.name)
    return guest_names


def run_guest_command(qemu: pexpect.spawn, command: str, timeout: float = 20.0) -> str:
    qemu.sendline(command)
    qemu.expect(r"# ", timeout=timeout)
    output = qemu.before.replace("\r\r", "\r")

    qemu.sendline("printf '__EXIT_CODE__%s\\n' $?")
    qemu.expect(r"# ", timeout=5)
    exit_capture = qemu.before.replace("\r\r", "\r")
    exit_code = last_exit_code(exit_capture)
    if exit_code != 0:
        raise RuntimeError(
            f"guest command failed with exit code {exit_code}: {command}\n{output}\n{exit_capture}"
        )
    return output


def power_off(qemu: pexpect.spawn) -> None:
    try:
        qemu.sendline("poweroff -f")
        qemu.expect(pexpect.EOF, timeout=20)
    except (pexpect.EOF, pexpect.TIMEOUT):
        pass
    finally:
        qemu.terminate(force=True)


def main() -> int:
    args = parse_args()
    share_dir = pathlib.Path(args.share_dir).resolve()
    log_file = pathlib.Path(args.log_file).resolve()
    buildroot_dir = pathlib.Path(args.buildroot_dir).resolve()
    qemu_binary = pathlib.Path(args.qemu_binary).resolve()

    if not qemu_binary.exists():
        raise SystemExit(f"missing qemu binary: {qemu_binary}")

    image_dir = buildroot_dir / "build" / "images"
    fw_jump = image_dir / "fw_jump.elf"
    kernel = image_dir / "Image"
    rootfs = image_dir / "rootfs.ext2"
    for required in (fw_jump, kernel, rootfs):
        if not required.exists():
            raise SystemExit(f"missing buildroot image: {required}")

    guest_domains = copy_domains_into_share(args.domains, share_dir)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timeout_multiplier = args.timeout_multiplier
    qemu_cmd = [
        str(qemu_binary),
        "-M",
        "virt-capstone",
        "-m",
        "8G",
        "-nographic",
        "-bios",
        str(fw_jump),
        "-kernel",
        str(kernel),
        "-append",
        "root=/dev/vda ro",
        "-drive",
        f"file={rootfs},format=raw,id=hd0",
        "-device",
        "virtio-blk-device,drive=hd0",
        "-virtfs",
        f"local,path={share_dir},mount_tag=hostshare,security_model=none,id=hostshare",
        "-chardev",
        "stdio,mux=on,id=ch0,signal=on",
        "-mon",
        "chardev=ch0,mode=readline",
        "-serial",
        "chardev:ch0",
        "-cpu",
        "rv64,sstc=false,h=false",
    ]

    with log_file.open("w", encoding="utf-8") as log:
        log.write("> starting qemu with:\n")
        log.write(" ".join(qemu_cmd) + "\n\n")
        qemu = pexpect.spawn(
            qemu_cmd[0],
            qemu_cmd[1:],
            timeout=max(5.0, 5.0 * timeout_multiplier),
            encoding="utf-8",
            codec_errors="replace",
            cwd=str(buildroot_dir),
            env={"QEMU_AUDIO_DRV": "none", **os.environ},
        )
        qemu.logfile_read = log
        qemu.logfile_send = log

        try:
            qemu.expect("buildroot login:", timeout=120 * timeout_multiplier)
            qemu.sendline("root")
            qemu.expect(r"# ", timeout=30 * timeout_multiplier)

            run_guest_command(qemu, "dmesg -n 1", timeout=10 * timeout_multiplier)
            run_guest_command(qemu, "stty columns 29999", timeout=10 * timeout_multiplier)
            run_guest_command(qemu, "mkdir -p /mnt/host", timeout=10 * timeout_multiplier)
            run_guest_command(
                qemu,
                "mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host",
                timeout=20 * timeout_multiplier,
            )
            run_guest_command(qemu, "insmod /capstone.ko", timeout=20 * timeout_multiplier)

            for domain_name in guest_domains:
                output = run_guest_command(
                    qemu,
                    f"/capstone-test.user /mnt/host/{domain_name}",
                    timeout=30 * timeout_multiplier,
                )
                missing = [marker for marker in SUCCESS_MARKERS if marker not in output]
                if missing:
                    raise RuntimeError(
                        f"domain smoke missing success markers for {domain_name}: {missing}\n{output}"
                    )

            power_off(qemu)
        except Exception:
            qemu.terminate(force=True)
            raise

    print(f"QEMU smoke passed. Full log: {log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



