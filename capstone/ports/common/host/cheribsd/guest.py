"""A disposable CheriBSD QEMU guest; one process per test, loopback-only SSH."""

import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

import pexpect


def _slow(name, default):
    """A timeout that a slower emulator can raise without editing this file.

    Every budget here was a literal tuned to the platform this was written
    against. On a vehicle built from source and run under TCG they are all too
    short in the same way -- `service sshd onestart` generates three host keys
    before it returns, and an RSA-3072 keygen alone can exceed the 45 s a single
    command was given. The failure then reads as "guest SSH did not become
    ready" or a bare pexpect TIMEOUT, i.e. as a broken platform rather than a
    short timer, which is the expensive kind of wrong. Defaults are unchanged.
    """
    return int(os.environ.get(name, str(default)))

class Guest:
    def __init__(
        self, sdk, rootfs, image, output, port, disable_default_revocation=False
    ):
        self.output = output
        self.process = None
        self.log = None
        self.reader = None
        self.reader_stop = threading.Event()
        self.port = port
        self.disable_default_revocation = disable_default_revocation
        key = output / "guest-key"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                "allocator-tests",
                "-f",
                str(key),
            ],
            check=True,
        )
        self.public_key = key.with_suffix(".pub").read_text().strip()
        self.ssh_options = [
            "-i",
            str(key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "-o",
            "LogLevel=ERROR",
        ]
        self.argv = [
            str(sdk / "bin/qemu-system-riscv64cheri"),
            "-M",
            "virt",
            "-m",
            "2048",
            "-smp",
            "1",
            "-nographic",
            "-snapshot",
            "-append",
            "-s",
            "-bios",
            str(sdk / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"),
            "-kernel",
            str(rootfs / "boot/kernel/kernel"),
            "-drive",
            f"file={image},format=raw,if=none,id=disk",
            "-device",
            "virtio-blk-device,drive=disk",
            "-netdev",
            f"user,id=net,hostfwd=tcp:127.0.0.1:{port}-:22",
            "-device",
            "virtio-net-device,netdev=net",
        ]

    def ssh(self, command, timeout=120):
        return subprocess.run(
            ["ssh", *self.ssh_options, "-p", str(self.port), "root@127.0.0.1", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def copy(self, source, destination):
        result = subprocess.run(
            [
                "scp",
                "-O",
                *self.ssh_options,
                "-P",
                str(self.port),
                str(source),
                str(destination),
            ],
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode:
            detail = result.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"guest copy failed ({result.returncode}): {detail}")

    def drain_console(self):
        # SSH runs independently of pexpect. Keep consuming serial/debug output
        # so a full PTY cannot block QEMU and obscure a guest failure.
        while not self.reader_stop.is_set():
            try:
                self.process.read_nonblocking(65536, timeout=0.1)
                self.log.flush()
            except pexpect.TIMEOUT:
                continue
            except pexpect.EOF:
                break

    def start(self):
        self.log = (self.output / "serial.log").open("w")
        self.process = pexpect.spawn(
            self.argv[0],
            self.argv[1:],
            encoding="utf-8",
            codec_errors="replace",
            timeout=_slow("CHERI_GUEST_BOOT_SECONDS", 180),
            env={**os.environ, "TMPDIR": str(self.output)},
        )
        self.process.logfile_read = self.log
        # Kernel messages can interrupt the rest of this prompt on the serial
        # console (for example, "random: unblocking device" before its colon).
        self.process.expect("Enter full pathname of shell")
        self.process.sendline("")
        self.process.expect(r"# ")
        commands = [
            "stty -echo",
            "mount -u -w /",
            "ifconfig vtnet0 inet 10.0.2.15 netmask 255.255.255.0",
            "ifconfig lo0 inet 127.0.0.1",
            "route add default 10.0.2.2",
            "mkdir -p /root/.ssh",
            "chmod 700 /root/.ssh",
            'printf "%s\\n" '
            + shlex.quote(self.public_key)
            + " > /root/.ssh/authorized_keys",
            "chmod 600 /root/.ssh/authorized_keys",
            # A stock CheriBSD world ships sshd_config with PermitRootLogin no, so
            # root is refused by EVERY method and the key above is never consulted:
            # the host sees "Permission denied (publickey,keyboard-interactive)" and
            # the loop below reports only "did not become ready". cheribuild's own
            # disk-image recipe enables it, which is why this never surfaced on an
            # image built that way. Idempotent, and scoped to the disposable guest.
            "sed -i '' -e 's/^#*PermitRootLogin.*/PermitRootLogin yes/' "
            "/etc/ssh/sshd_config || true",
            "grep -q '^PermitRootLogin yes' /etc/ssh/sshd_config || "
            "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
            "service sshd onestart",
        ]
        if self.disable_default_revocation:
            commands.insert(2, "sysctl security.cheri.runtime_revocation_default=0")
        for command in commands:
            self.process.sendline(command)
            self.process.expect(r"# ", timeout=_slow("CHERI_GUEST_CMD_SECONDS", 45))
        self.reader = threading.Thread(target=self.drain_console, daemon=True)
        self.reader.start()
        # The budget was a hardcoded 45 s, which is ample on the platform this was
        # written against and not on a slower one: `service sshd onestart` generates
        # three host keys first, and an RSA-3072 keygen under TCG can eat the whole
        # window on its own. A working guest then reports "SSH did not become ready",
        # which reads like a broken platform rather than a short timer. Overridable,
        # same default.
        deadline = time.monotonic() + _slow("CHERI_SSH_READY_SECONDS", 45)
        while time.monotonic() < deadline:
            result = self.ssh("mkdir -p /tmp/allocator-tests")
            if result.returncode == 0:
                if self.disable_default_revocation:
                    policy = self.ssh(
                        "sysctl -n security.cheri.runtime_revocation_default"
                    )
                    if policy.returncode or policy.stdout.strip() != "0":
                        raise RuntimeError("guest default revocation was not disabled")
                return
            time.sleep(1)
        # Say WHY. This loop discarded every attempt's stderr and then reported
        # only that the guest "did not become ready", which is indistinguishable
        # between a slow keygen, a refused connection and a host-key problem --
        # three failures with three different fixes.
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise RuntimeError("guest SSH did not become ready; last attempt said: "
                           + (detail[-1] if detail else "<no output>"))

    def close(self):
        self.reader_stop.set()
        if self.reader is not None:
            self.reader.join(timeout=1)
        if self.process is not None:
            self.process.terminate(force=True)
        if self.log is not None:
            self.log.close()
