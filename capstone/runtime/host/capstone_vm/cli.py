"""One VM per state directory, with QMP lifecycle and standard SSH transport.

No application protocol or guest execution daemon: SSH preserves terminal, pipe,
EOF and exit status behavior, and the guest's shell starts capstone-exec.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Iterator
import uuid


class VMError(RuntimeError):
    pass


QEMU_ENV = frozenset("""
CAPSTONE_CINC_UNTAGGED_SURVIVE CAPSTONE_GP_FABRICATE CAPSTONE_GP_NONLIN
CAPSTONE_GP_STANDIN CAPSTONE_LCC_UNTAGGED_SURVIVE CAPSTONE_QUIET_GP
CAPSTONE_REV_NODES CAPSTONE_SLOT_CLEARSET_INCLUDES_NONLIN CAPSTONE_SLOT_LOG
CAPSTONE_SLOT_MAX_LINES CAPSTONE_SLOT_TABLE_BITS CAPSTONE_STC_FIRSTSEEN
CAPSTONE_STC_PROBE_OFF CAPSTONE_STC_PROBE_OPS CAPSTONE_STOREWATCH
CAPSTONE_STOREWATCH_HI CAPSTONE_STOREWATCH_LO CAPSTONE_STOREWATCH_LOW
CAPSTONE_TAGWATCH CAPSTONE_TAGWATCH_GRANULE CAPSTONE_TAGWATCH_HI
CAPSTONE_TAGWATCH_LO CAPSTONE_TAGWATCH_MAX CAPSTONE_TAGWATCH_VICTIM
""".split())


def qemu_process_environment(recorded: dict[str, str]) -> dict[str, str]:
    """Use only the session's recorded QEMU settings on every boot."""
    environment = {name: value for name, value in os.environ.items()
                   if name not in QEMU_ENV}
    environment.update(recorded)
    return environment


def qmp(state: Path, command: str) -> dict:
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(5)
        connection.connect(str(state / "qmp.sock"))
        with connection.makefile("rwb", buffering=0) as stream:
            if "QMP" not in json.loads(stream.readline()):
                raise VMError("Invalid QMP greeting")
            for name in ("qmp_capabilities", command):
                stream.write(json.dumps({"execute": name, "id": name}).encode() + b"\n")
                while True:
                    line = stream.readline()
                    if not line:
                        raise VMError("QMP disconnected")
                    response = json.loads(line)
                    if response.get("id") == name:
                        if "error" in response:
                            raise VMError(str(response["error"]))
                        break
            return response["return"]


def running(state: Path) -> bool:
    try:
        # A paused guest still owns its resources and must not be replaced.
        qmp(state, "query-status")
        return True
    except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
        return False


@contextmanager
def lock(state: Path) -> Iterator[None]:
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.stat().st_uid != os.getuid() or state.stat().st_mode & 0o077:
        raise VMError(f"State directory must be private (mode 0700): {state}")
    with (state / "lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


class Console:
    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.pending = b""

    def expect(self, pattern: bytes, timeout: float = 30) -> bytes:
        deadline = time.monotonic() + timeout
        while True:
            match = re.search(pattern, self.pending)
            if match:
                result, self.pending = self.pending[:match.end()], self.pending[match.end():]
                return result
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VMError(f"Guest console timeout; last output: {self.pending[-2000:]!r}")
            self.connection.settimeout(remaining)
            data = self.connection.recv(65536)
            if not data:
                raise VMError("Guest console disconnected")
            self.pending = (self.pending + data)[-1024 * 1024:]

    def send(self, command: str) -> None:
        self.connection.sendall(command.encode() + b"\n")

    def setup(self, script: str) -> None:
        marker = uuid.uuid4().hex
        self.send(f"sh {shlex.quote(script)}; printf '\\n{marker[:16]}''{marker[16:]}:%s\\n' $?")
        output = self.expect(marker.encode() + rb":([0-9]+)\r?\n", 60)
        status = re.search(marker.encode() + rb":([0-9]+)", output)
        if not status or int(status[1]):
            raise VMError(f"Guest setup failed: {output.decode(errors='replace')}")


def ssh_command(state: Path, config: dict, *, terminal: bool = False) -> list[str]:
    return [
        "ssh", "-tt" if terminal else "-T", "-p", str(config["port"]),
        "-i", str(state / "identity"), "-o", "IdentitiesOnly=yes",
        "-o", "PasswordAuthentication=no", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5",
        "-o", "UserKnownHostsFile=" + str(state / "known_hosts"),
        "-o", "GlobalKnownHostsFile=/dev/null", "-l", "root", "127.0.0.1",
    ]


def require_file(path: Path) -> Path:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise VMError(f"Not a regular file: {path}")
    return path


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class InterruptedRun(Exception):
    def __init__(self, signum: int):
        self.signum = signum


def run_application(state: Path, config: dict, words: list[str], *, cwd: str | None = None, environment: list[str] = (), result_path: Path | None = None) -> int:
    """Forward host interruption without putting pipes through a pseudo-terminal.

    The short-lived PID file identifies this SSH command. Verify its random
    environment token before signalling: a stale PID must never kill a new job.
    The guest still runs an ordinary capstone-exec process, with no job daemon.
    """
    if any(not re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", value) or value.startswith("CAPSTONE_JOB=")
           for value in environment):
        raise VMError("Environment must contain NAME=value assignments; CAPSTONE_JOB is reserved")
    job_id = uuid.uuid4().hex
    jobs = state / "assets" / "jobs"
    jobs.mkdir(exist_ok=True)
    job = jobs / job_id
    job.mkdir()
    remote = f"/mnt/control/jobs/{job_id}/pid"
    script = (f"cd {shlex.quote(cwd)} || exit 125; " if cwd is not None else "")
    script += f"echo $$ > {remote}; exec " + shlex.join([
        "capstone-job", f"/mnt/control/jobs/{job_id}/result.json", "--", "capstone-exec", "--", *words])
    command = ssh_command(state, config) + ["exec " + shlex.join(
        ["env", "CAPSTONE_JOB=" + job_id, *environment, "sh", "-c", script])]
    previous = {}

    def completed() -> int:
        record_file = job / "result.json"
        if not record_file.exists():
            raise VMError("Guest command ended without a waitpid result; completion is unknown")
        record = json.loads(record_file.read_text())
        kind, value = record.get("kind"), record.get("value")
        if record.get("version") != 1 or kind not in ("exit", "signal") or type(value) is not int or not 0 <= value <= 255:
            raise VMError("Invalid guest waitpid result")
        if result_path is not None:
            temporary = result_path.with_name(result_path.name + "." + job_id + ".tmp")
            try:
                temporary.write_text(json.dumps(record) + "\n")
                temporary.replace(result_path)
            finally:
                temporary.unlink(missing_ok=True)
        if kind == "signal":
            print(f"capstone-vm: application terminated by signal {value}", file=sys.stderr)
            return 128 + value
        return value

    def interrupt(signum, _frame):
        raise InterruptedRun(signum)

    process = None
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupt)
        # The terminal sends Ctrl-C to us, not the SSH transport. Keep the latter
        # alive long enough to receive the guest's final output and exit status.
        process = subprocess.Popen(command, start_new_session=True)
        try:
            process.wait()
            return completed()
        except InterruptedRun as stopped:
            for signum in previous:
                signal.signal(signum, signal.SIG_IGN)
            deadline = time.monotonic() + 5
            while not (job / "pid").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            if (job / "pid").exists():
                pid = (job / "pid").read_text().strip()
                if not pid.isdecimal() or int(pid) <= 1:
                    raise VMError("Invalid guest job PID")
                owns = (f"test -r /proc/{pid}/environ && "
                        f"tr '\\000' '\\n' < /proc/{pid}/environ 2>/dev/null | "
                        f"grep -qx CAPSTONE_JOB={job_id}")
                cancel = (f"if {owns}; then kill -{stopped.signum} {pid}; fi; "
                          f"n=0; while {owns}; do n=$((n+1)); "
                          "test $n -lt 100 || exit 1; sleep 0.05; done")
                subprocess.run(ssh_command(state, config) + [cancel], stdin=subprocess.DEVNULL,
                               check=True, timeout=10)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    # Guest death was confirmed above. A full consumer pipe may
                    # still block SSH; the finally block closes that transport.
                    pass
            elif process.poll() is None:
                raise VMError("Guest job did not publish its PID; cancellation could not be confirmed")
            return completed()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        shutil.rmtree(job)


def start(args: argparse.Namespace, state: Path) -> int:
    with lock(state):
        files = {key: require_file(getattr(args, key)) for key in
                 ("qemu", "kernel", "firmware", "rootfs")}
        for key in ("launcher", "ssh_server", "module", "job_helper"):
            if getattr(args, key, None):
                files[key] = require_file(getattr(args, key))
        if "launcher" in files and "job_helper" not in files:
            files["job_helper"] = require_file(files["launcher"].with_name("capstone-job"))
        share = args.share.resolve(strict=True)
        if not share.is_dir() or "," in str(share):
            raise VMError("Share must be a directory whose path contains no comma")
        environment = getattr(args, "environment", None)
        if environment is None:
            environment = {name: value for name, value in os.environ.items()
                           if name in QEMU_ENV}
        if any(name not in QEMU_ENV or not isinstance(value, str)
               for name, value in environment.items()):
            raise VMError("Invalid QEMU environment in session configuration")
        environment.setdefault("CAPSTONE_GP_NONLIN", "1")
        environment.setdefault("CAPSTONE_REV_NODES", "65536")
        identity_config = {"files": {name: {"path": str(path), "sha256": fingerprint(path)}
                                     for name, path in files.items()},
                           "share": str(share), "memory": args.memory,
                           "environment": environment}
        if running(state):
            old = json.loads((state / "config.json").read_text())
            if old.get("identity") != identity_config or args.port not in (0, old["port"]):
                raise VMError("Running VM has a different configuration; stop it explicitly before restarting")
            print(f"VM already running: {state}")
            return 0
        # QEMU's comma-separated option syntax has no portable escaping here.
        if any("," in str(path) for path in [state, *files.values()]):
            raise VMError("VM paths must not contain commas")
        if len(str(state / "console.sock").encode()) >= 104:
            raise VMError("State path is too long for Unix sockets")
        assets = state / "assets"
        assets.mkdir(exist_ok=True)
        for key in ("launcher", "ssh_server", "module", "job_helper"):
            if key in files:
                shutil.copyfile(files[key], assets / key)
            else:
                (assets / key).unlink(missing_ok=True)
        identity = state / "identity"
        if not identity.exists():
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                            "capstone-vm", "-f", str(identity)], check=True)
        shutil.copyfile(state / "identity.pub", assets / "authorized_keys")
        # Host keys are made by the guest, then read through our private share.
        (assets / "hostkey.pub").unlink(missing_ok=True)
        setup = """#!/bin/sh
set -eu
dmesg -n 1
mkdir -p /mnt/host /root/.ssh /etc/dropbear
if ! grep -q ' /mnt/host ' /proc/mounts; then
    mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host
fi
if [ -f /mnt/control/launcher ]; then
    cp /mnt/control/launcher /usr/bin/capstone-exec
    chmod 0755 /usr/bin/capstone-exec
fi
if [ -f /mnt/control/ssh_server ]; then
    cp /mnt/control/ssh_server /usr/sbin/dropbear
    chmod 0755 /usr/sbin/dropbear
    ln -sf /usr/sbin/dropbear /usr/sbin/dropbearkey
fi
if [ -f /mnt/control/job_helper ]; then
    cp /mnt/control/job_helper /usr/bin/capstone-job
    chmod 0755 /usr/bin/capstone-job
fi
test -x /usr/bin/capstone-exec
test -x /usr/bin/capstone-job
cp /mnt/control/authorized_keys /root/.ssh/authorized_keys
chmod 0700 /root/.ssh
chmod 0600 /root/.ssh/authorized_keys
if [ -f /mnt/control/module ]; then
    if [ -c /dev/capstone ]; then rmmod capstone; fi
    insmod /mnt/control/module
elif [ ! -c /dev/capstone ]; then
    insmod /capstone.ko
fi
capstone-exec --stats >/dev/null
ifconfig eth0 10.0.2.15 netmask 255.255.255.0 up
killall dropbear 2>/dev/null || true
if [ ! -f /etc/dropbear/dropbear_ed25519_host_key ]; then
    dropbearkey -t ed25519 -f /etc/dropbear/dropbear_ed25519_host_key >/dev/null
fi
dropbearkey -y -f /etc/dropbear/dropbear_ed25519_host_key | sed -n '/^ssh-ed25519 /p' > /mnt/control/hostkey.pub
dropbear -s -g -p 22
"""
        (assets / "setup.sh").write_text(setup)
        # An ephemeral host port is chosen once; QEMU fails safely if a racer takes it.
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", args.port))
            port = reservation.getsockname()[1]
        for name in ("console.sock", "qmp.sock"):
            (state / name).unlink(missing_ok=True)
        command = [
            str(files["qemu"]), "-M", "virt-capstone", "-m", args.memory,
            "-smp", "1", "-display", "none", "-monitor", "none",
            "-bios", str(files["firmware"]), "-kernel", str(files["kernel"]),
            "-append", "root=/dev/vda rw cma=512M", "-snapshot",
            "-drive", f"file={files['rootfs']},format=raw,id=hd0,if=none",
            "-device", "virtio-blk-device,drive=hd0",
            "-virtfs", f"local,path={share},mount_tag=hostshare,security_model=none,id=hostshare",
            "-virtfs", f"local,path={assets},mount_tag=control,security_model=none,id=control",
            "-chardev", f"socket,id=console,path={state / 'console.sock'},server=on,wait=on,logfile={state / 'console.log'}",
            "-serial", "chardev:console", "-qmp", f"unix:{state / 'qmp.sock'},server=on,wait=off",
            "-netdev", f"user,id=net,restrict=on,hostfwd=tcp:127.0.0.1:{port}-:22",
            "-device", "virtio-net-device,netdev=net", "-device", "virtio-rng-device",
            "-cpu", "rv64,sstc=false,h=false",
        ]
        config = {"port": port, "share": str(share), "command": command, "identity": identity_config}
        (state / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        global_lock = Path(os.environ.get("CAPSTONE_QEMU_LOCK", Path.home() / ".capstone-locks/qemu.lock"))
        global_lock.parent.mkdir(parents=True, exist_ok=True)
        with global_lock.open("a") as ownership, (state / "qemu.log").open("wb") as log:
            try:
                fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise VMError(f"Another Capstone VM owns {global_lock}") from error
            # Retain the common test-suite lock in QEMU itself, across CLI exits.
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=log, env=qemu_process_environment(environment),
                                       start_new_session=True,
                                       pass_fds=(ownership.fileno(),))
        try:
            deadline = time.monotonic() + args.boot_timeout
            with socket.socket(socket.AF_UNIX) as connection:
                while True:
                    if process.poll() is not None:
                        raise VMError(f"QEMU exited; see {state / 'qemu.log'}")
                    try:
                        connection.connect(str(state / "console.sock"))
                        break
                    except (FileNotFoundError, ConnectionRefusedError):
                        if time.monotonic() >= deadline:
                            raise VMError("QEMU console startup timed out")
                        time.sleep(0.05)
                console = Console(connection)
                console.expect(rb"buildroot login:", max(1, deadline - time.monotonic()))
                console.send("root")
                console.expect(rb"# ")
                console.send("mkdir -p /mnt/control && mount -t 9p -o trans=virtio,version=9p2000.L control /mnt/control")
                console.expect(rb"# ")
                console.setup("/mnt/control/setup.sh")
            key = (assets / "hostkey.pub").read_text().split()
            if len(key) < 2 or key[0] != "ssh-ed25519":
                raise VMError("Guest did not provide an SSH host key")
            (state / "known_hosts").write_text(f"[127.0.0.1]:{port} {key[0]} {key[1]}\n")
            subprocess.run(ssh_command(state, config) + ["true"], check=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        print(f"VM ready; share mounted at /mnt/host. State: {state}")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("/tmp/capstone/vm"))
    sub = parser.add_subparsers(dest="action", required=True)
    up = sub.add_parser("up", help="Boot and provision one persistent guest")
    for option in ("qemu", "kernel", "firmware", "rootfs", "share"):
        up.add_argument("--" + option, type=Path, required=True)
    for option in ("launcher", "ssh-server", "module", "job-helper"):
        up.add_argument("--" + option, type=Path, help="Override the installed guest component")
    up.add_argument("--port", type=int, default=0)
    up.add_argument("--memory", default="8G")
    up.add_argument("--boot-timeout", type=float, default=120)
    sub.add_parser("down", help="Stop this VM and discard its temporary disk changes")
    sub.add_parser("restart", help="Explicitly restart using the recorded files and settings")
    sub.add_parser("status")
    sub.add_parser("shell", help="Open an ordinary interactive Linux shell")
    for name in ("exec", "run"):
        cmd = sub.add_parser(name, help="Run guest argv" if name == "exec" else "Run a Capstone application")
        if name == "run":
            cmd.add_argument("--cwd", help="Guest working directory")
            cmd.add_argument("--result", type=Path, help="Write the actual guest exit/signal result as JSON")
            cmd.add_argument("-e", "--env", action="append", default=[], help="Guest NAME=value")
        cmd.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    state = args.state.expanduser().resolve()
    try:
        if args.action == "up":
            return start(args, state)
        if args.action == "restart":
            with lock(state):
                config = json.loads((state / "config.json").read_text())["identity"]
                if running(state):
                    qmp(state, "quit")
                deadline = time.monotonic() + 5
                while running(state):
                    if time.monotonic() >= deadline:
                        raise VMError("VM did not stop after QMP quit")
                    time.sleep(0.05)
            settings = {key: Path(value["path"]) for key, value in config["files"].items()}
            settings.update(share=Path(config["share"]), memory=config["memory"],
                            environment=config["environment"], port=0, boot_timeout=120)
            return start(argparse.Namespace(**settings), state)
        if args.action == "status":
            active = running(state)
            print("running" if active else "stopped")
            return 0 if active else 1
        if args.action == "down":
            with lock(state):
                if (state / "qmp.sock").exists():
                    try:
                        qmp(state, "quit")
                    except (ConnectionRefusedError, FileNotFoundError):
                        pass
            return 0
        if not running(state):
            raise VMError("VM is not running; use up first")
        config = json.loads((state / "config.json").read_text())
        command = ssh_command(state, config, terminal=args.action == "shell")
        if args.action != "shell":
            words = args.command
            if words[:1] == ["--"]:
                words = words[1:]
            if not words:
                raise VMError("An executable and optional arguments are required")
            if args.action == "run":
                return run_application(state, config, words, cwd=args.cwd, environment=args.env, result_path=args.result)
            command.append("exec " + shlex.join(words))
        return subprocess.call(command)
    except (VMError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"capstone-vm: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
