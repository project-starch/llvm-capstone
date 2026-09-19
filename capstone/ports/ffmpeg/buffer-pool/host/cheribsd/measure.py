#!/usr/bin/env python3
"""Measure the spatial arena port in fresh CheriBSD/QEMU snapshots."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import shutil
import time

import pexpect

PORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PORT / "host/memory"))
from measure import observations


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("campaign", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--rootfs", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--workload", action="append")
    p.add_argument("--port", type=int, default=10426)
    p.add_argument("--runtime-revocation", choices=("off", "on"), default="off")
    p.add_argument("--label", default="cheri-spatial-arena")
    p.add_argument(
        "--heap-probe", type=Path, help="Optional separate outer-heap lifetime control"
    )
    a = p.parse_args()
    if a.repetitions < 1:
        p.error("repetitions must be positive")
    manifest = json.loads((a.campaign / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("incomplete source campaign")
    workloads = [w[0] for w in manifest["workloads"]]
    if a.workload:
        if not set(a.workload) <= set(workloads):
            p.error("unknown workload")
        workloads = a.workload
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, a.output / "collector.py")
    key = a.output / "guest-key"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "replay-measurement",
            "-f",
            str(key),
        ],
        check=True,
    )
    public_key = key.with_suffix(".pub").read_text().strip()
    qemu = a.sdk / "bin/qemu-system-riscv64cheri"
    bios = a.sdk / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"
    kernel = a.rootfs / "boot/kernel/kernel"
    fingerprints = {
        str(f): digest(f)
        for f in (
            qemu,
            bios,
            kernel,
            a.image,
            a.build / "bin/replay",
            a.build / "bin/pool-security",
            a.sdk / "bin/clang",
            Path(__file__).resolve(),
            a.rootfs / "lib/libc.so.7",
            a.build / "CMakeCache.txt",
            a.build / "compile_commands.json",
        )
    }
    if a.heap_probe:
        fingerprints[str(a.heap_probe)] = digest(a.heap_probe)
    for workload in workloads:
        for name in ("commands.bin", "recorded.bin"):
            f = a.campaign / "recordings" / workload / name
            fingerprints[str(f)] = digest(f)
    report = dict(
        schema="ffpool-cheribsd-v1",
        label=a.label,
        scope="bounded spatial leases within reusable arenas",
        runtime_revocation=a.runtime_revocation,
        repetitions=a.repetitions,
        heap_probe_requested=bool(a.heap_probe),
        fingerprints=fingerprints,
        results=[],
        status="running",
    )

    def save():
        (a.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    ssh_options = [
        "-i",
        str(key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
        "-o",
        "LogLevel=ERROR",
    ]

    def ssh(command, **kwargs):
        return subprocess.run(
            ["ssh", *ssh_options, "-p", str(a.port), "root@127.0.0.1", command],
            timeout=120,
            **kwargs,
        )

    def scp(source, dest):
        subprocess.run(
            ["scp", "-O", *ssh_options, "-P", str(a.port), str(source), str(dest)],
            check=True,
            timeout=120,
            capture_output=True,
        )

    policy = (
        "_RUNTIME_REVOCATION_DISABLE=1"
        if a.runtime_revocation == "off"
        else "_RUNTIME_REVOCATION_ENABLE=1"
    )
    try:
        for workload in workloads:
            for repeat in range(1, a.repetitions + 1):
                run = a.output / f"{workload}-{repeat}"
                run.mkdir()
                entry = dict(workload=workload, repetition=repeat, status="running")
                report["results"].append(entry)
                argv = [
                    str(qemu),
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
                    str(bios),
                    "-kernel",
                    str(kernel),
                    "-drive",
                    f"file={a.image},format=raw,if=none,id=disk",
                    "-device",
                    "virtio-blk-device,drive=disk",
                    "-netdev",
                    f"user,id=net,hostfwd=tcp:127.0.0.1:{a.port}-:22",
                    "-device",
                    "virtio-net-device,netdev=net",
                ]
                (run / "qemu-command.json").write_text(
                    json.dumps(argv, indent=2) + "\n"
                )
                print(f"BOOT {a.label} {workload} {repeat}", flush=True)
                guest = None
                with (run / "serial.log").open("w") as log:
                    try:
                        guest = pexpect.spawn(
                            argv[0],
                            argv[1:],
                            encoding="utf-8",
                            codec_errors="replace",
                            timeout=180,
                            env={**os.environ, "TMPDIR": str(run)},
                        )
                        guest.logfile_read = log
                        guest.expect(
                            "Enter full pathname of shell or RETURN for /bin/sh:"
                        )
                        guest.sendline("")
                        guest.expect(r"# ")
                        # Keep every command below the guest console line limit.
                        setup = [
                            "stty -echo",
                            "mount -u -w /",
                            "ifconfig vtnet0 inet 10.0.2.15 netmask 255.255.255.0",
                            "ifconfig lo0 inet 127.0.0.1",
                            "route add default 10.0.2.2",
                            "mkdir -p /root/.ssh",
                            "chmod 700 /root/.ssh",
                            'printf "%s\\n" '
                            + shlex.quote(public_key)
                            + " > /root/.ssh/authorized_keys",
                            "chmod 600 /root/.ssh/authorized_keys",
                            "service sshd onestart",
                        ]
                        for command in setup:
                            guest.sendline(command)
                            guest.expect(r"# ", timeout=45)
                        # sshd daemonizes before its listener is necessarily ready.
                        # Poll readiness only; never retry a replay or hide a failed run.
                        deadline = time.monotonic() + 45
                        ready = None
                        with (run / "ssh-readiness.log").open("w") as readiness:
                            while time.monotonic() < deadline:
                                ready = ssh(
                                    "mkdir -p /tmp/replay; ulimit -c 0",
                                    capture_output=True,
                                    text=True,
                                )
                                readiness.write(f"{ready.returncode}: {ready.stderr}\n")
                                readiness.flush()
                                if ready.returncode == 0:
                                    break
                                time.sleep(1)
                        if ready is None or ready.returncode:
                            guest.sendline(
                                "sockstat -4 -l; service sshd status; tail /var/log/auth.log"
                            )
                            guest.expect(r"# ", timeout=15)
                            raise RuntimeError(
                                "SSH readiness failed; see ssh-readiness.log and serial.log"
                            )
                        for binary in ("replay", "pool-security"):
                            scp(
                                a.build / "bin" / binary,
                                "root@127.0.0.1:/tmp/replay/" + binary,
                            )
                        scp(
                            a.campaign / "recordings" / workload / "commands.bin",
                            "root@127.0.0.1:/tmp/replay/input.bin",
                        )
                        info = ssh(
                            "uname -a; sysctl security.cheri",
                            capture_output=True,
                            text=True,
                        )
                        (run / "guest-info.txt").write_text(info.stdout + info.stderr)
                        command = f"cd /tmp/replay && ulimit -c 0 && env {policy} ./replay input.bin output.bin 0"
                        result = ssh(command, capture_output=True, text=True)
                        (run / "stdout.txt").write_text(result.stdout)
                        (run / "stderr.txt").write_text(result.stderr)
                        entry["exit_code"] = result.returncode
                        expected_runtime = int(a.runtime_revocation == "on")
                        if (
                            f"FF2 CHERI runtime_revoke={expected_runtime} pointer_bytes=16"
                            not in result.stdout
                        ):
                            raise ValueError(
                                "runtime revocation policy was not confirmed by the replay process"
                            )
                        if result.returncode:
                            raise RuntimeError(
                                f"replay failed with status {result.returncode}"
                            )
                        scp("root@127.0.0.1:/tmp/replay/output.bin", run / "output.bin")
                        stats, _ = observations(
                            run / "output.bin",
                            a.campaign / "recordings" / workload / "recorded.bin",
                            0,
                        )
                        header = struct.unpack(
                            "<16Q", (run / "output.bin").read_bytes()[:128]
                        )
                        entry.update(
                            measurements=stats,
                            output_sha256=digest(run / "output.bin"),
                            cheri=dict(
                                leases=header[12],
                                cumulative_bounds_slack_bytes=header[13],
                                max_bounds_slack_bytes=header[14],
                                pointer_bytes=header[15],
                            ),
                        )
                        if header[15] != 16:
                            raise ValueError("not a confirmed 128-bit purecap run")
                        # Companion controls use separate processes after the measurement.
                        entry["controls"] = []
                        if repeat == 1 and workload == workloads[0]:
                            for case in (0, 3, 5, 10, 11):
                                h = [0x4650465452433032, 1] + [0] * 10 + [case, 1, 0, 0]
                                fixture = run / f"control-{case}.bin"
                                fixture.write_bytes(
                                    struct.pack("<16Q", *h) + bytes(128)
                                )
                                scp(fixture, "root@127.0.0.1:/tmp/replay/control.bin")
                                control_command = f"cd /tmp/replay && ulimit -c 0 && env {policy} ./pool-security control.bin control-out.bin 0"
                                # Keep a shell parent: SSH otherwise reports an exit-signal
                                # as 255 rather than the child's conventional 128 + signal.
                                control_command += (
                                    '; rc=$?; echo FF2_CONTROL_EXIT=$rc; exit "$rc"'
                                )
                                control = ssh(
                                    "sh -c " + shlex.quote(control_command),
                                    capture_output=True,
                                    text=True,
                                )
                                (run / f"control-{case}.txt").write_text(
                                    control.stdout + control.stderr
                                )
                                expected_code = 162 if case in (10, 11) else 0
                                passed = (
                                    control.returncode == expected_code
                                    and f"FF2_PROBE case={case} ready" in control.stdout
                                    and f"FF2_CONTROL_EXIT={expected_code}"
                                    in control.stdout
                                )
                                entry["controls"].append(
                                    dict(
                                        case=case,
                                        exit_code=control.returncode,
                                        passed=passed,
                                    )
                                )
                                if not passed:
                                    raise ValueError(
                                        f"companion control {case} failed: {control.returncode}"
                                    )
                        if a.heap_probe and repeat == 1 and workload == workloads[0]:
                            scp(a.heap_probe, "root@127.0.0.1:/tmp/replay/heap-probe")
                            entry["heap_controls"] = []
                            for stale in (False, True):
                                command = (
                                    f"cd /tmp/replay && ulimit -c 0 && env {policy} ./heap-probe"
                                    + (" stale" if stale else "")
                                )
                                command += '; rc=$?; echo HEAP_EXIT=$rc; exit "$rc"'
                                check = ssh(
                                    "sh -c " + shlex.quote(command),
                                    capture_output=True,
                                    text=True,
                                )
                                (run / f"heap-{int(stale)}.txt").write_text(
                                    check.stdout + check.stderr
                                )
                                expected = (
                                    162 if stale and a.runtime_revocation == "on" else 0
                                )
                                passed = (
                                    check.returncode == expected
                                    and f"HEAP_EXIT={expected}" in check.stdout
                                )
                                passed &= (
                                    f'HEAP_PROBE runtime={int(a.runtime_revocation == "on")}'
                                    in check.stdout
                                )
                                if stale:
                                    passed &= (
                                        "HEAP_PROBE stale-access-ready" in check.stdout
                                    )
                                entry["heap_controls"].append(
                                    dict(
                                        stale=stale,
                                        exit_code=check.returncode,
                                        passed=passed,
                                    )
                                )
                                if not passed:
                                    raise ValueError(
                                        "outer-heap lifetime control failed"
                                    )
                        entry["status"] = "passed"
                        print(
                            f"PASS {workload} {repeat}: payload={header[4]} metadata={header[3]} slack_max={header[14]}",
                            flush=True,
                        )
                    except Exception as e:
                        entry.update(status="failed", error=str(e))
                        if isinstance(e, subprocess.CalledProcessError):
                            (run / "transport-error.txt").write_text(
                                str(e.stdout) + "\n" + str(e.stderr)
                            )
                        raise
                    finally:
                        if guest is not None:
                            guest.terminate(force=True)
                            guest.close()
                        save()
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
