#!/usr/bin/env python3
"""Run allocator-independent quarantine, fallback, and process tests in QEMU."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ports/common/host"))
from port_support import digest, run_guest, stage_run, write_json
from verdicts import fallback_passed, isolation_passed, reentry_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
    env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "90")
    env.setdefault("CAPSTONE_GP_NONLIN", "1")
    env.setdefault("CAPSTONE_REV_NODES", "8388608")
    for case in [0, 1, 2, 3, 4, 5, "processes"]:
        if case == "processes":
            inputs = {
                "launcher.user": args.linux_build / "bin/fault-launcher",
                "supervisor.user": args.linux_build / "bin/fault-supervisor",
                "healthy.dom": args.domain_build / "bin/fault-probe-6.dom",
            }
            inputs.update(
                {
                    f"fault-{i}.dom": args.domain_build / f"bin/fault-probe-{i}.dom"
                    for i in range(4)
                }
            )
            command = (
                "cp /mnt/host/launcher.user /tmp/runtime-launcher && "
                "cp /mnt/host/supervisor.user /tmp/runtime-supervisor && "
                "chmod 0755 /tmp/runtime-launcher /tmp/runtime-supervisor && "
                "/tmp/runtime-supervisor /tmp/runtime-launcher /mnt/host/healthy.dom "
                + " ".join(f"/mnt/host/fault-{i}.dom" for i in range(4))
            )
            done = "__CAPSTONE_RUNTIME_ISOLATION_DONE__"
        else:
            inputs = {
                "reentry.user": args.linux_build / "bin/fault-reentry",
                "fault.dom": args.domain_build / f"bin/fault-probe-{case}.dom",
            }
            command = (
                "cp /mnt/host/reentry.user /tmp/runtime-reentry && "
                "chmod 0755 /tmp/runtime-reentry && "
                "/tmp/runtime-reentry /mnt/host/fault.dom"
            )
            done = "__CAPSTONE_FAULT_REENTRY_DONE__"
        run, hashes = stage_run(args.output, f"runtime-fault-{case}-", inputs)
        write_json(
            run / "manifest.json",
            {
                "case": case,
                "sha256": hashes,
                "qemu_sha256": digest(env["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(
                    Path(env["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                ),
                "revocation_nodes": int(env["CAPSTONE_REV_NODES"]),
            },
        )
        print(f"Runtime fault artifacts: {run}", flush=True)
        result = run_guest(run, command, done, env=env, timeout_multiplier=1)
        log = run / "serial.log"
        serial = log.read_text(errors="replace") if log.exists() else ""
        if case == "processes":
            ok = isolation_passed(serial, result.returncode)
        elif case in (4, 5):
            ok = fallback_passed(serial, case, result.returncode)
        else:
            ok = reentry_passed(serial, case, result.returncode)
        write_json(
            run / "verdict.json", {"passed": ok, "runner_exit": result.returncode}
        )
        if not ok:
            raise SystemExit(f"FAIL {case}; inspect {run}")
        print(f"PASS {case}", flush=True)


if __name__ == "__main__":
    main()
