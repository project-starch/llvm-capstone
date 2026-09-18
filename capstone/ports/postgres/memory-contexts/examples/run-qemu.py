#!/usr/bin/env python3
"""Run one allocator client in spatial and/or Sublet Capstone domains."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json


def passed(serial, client, runner_exit):
    return (
        runner_exit == 0
        and re.findall(r"PG_CLIENT (\w+) RESULT (\d+)", serial) == [(client, "0")]
        and "__CAPSTONE_PG_HOST_DONE__" in serial
        and "_FAILED__" not in serial
        and "_BAD__" not in serial
        and "domain halted by capability fault" not in serial
        and "domain capability fault delivered" not in serial
        and "__CAPSTONE_PG_DOMAIN_FAULT__" not in serial
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("client", choices=("allocset", "generation", "slab", "bump"))
    parser.add_argument("--mode", choices=("both", "spatial", "sublet"), default="both")
    parser.add_argument("--domain-build", type=Path, required=True)
    parser.add_argument("--linux-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regions = json.loads((args.domain_build / "regions.json").read_text())
    if regions != json.loads((args.linux_build / "regions.json").read_text()):
        parser.error("domain and guest region settings differ")
    env = dict(os.environ)
    env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
    env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
    env.setdefault("CAPSTONE_GP_NONLIN", "1")
    env.setdefault("CAPSTONE_REV_NODES", "1048576")
    modes = ("spatial", "sublet") if args.mode == "both" else (args.mode,)
    for mode in modes:
        run, hashes = stage_run(
            args.output,
            f"{args.client}-{mode}-",
            {
                "client.dom": args.domain_build
                / f"bin/client-{args.client}-{mode}.dom",
                "loader.user": args.linux_build / "bin/domain-loader",
            },
        )
        # The generic loader takes an input file, but clients do not replay a trace.
        (run / "share/input.bin").write_bytes(b"")
        hashes["input.bin"] = digest(run / "share/input.bin")
        write_json(
            run / "manifest.json",
            {
                "client": args.client,
                "mode": mode,
                "sha256": hashes,
                "regions": regions,
                "revocation_nodes": int(env["CAPSTONE_REV_NODES"]),
                "qemu_sha256": digest(env["CAPSTONE_QEMU_BINARY"]),
                "compiler_sha256": digest(
                    Path(env["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                ),
            },
        )
        print(f"Client artifacts: {run}", flush=True)
        command = (
            "cp /mnt/host/loader.user /tmp/pg-client-loader && "
            "chmod 0755 /tmp/pg-client-loader && "
            "/tmp/pg-client-loader /mnt/host/client.dom /mnt/host/input.bin"
        )
        if mode == "sublet":
            command += " --linear-arena"
        # No tail thread is needed for these short programs; the loader prints
        # their report after return and must finish cleanup before we pass.
        result = run_guest(
            run, command, "__CAPSTONE_PG_HOST_DONE__", env=env, timeout_multiplier=1
        )
        log = run / "serial.log"
        serial = log.read_text(errors="replace") if log.exists() else ""
        ok = passed(serial, args.client, result.returncode)
        write_json(
            run / "verdict.json", {"passed": ok, "runner_exit": result.returncode}
        )
        if not ok:
            raise SystemExit(f"FAIL {args.client}/{mode}; inspect {run}")
        print(f"PASS {args.client}/{mode}", flush=True)


if __name__ == "__main__":
    main()
