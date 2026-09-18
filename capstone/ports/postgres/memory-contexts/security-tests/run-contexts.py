#!/usr/bin/env python3
"""Paired upstream/Sublet context policy, liveness and exact-access fault tests."""

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, run_guest, stage_run, write_json
from context_checks import classify

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=Path)
p.add_argument("--domain-build", type=Path, required=True)
p.add_argument("--linux-build", type=Path, required=True)
p.add_argument("--kinds", default="0,1,2")
p.add_argument("--cases", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14")
p.add_argument("--modes", default="spatial,sublet")
p.add_argument(
    "--resume",
    action="store_true",
    help="Resume this exact suite directory after verifying passing inputs",
)
a = p.parse_args()
regions = json.loads((a.domain_build / "regions.json").read_text())
if regions != json.loads((a.linux_build / "regions.json").read_text()):
    p.error("domain and guest region configurations differ")
a.output.mkdir(parents=True, exist_ok=True)
if not a.resume:
    a.output = Path(tempfile.mkdtemp(prefix="suite-", dir=a.output))
print(f"Context suite artifacts: {a.output}", flush=True)
verdicts = []
if a.resume:
    previous = a.output / "verdicts.json"
    if not previous.exists():
        p.error("resume needs an existing suite verdicts.json")
    for row in json.loads(previous.read_text()):
        if not row["passed"]:
            continue  # the failed attempt's own verdict.json remains immutable
        run = Path(row["run"])
        manifest = json.loads((run / "manifest.json").read_text())
        current = {
            "context.dom": a.domain_build / f"bin/contexts-{row['mode']}.dom",
            "loader.user": a.linux_build / "bin/domain-loader",
        }
        if any(
            digest(path) != manifest["sha256"][name] for name, path in current.items()
        ):
            p.error("resume binaries changed; start a new suite")
        if (
            manifest["regions"] != regions
            or manifest["qemu_sha256"] != digest(os.environ["CAPSTONE_QEMU_BINARY"])
            or manifest["compiler_sha256"]
            != digest(Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang")
            or manifest["nodes"] != int(os.environ.get("CAPSTONE_REV_NODES", "1048576"))
        ):
            p.error("resume runtime changed; start a new suite")
        if any(
            digest(run / "share" / name) != sha
            for name, sha in manifest["sha256"].items()
        ):
            p.error("resume staged inputs changed")
        if (run / "share/selection.bin").read_bytes() != struct.pack(
            "<II", row["kind"], row["case"]
        ):
            p.error("resume case identity mismatch")
        verdicts.append(row)
    if len({(r["kind"], r["case"], r["mode"]) for r in verdicts}) != len(verdicts):
        p.error("duplicate passing verdicts")
requested = 0
for kind in map(int, a.kinds.split(",")):
    for test in map(int, a.cases.split(",")):
        if kind not in (0, 1, 2) or test not in range(15):
            p.error("invalid kind or case")
        if (
            (kind == 2 and test in (1, 5, 11))
            or (kind == 1 and test in (7, 12))
            or (kind != 0 and test == 8)
            or (kind != 2 and test == 13)
        ):
            continue  # Bump has no free/realloc; Slab requires one fixed size.
        policy = {}
        for mode in a.modes.split(","):
            if mode not in ("spatial", "sublet"):
                p.error("invalid mode")
            requested += 1
            prior = next(
                (
                    r
                    for r in verdicts
                    if (r["kind"], r["case"], r["mode"]) == (kind, test, mode)
                ),
                None,
            )
            if prior:
                if test == 10:
                    policy[mode] = prior["policy"]
                continue
            run, hashes = stage_run(
                a.output,
                f"{kind}-{test}-{mode}-",
                {
                    "context.dom": a.domain_build / f"bin/contexts-{mode}.dom",
                    "loader.user": a.linux_build / "bin/domain-loader",
                },
            )
            (run / "share/selection.bin").write_bytes(struct.pack("<II", kind, test))
            hashes["selection.bin"] = digest(run / "share/selection.bin")
            write_json(
                run / "manifest.json",
                {
                    "sha256": hashes,
                    "regions": regions,
                    "qemu_sha256": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
                    "compiler_sha256": digest(
                        Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"
                    ),
                    "nodes": int(os.environ.get("CAPSTONE_REV_NODES", "1048576")),
                },
            )
            command = "cp /mnt/host/loader.user /tmp/pg-loader && chmod 0755 /tmp/pg-loader && /tmp/pg-loader /mnt/host/context.dom /mnt/host/selection.bin --tail"
            if mode == "sublet":
                command += " --linear-arena"
            env = dict(os.environ)
            env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
            env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
            result = run_guest(
                run, command, "__CAPSTONE_PG_HOST_DONE__", env=env, timeout_multiplier=1
            )
            serial = (
                (run / "serial.log").read_text(errors="replace")
                if (run / "serial.log").exists()
                else ""
            )
            row = classify(serial, kind, test, mode, result.returncode)
            row["run"] = str(run)
            if "policy" in row:
                policy[mode] = row["policy"]
            verdicts.append(row)
            write_json(run / "verdict.json", row)
            write_json(a.output / "verdicts.json", verdicts)
            print(
                f"{'PASS' if row['passed'] else 'FAIL'} kind={kind} case={test} mode={mode}",
                flush=True,
            )
            if not row["passed"]:
                raise SystemExit(1)
        if len(policy) == 2 and policy["spatial"] != policy["sublet"]:
            verdicts[-1]["passed"] = False
            verdicts[-1]["error"] = "allocation policy counters differ"
            write_json(a.output / "verdicts.json", verdicts)
            raise SystemExit(f"Policy mismatch for kind {kind}: {policy}")
if not requested:
    p.error("no applicable allocator test arms selected")
print(f"PASS: {requested} requested allocator arms", flush=True)
