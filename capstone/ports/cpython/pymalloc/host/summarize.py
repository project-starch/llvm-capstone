#!/usr/bin/env python3
"""Export checked component results without raw guest transcripts or local paths."""

import argparse
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, write_checksums, write_json
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("native", type=Path)
p.add_argument("spatial", type=Path)
p.add_argument("sublet", type=Path)
p.add_argument("security", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
names = (
    "magic",
    "count",
    "mode",
    "status",
    "completed",
    "allocations",
    "frees",
    "reallocations",
    "arenas",
    "arena_frees",
    "metadata_high_water",
    "request_checksum",
)
reports = {}
for name, path in (
    ("native", a.native),
    ("spatial", a.spatial / "report.bin"),
    ("sublet", a.sublet / "report.bin"),
):
    reports[name] = dict(zip(names, struct.unpack("<12Q", path.read_bytes())))
    if reports[name]["magic"] != 0x31594C50524D5950 or reports[name]["status"] != 0:
        p.error(f"invalid {name} report")
for key in (
    "count",
    "completed",
    "allocations",
    "frees",
    "reallocations",
    "request_checksum",
):
    if len({report[key] for report in reports.values()}) != 1:
        p.error(f"reports disagree on {key}")
if reports["sublet"]["mode"] != 1 or reports["spatial"]["mode"] != 0:
    p.error("protection mode mismatch")
raw = a.trace.read_bytes()
count = struct.unpack_from("<12Q", raw)[1]
if len(raw) != 96 + count * 32 or count != reports["native"]["completed"]:
    p.error("trace length/completion mismatch")
end = struct.unpack_from("<4Q", raw, len(raw) - 32)
if (
    end[0] != 5
    or end[1] != reports["native"]["allocations"] - reports["native"]["frees"]
):
    p.error("trace live-at-end mismatch")
security = json.loads((a.security / "verdicts.json").read_text())
if sorted((row["mode"], row["case"]) for row in security) != [
    (mode, case) for mode in range(2) for case in range(9)
] or not all(row["passed"] for row in security):
    p.error("requires all 18 passing paired security cases")
port = Path(__file__).resolve().parent.parent
sources = {
    str(path.relative_to(port)): digest(path)
    for path in sorted(port.rglob("*"))
    if path.is_file()
    and "results" not in path.relative_to(port).parts
    and (
        path.suffix in (".c", ".h", ".inc", ".patch", ".cmake", ".json", ".py")
        or path.name == "CMakeLists.txt"
    )
}
buildroot = Path(os.environ["CAPSTONE_BUILDROOT_DIR"])
summary = {
    "scope": "CPython 3.13.7 GIL allocator component; native capture and QEMU replay; no FPGA or full interpreter port",
    "workload": "20 rounds of JSON, regex and bytearray operations",
    "trace_sha256": digest(a.trace),
    "live_at_end": end[1],
    "reports": reports,
    "security": [
        {key: value for key, value in row.items() if key != "run"} for row in security
    ],
    "source_sha256": sources,
    "tools_sha256": {
        "clang": digest(Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"]) / "bin/clang"),
        "qemu": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
        **{
            name: digest(buildroot / "build/images" / name)
            for name in ("fw_jump.elf", "Image", "rootfs.ext2")
        },
    },
    "limits": "Fresh replay state; successful MEM/OBJ requests only; synthetic payloads; fixed region budgets; metadata high-water is not RSS; no timing measurements",
}
a.output.mkdir(parents=True, exist_ok=False)
write_json(a.output / "summary.json", summary)
write_checksums(a.output)
