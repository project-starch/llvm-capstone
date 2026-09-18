"""Validate independent trace accounting and export portable component evidence."""

import argparse
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, write_checksums, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("recording", type=Path)
p.add_argument("spatial", type=Path)
p.add_argument("sublet", type=Path)
p.add_argument("security", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
fields = "magic count mode status completed inits objects resets owned_frees borrowed_frees rebinds peak_used checksum live_contexts object_header_size layout_checksum".split()


def report(path):
    r = dict(zip(fields, struct.unpack("<16Q", path.read_bytes())))
    if r["magic"] != 0x315854434C4D4747 or r["status"] or r["count"] != r["completed"]:
        p.error(f"invalid report: {path}")
    return r


reports = {
    "native": report(a.recording / "native.bin"),
    "full_reference": report(a.recording / "full-reference.bin"),
    "spatial": report(a.spatial / "report.bin"),
    "sublet": report(a.sublet / "report.bin"),
}
for key in fields:
    if key != "rebinds" and reports["native"][key] != reports["full_reference"][key]:
        p.error(f"full-library reference mismatch: {key}")
if reports["spatial"]["mode"] != 0 or reports["sublet"]["mode"] != 1:
    p.error("protection mode mismatch")
raw = (a.recording / "trace.bin").read_bytes()
header = struct.unpack_from("<16Q", raw)
if (
    header[0] != 0x315854434C4D4747
    or header[14] != 32
    or len(raw) != 128 + 48 * header[1]
):
    p.error("invalid trace framing")
events = list(struct.iter_unpack("<6Q", raw[128:]))
if events[-1][0] != 5:
    p.error("missing END")
for name, r in reports.items():
    expected = {
        key: 0
        for key in (
            "inits",
            "objects",
            "resets",
            "owned_frees",
            "borrowed_frees",
            "peak_used",
            "checksum",
            "layout_checksum",
        )
    }
    contexts = {}
    for k, (op, c, b, n, t, arg) in enumerate(events):
        if op == 5:
            if k + 1 != len(events) or c != len(contexts) or b or n or t or arg:
                p.error("invalid END")
            break
        if op == 1:
            if c in contexts:
                p.error("overlapping context identity")
            contexts[c] = [b, arg, 0]
            expected["inits"] += 1
        elif op in (2, 3, 4):
            if c not in contexts or contexts[c][0] != b:
                p.error("invalid context lifetime")
            if op == 2:
                offset = contexts[c][2] + r["object_header_size"]
                contexts[c][2] = offset + ((n + 15) // 16) * 16
                expected["objects"] += 1
                expected["peak_used"] = max(expected["peak_used"], contexts[c][2])
                expected["layout_checksum"] = (
                    (expected["layout_checksum"] * 33)
                    ^ (contexts[c][2] + 7 * offset + t)
                ) & ((1 << 64) - 1)
            elif op == 3:
                contexts[c][2] = 0
                expected["resets"] += 1
            else:
                expected["owned_frees" if contexts[c][1] else "borrowed_frees"] += 1
                del contexts[c]
        else:
            p.error("unknown event")
        expected["checksum"] = (
            (expected["checksum"] * 33) ^ (op + 7 * c + 13 * b + 17 * n + t + arg)
        ) & ((1 << 64) - 1)
    if r["completed"] != len(events) or r["live_contexts"] != len(contexts):
        p.error(f"completion mismatch: {name}")
    for key, value in expected.items():
        if r[key] != value:
            p.error(f"independent {name} accounting mismatch: {key}")
security = json.loads((a.security / "verdicts.json").read_text())
if sorted((r["mode"], r["case"]) for r in security) != [
    (m, c) for m in range(2) for c in range(10)
] or not all(r["passed"] for r in security):
    p.error("requires all 20 passing paired security cases")
manifests = []
for row in security:
    run = Path(row["run"])
    manifest = json.loads((run / "manifest.json").read_text())
    for name, sha in manifest["sha256"].items():
        if digest(run / "share" / name) != sha:
            p.error("security input hash mismatch")
    manifests.append(manifest)
security_inputs = {json.dumps(m, sort_keys=True) for m in manifests}
recording = json.loads((a.recording / "manifest.json").read_text())
if recording["trace_sha256"] != digest(a.recording / "trace.bin"):
    p.error("recording checksum mismatch")
port = Path(__file__).resolve().parent.parent
sources = {
    str(f.relative_to(port)): digest(f)
    for f in sorted(port.rglob("*"))
    if f.is_file()
    and "results" not in f.relative_to(port).parts
    and (
        f.suffix in (".c", ".h", ".inc", ".patch", ".cmake", ".json", ".py")
        or f.name == "CMakeLists.txt"
    )
}
replays = {}
for name, path in (("spatial", a.spatial), ("sublet", a.sublet)):
    passed = []
    for run in sorted(path.glob("qemu-*")):
        verdict = run / "verdict.json"
        if (
            verdict.exists()
            and json.loads(verdict.read_text())["passed"]
            and (run / "share/report.bin").read_bytes()
            == (path / "report.bin").read_bytes()
        ):
            m = json.loads((run / "manifest.json").read_text())
            for f, sha in m["sha256"].items():
                if digest(run / "share" / f) != sha:
                    p.error("replay input hash mismatch")
            passed.append(m)
    if not passed:
        p.error("missing passing replay provenance")
    replays[name] = passed
a.output.mkdir(parents=True, exist_ok=False)
write_json(
    a.output / "summary.json",
    {
        "scope": "whisper.cpp 1.9.4 ggml context allocator; native transcription capture and QEMU replay; not inference in a domain",
        "recording": recording,
        "reports": reports,
        "security": [{k: v for k, v in row.items() if k != "run"} for row in security],
        "replay_manifests": replays,
        "security_manifests": [json.loads(m) for m in sorted(security_inputs)],
        "source_sha256": sources,
        "images_sha256": {
            name: digest(
                Path(os.environ["CAPSTONE_BUILDROOT_DIR"]) / "build/images" / name
            )
            for name in ("fw_jump.elf", "Image", "rootfs.ext2")
        },
        "limits": "Exclusive owner rebind contract; synthetic payloads; header-size-adjusted capacities; 384 MiB backing budget; no FPGA, timing, or complete-process memory claim",
    },
)
write_checksums(a.output)
print("PASS independent accounting, full-library reference and all security verdicts")
