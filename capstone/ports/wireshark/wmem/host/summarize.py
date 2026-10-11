"""Validate independent trace accounting and export portable component evidence."""

import argparse
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common/host"))
from port_support import digest, write_checksums, write_json

MAGIC = 0x31304D454D575357
FIELDS = "magic count mode status completed news allocs frees reallocs free_alls gcs destroys checksum live_allocators regions_created regions_peak".split()
CASES = range(13)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("spatial", type=Path)
p.add_argument("sublet", type=Path)
p.add_argument("security", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()


def report(path):
    r = dict(zip(FIELDS, struct.unpack("<16Q", path.read_bytes())))
    if r["magic"] != MAGIC or r["status"] or r["count"] != r["completed"]:
        p.error(f"invalid report: {path}")
    return r


reports = {"spatial": report(a.spatial / "report.bin"), "sublet": report(a.sublet / "report.bin")}
if reports["spatial"]["mode"] != 0 or reports["sublet"]["mode"] != 1:
    p.error("protection mode mismatch")
raw = a.trace.read_bytes()
header = struct.unpack_from("<16Q", raw)
if header[0] != MAGIC or len(raw) != 128 + 48 * header[1]:
    p.error("invalid trace framing")
events = list(struct.iter_unpack("<6Q", raw[128:]))
if events[-1][0] != 8:
    p.error("missing END")
# Independent accounting: replay the trace's bookkeeping without the allocator.
expected = {k: 0 for k in ("news", "allocs", "frees", "reallocs", "free_alls", "gcs", "destroys", "checksum")}
pools, live = set(), {}
for k, (op, pool, obj, size, kind, arg) in enumerate(events):
    if op == 8:
        if k + 1 != len(events) or pool != len(pools) or obj or size or kind or arg:
            p.error("invalid END")
        break
    if op == 1:
        if pool in pools:
            p.error("duplicate pool identity")
        pools.add(pool)
        expected["news"] += 1
    elif pool not in pools:
        p.error("operation on an unknown pool")
    elif op == 2:
        if obj in live:
            p.error("object identity reused while live")
        live[obj] = pool
        expected["allocs"] += 1
    elif op == 3:
        if live.pop(obj, None) != pool:
            p.error("free of an object the pool does not hold")
        expected["frees"] += 1
    elif op == 4:
        if live.get(obj) != pool:
            p.error("resize of an object the pool does not hold")
        expected["reallocs"] += 1
    elif op == 6:
        expected["gcs"] += 1
    else:
        for o in [o for o, q in live.items() if q == pool]:
            del live[o]
        if op == 5:
            expected["free_alls"] += 1
        else:
            pools.remove(pool)
            expected["destroys"] += 1
    expected["checksum"] = (
        (expected["checksum"] * 33) ^ (op + 7 * pool + 13 * obj + 17 * size + kind + arg)
    ) & ((1 << 64) - 1)
for name, r in reports.items():
    if r["completed"] != len(events) or r["live_allocators"] != len(pools):
        p.error(f"completion mismatch: {name}")
    for key, value in expected.items():
        if r[key] != value:
            p.error(f"independent {name} accounting mismatch: {key}")
    if reports["spatial"][key] != reports["sublet"][key]:
        p.error(f"spatial and sublet reports differ: {key}")
security = json.loads((a.security / "verdicts.json").read_text())
if sorted((r["mode"], r["case"]) for r in security) != [
    (m, c) for m in range(2) for c in CASES
] or not all(r["passed"] for r in security):
    p.error(f"requires all {2 * len(CASES)} passing paired security cases")
manifests = []
for row in security:
    run = Path(row["run"])
    manifest = json.loads((run / "manifest.json").read_text())
    for name, sha in manifest["sha256"].items():
        if digest(run / "share" / name) != sha:
            p.error("security input hash mismatch")
    manifests.append(manifest)
port = Path(__file__).resolve().parent.parent
sources = {
    str(f.relative_to(port)): digest(f)
    for f in sorted(port.rglob("*"))
    if f.is_file()
    and "results" not in f.relative_to(port).parts
    and "__pycache__" not in f.relative_to(port).parts
    and (
        f.suffix in (".c", ".h", ".patch", ".cmake", ".json", ".py", ".manifest")
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
            and (run / "share/report.bin").read_bytes() == (path / "report.bin").read_bytes()
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
        "scope": "wireshark 4.6.8 wmem core and all four allocators; directed QEMU replay and paired lifetime fixtures; not packet dissection in a domain",
        "trace_sha256": digest(a.trace),
        "events": len(events),
        "reports": reports,
        "security": [{k: v for k, v in row.items() if k != "run"} for row in security],
        "replay_manifests": replays,
        "security_manifests": [json.loads(m) for m in sorted({json.dumps(m, sort_keys=True) for m in manifests})],
        "source_sha256": sources,
        "images_sha256": {
            name: digest(Path(os.environ["CAPSTONE_BUILDROOT_DIR"]) / "build/images" / name)
            for name in ("fw_jump.elf", "Image", "rootfs.ext2")
        },
        "limits": "Individual recycler frees are not revoked; regions reissued at exact size only; synthetic payloads; 384 MiB backing budget; no FPGA, timing, or complete-process memory claim",
    },
)
write_checksums(a.output)
print("PASS independent accounting, mode-identical reports and all security verdicts")
