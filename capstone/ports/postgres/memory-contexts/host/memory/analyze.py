#!/usr/bin/env python3
"""Reconcile QEMU memory snapshots against an independent trace-lifetime model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

HEADER = struct.Struct("<8sIIQIIQII32s")
RECORD = struct.Struct("<IIIIQQQ")


def trace_memory(path, wanted):
    raw = path.read_bytes()
    header = HEADER.unpack_from(raw)
    if header[0] != b"A11TRACE" or header[1:3] != (1, RECORD.size) or header[5]:
        raise ValueError("expected a flattened A11 trace")
    contexts, objects = {}, {}
    live = count = peak = 0
    snapshots = {}

    def remove_payload(cid):
        nonlocal live, count
        for oid in contexts[cid]["objects"]:
            owner, size = objects.pop(oid)
            assert owner == cid
            live -= size
            count -= 1
        contexts[cid]["objects"].clear()

    def delete(cid):
        for child in list(contexts[cid]["children"]):
            delete(child)
        remove_payload(cid)
        parent = contexts[cid]["parent"]
        if parent:
            contexts[parent]["children"].remove(cid)
        del contexts[cid]

    def drop(oid):
        nonlocal live, count
        cid, size = objects.pop(oid)
        contexts[cid]["objects"].remove(oid)
        live -= size
        count -= 1

    def add(cid, oid, size):
        nonlocal live, count
        assert oid and oid not in objects
        objects[oid] = (cid, size)
        contexts[cid]["objects"].add(oid)
        live += size
        count += 1

    events = 0
    for events, rec in enumerate(RECORD.iter_unpack(raw[HEADER.size :]), 1):
        op, cid, oid, aux, size, s2, s3 = rec
        if 6 <= op <= 9:
            assert cid not in contexts
            contexts[cid] = dict(parent=aux, children=set(), objects=set())
            if aux:
                contexts[aux]["children"].add(cid)
        elif op == 1:
            add(cid, oid, size)
        elif op == 2:
            assert objects[oid][0] == cid
            drop(oid)
        elif op == 3:
            drop(oid)
            if aux:
                add(cid, aux, size)
        elif op == 4:
            for child in list(contexts[cid]["children"]):
                delete(child)
            remove_payload(cid)
        elif op == 5:
            delete(cid)
        elif op not in (0, 10):
            raise ValueError(f"unknown trace operation {op}")
        peak = max(peak, live)
        if events in wanted:
            snapshots[events] = (live, count, len(contexts), op)
    if op != 0:
        raise ValueError("trace lacks footer")
    return snapshots, dict(
        events=events,
        payload_peak_bytes=peak,
        final_payload_bytes=live,
        final_objects=count,
        final_contexts=len(contexts),
    )


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    verdict = json.loads((run / "verdict.json").read_text())
    if not verdict["passed"] or not manifest["memory_profile"]:
        raise ValueError("run is not a completed memory profile")
    text = (
        run
        / (
            "share/payload.log"
            if (run / "share/payload.log").exists()
            else "serial.log"
        )
    ).read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    rows, columns = [], None
    for line in text.splitlines():
        if line.startswith("PGMEM_HEADER,"):
            if columns:
                raise ValueError("multiple profile headers")
            columns = line.split(",")[1:]
        elif line.startswith("PGMEM,"):
            values = line.split(",")[1:]
            if columns is None or len(values) != len(columns):
                raise ValueError(f"corrupted memory row: {line}")
            rows.append(
                dict(zip(columns, [values[0], *map(int, values[1:])], strict=True))
            )
    if not rows or "__CAPSTONE_PG_MEMORY_DONE__" not in text:
        raise ValueError("missing complete memory profile")
    trace = run / "share/trace.a11"
    if (
        hashlib.sha256(trace.read_bytes()).hexdigest()
        != manifest["sha256"]["trace.a11"]
    ):
        raise ValueError("trace hash changed")
    expected, oracle = trace_memory(trace, {r["event"] for r in rows})
    peaks = {}
    for r in rows:
        observed = (
            r["live_payload_bytes"],
            r["live_objects"],
            r["live_contexts"],
            r["op"],
        )
        if observed != expected[r["event"]]:
            raise ValueError(
                f"payload oracle mismatch at {r['event']}: {observed} != {expected[r['event']]}"
            )
        if r["backing_bytes"] > r["arena_capacity_bytes"]:
            raise ValueError("backing exceeds arena")
        if r["assigned_backing_bytes"] > r["backing_bytes"]:
            raise ValueError("assigned backing exceeds acquired backing")
        if r["metadata_records_live_bytes"] > r["metadata_records_reserved_bytes"]:
            raise ValueError("metadata exceeds record capacity")
        if any(value < 0 for key, value in r.items() if key != "kind"):
            raise ValueError("negative memory counter")
        if r["block_bytes"] + r["stranded_block_bytes"] > r["assigned_backing_bytes"]:
            raise ValueError("carved blocks exceed assigned backing")
        if r["live_payload_bytes"] > r["block_bytes"]:
            raise ValueError("live payload exceeds manager backing blocks")
        if r["kind"] != "event":
            if r["kind"] in peaks:
                raise ValueError("duplicate exact peak")
            peaks[r["kind"]] = r
    if set(peaks) != {"payload_peak", "backing_peak", "metadata_peak", "tracked_peak"}:
        raise ValueError("incomplete exact peaks")
    if peaks["payload_peak"]["live_payload_bytes"] != oracle["payload_peak_bytes"]:
        raise ValueError("exact peak disagrees with trace oracle")
    series = [r for r in rows if r["kind"] == "event"]
    if series[-1]["event"] != oracle["events"]:
        raise ValueError("incomplete event series")
    with (run / "memory.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        program=manifest["program"],
        trace_sha256=manifest["sha256"]["trace.a11"],
        oracle=oracle,
        rows=len(series),
        peaks=peaks,
        final=series[-1],
        regions=manifest["regions"],
        node_capacity=manifest["node_capacity"],
        scope="Allocator backing and record storage; node counts are cumulative demand, not live nodes. Global node/tag bytes and full application memory are not measured.",
    )
    (run / "memory-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
