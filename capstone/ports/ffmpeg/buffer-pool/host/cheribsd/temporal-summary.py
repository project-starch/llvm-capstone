#!/usr/bin/env python3
"""Validate and summarize fixed-payload temporal pool churn in QEMU."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))
from measure import observations


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def colors(text):
    fields = ("round", "busy", "live", "peak", "issued", "freed")
    return [
        dict(zip(fields, map(int, m)))
        for m in re.findall(
            r"FF2_COLORS round=(\d+) busy=(\d+) live=(\d+) peak=(\d+) issued=(\d+) freed=(\d+)",
            text,
        )
    ]


def nodes(text):
    rows = []
    for marker, section in re.findall(
        r"Print = Scalar\(0xff20([0-9a-f]{12})\)(.*?)(?=Print = Scalar\(0xff20|\Z)",
        text,
        re.S,
    ):
        match = re.search(
            r"REV-NODES alloced_n=(\d+) free_list=(\d+) pool=(\d+)", section
        )
        require(match is not None, "node checkpoint has no counters")
        high, free, capacity = map(int, match.groups())
        require(0 <= free <= high <= capacity, "invalid node accounting")
        rows.append(
            dict(
                round=int(marker, 16),
                high_water=high,
                free=free,
                not_on_free_list=high - free,
                capacity=capacity,
            )
        )
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("picasso", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("capstone", nargs="+", type=Path)
    p.add_argument("--extension", action="store_true")
    a = p.parse_args()
    manifest = json.loads((a.picasso / "manifest.json").read_text())
    require(manifest["status"] == "complete", "incomplete PICASSO campaign")
    require(
        manifest["lease_protection"] == "picasso"
        and manifest["runtime_revocation"] == "on",
        "wrong protection arm",
    )
    repetitions, rounds = (1, 2200000) if a.extension else (3, 300000)
    workloads = ("short",) if a.extension else ("short", "turnover", "larger")
    require(len(a.capstone) == repetitions, "wrong Capstone repetition count")
    require(
        manifest["repetitions"] == repetitions and manifest["churn_rounds"] == rounds,
        "campaign differs from the registered protocol",
    )
    fingerprints = {}
    for filename, expected in manifest["fingerprints"].items():
        f = Path(filename)
        if f.name == "measure.py":
            f = a.picasso / "collector.py"
        require(digest(f) == expected, "PICASSO artifact hash changed: " + f.name)
        # Preserve duplicate filenames without exporting absolute host paths.
        fingerprints[str(len(fingerprints)) + ":" + f.name] = expected
    result = dict(
        schema="ffpool-temporal-reuse-v1",
        rounds=rounds,
        repetitions=repetitions,
        picasso_fingerprints=fingerprints,
        replay=[],
        churn=[],
        controls=[],
        evidence=[],
    )
    native_manifest = json.loads((a.native / "manifest.json").read_text())
    require(native_manifest["status"] == "complete", "incomplete native source")
    result["capstone_platform_fingerprints"] = {}
    platform_names = {
        "Image",
        "fw_jump.elf",
        "rootfs.ext2",
        "clang",
        "qemu-system-riscv64",
        "domain-loader",
    }
    for filename, expected in native_manifest["fingerprints"].items():
        f = Path(filename)
        if f.name in platform_names:
            require(digest(f) == expected, "Capstone platform changed: " + f.name)
            result["capstone_platform_fingerprints"][f.name] = expected
    require(
        set(result["capstone_platform_fingerprints"]) == platform_names,
        "missing Capstone platform identity",
    )

    def evidence(path, label):
        result["evidence"].append(dict(artifact=label, sha256=digest(path)))

    seen = set()
    for entry in manifest["results"]:
        workload, repetition = entry["workload"], entry["repetition"]
        require(entry["status"] == "passed", "replay failed")
        require((workload, repetition) not in seen, "duplicate replay")
        seen.add((workload, repetition))
        run = a.picasso / f"{workload}-{repetition}"
        require(
            digest(run / "output.bin") == entry["output_sha256"], "replay bytes changed"
        )
        stats, _ = observations(
            run / "output.bin", a.native / "recordings" / workload / "recorded.bin", 2
        )
        result["replay"].append(
            dict(
                workload=workload,
                repetition=repetition,
                measurements=stats,
                sha256=entry["output_sha256"],
            )
        )
        if workload != "short":
            continue
        require(
            {c["case"] for c in entry["controls"]} == {0, 3, 5, 10, 11, 12, 13},
            "missing companion controls",
        )
        for control in entry["controls"]:
            case = control["case"]
            expected = 0 if case in (0, 13) else 162
            path = run / f"control-{case}.txt"
            text = path.read_text()
            require(
                control["passed"]
                and control["exit_code"] == expected
                and f"FF2_PROBE case={case} ready" in text
                and f"FF2_CONTROL_EXIT={expected}" in text,
                "PICASSO control failed",
            )
            evidence(path, f"picasso/{repetition}/control-{case}")
            result["controls"].append(
                dict(
                    arm="picasso", repetition=repetition, case=case, exit_code=expected
                )
            )
            if case not in (12, 13):
                continue
            checkpoints = colors(text)
            points = [r for r in checkpoints if r["round"]]
            require(
                [r["round"] for r in points]
                == [1] + list(range(10000, rounds + 1, 10000)),
                "missing PICASSO checkpoints",
            )
            require(
                all(
                    r["live"] == 2
                    and r["peak"] == 2
                    and r["issued"] == r["round"] + 2
                    and r["freed"] == r["round"]
                    for r in points
                ),
                "token accounting mismatch",
            )
            sweeps = re.findall(r"revoke counter: (\d+)", text)
            footprint = re.search(
                r"FF2 status=0 events=1 metadata=(\d+) payload=(\d+)", text
            )
            if case == 13:
                require(
                    len(sweeps) == 1 and footprint is not None,
                    "missing completion accounting",
                )
                require(int(footprint[2]) == 128, "payload is not constant-size")
            result["churn"].append(
                dict(
                    arm="picasso",
                    repetition=repetition,
                    case=case,
                    checkpoints=checkpoints,
                    completed_sweeps=int(sweeps[0]) if sweeps else None,
                    metadata_carved=int(footprint[1]) if footprint else None,
                    payload_carved=int(footprint[2]) if footprint else None,
                    token_requested_peak_bytes=128,
                )
            )
    require(
        seen == {(w, r) for w in workloads for r in range(1, repetitions + 1)},
        "incomplete replay matrix",
    )
    cap_binary = None
    for repetition, campaign in enumerate(a.capstone, 1):
        verdicts = json.loads((campaign / "verdicts.json").read_text())
        expected_cases = {12, 13} if a.extension else {0, 3, 5, 12, 13}
        require(
            {v["case"] for v in verdicts} == expected_cases, "missing Capstone cases"
        )
        for v in verdicts:
            case = v["case"]
            require(
                v["passed"] and v["mode"] == 2 and v["rounds"] == rounds,
                "Capstone control failed",
            )
            run = campaign / f"mode-2-case-{case}"
            binary_hash = digest(run / "share/security.dom")
            require(
                digest(run / "share/host.user")
                == result["capstone_platform_fingerprints"]["domain-loader"],
                "Capstone loader differs from pinned platform",
            )
            cap_binary = cap_binary or binary_hash
            require(binary_hash == cap_binary, "Capstone binary differs across runs")
            text = (run / "serial.log").read_text()
            require(
                f"Print = Scalar(0x{0xff25000000000000 | case:x})" in text,
                "missing Capstone setup marker",
            )
            if case in (3, 5, 12):
                require(
                    v["cause"] in (24, 25) and v["pc"] == v["expected_pc"],
                    "wrong fault",
                )
                require(
                    f"cause = {v['cause']}, pc = {v['pc']}" in text,
                    "fault evidence changed",
                )
            else:
                require(
                    v["runner_exit"] == 0
                    and "FF2_SECURITY_DONE" in text
                    and "domain halted by capability fault" not in text,
                    "valid case did not complete",
                )
            evidence(run / "serial.log", f"capstone/{repetition}/case-{case}")
            result["controls"].append(
                dict(
                    arm="capstone",
                    repetition=repetition,
                    case=case,
                    expected=v["expected"],
                )
            )
            if case not in (12, 13):
                continue
            checkpoints = nodes(text)
            points = [r for r in checkpoints if r["round"]]
            require(
                [r["round"] for r in points]
                == [1] + list(range(10000, rounds + 1, 10000)),
                "missing Capstone checkpoints",
            )
            head = None
            if case == 13:
                head = struct.unpack_from(
                    "<16Q", (run / "share/result.bin").read_bytes()
                )
                require(
                    head[2] == 0 and head[4] == 128 and head[11] == 2,
                    "bad Capstone completion",
                )
                evidence(
                    run / "share/result.bin", f"capstone/{repetition}/case-13-output"
                )
            result["churn"].append(
                dict(
                    arm="capstone",
                    repetition=repetition,
                    case=case,
                    checkpoints=checkpoints,
                    metadata_carved=head[3] if head else None,
                    payload_carved=head[4] if head else None,
                )
            )
    result["capstone_binary_sha256"] = cap_binary
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.output / "checkpoints.csv").open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("arm", "repetition", "case", "round", "counter", "value"))
        for run in result["churn"]:
            counter = (
                "busy_color_ids"
                if run["arm"] == "picasso"
                else "nodes_not_on_free_list"
            )
            field = "busy" if run["arm"] == "picasso" else "not_on_free_list"
            for row in run["checkpoints"]:
                writer.writerow(
                    (
                        run["arm"],
                        run["repetition"],
                        run["case"],
                        row["round"],
                        counter,
                        row[field],
                    )
                )
    print(
        f"Validated {len(result['replay'])} native replays and {len(result['controls'])} controls"
    )


if __name__ == "__main__":
    main()
