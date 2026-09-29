#!/usr/bin/env python3
"""Measure PR #94 recovery on musl and the deployed SQLite amalgamation.

Source capstone-test-env.sh first, then run with --musl-dir (prepared musl
1.2.5), --sqlite-source (the port's generated sqlite3-capstone.c), and --out.
The compiler paths come from CAPSTONE_CLANG and CAPSTONE_LLVM_BIN.

Each C unit is optimized once to IR. The same IR then passes through llc
with recovery on and off. Count inttoptr occurrences in the embedded IR
immediately after the pass, and compare final assembly. These are static
coverage measurements, not execution or speed measurements. Total inttoptr
occurrences include conversions that are not recoverable round trips.
Artifacts and diagnostics stay in --out; summary.json contains result lines.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT).returncode


def ir_functions(text):
    # llc's stop-after output wraps IR in YAML; its IR block ends before '...'.
    text = text.split("\n...", 1)[0]
    functions = {}
    current = "<constants>"
    functions[current] = []
    for line in text.splitlines():
        m = re.match(r"\s*define .*?@([^ (]+)\(", line)
        if m:
            current = m.group(1)
            functions[current] = []
        functions[current].append(line)
    return {name: len(re.findall(r"\binttoptr\b", "\n".join(lines)))
            for name, lines in functions.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--musl-dir", type=Path, required=True)
    ap.add_argument("--sqlite-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    clang = Path(os.environ["CAPSTONE_CLANG"])
    llc = Path(os.environ["CAPSTONE_LLVM_BIN"]) / "llc"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "musl_survey", repo / "capstone/ports/musl-capstone/survey-musl-capstone.py")
    musl_survey = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(musl_survey)
    musl = args.musl_dir.resolve()
    for required in (musl / "obj/include/bits/alltypes.h", args.sqlite_source, clang, llc):
        if not required.is_file():
            raise SystemExit(f"Missing input: {required}")
    foreign = set(os.listdir(musl / "arch")) - musl_survey.FOREIGN_ARCH_KEEP
    sources = sorted(p for p in (musl / "src").rglob("*.c")
                     if not set(p.relative_to(musl).parts) & foreign)
    if not sources:
        raise SystemExit("Empty musl corpus")
    jobs = [("musl/" + str(p.relative_to(musl)), p,
             musl_survey.compile_flags(musl), "O1", False) for p in sources]
    sqlite = repo / "capstone/ports/sqlite"
    build = (sqlite / "build-sqlite-capstone.sh").read_text()
    block = re.search(r"^SQLITE_DEFINES=\(\n(.*?)^\)", build, re.M | re.S).group(1)
    defines = [line.strip() for line in block.splitlines() if line.strip().startswith("-D")]
    sqlite_flags = ["-target", "capstone64-unknown-elf", "-Xclang", "-target-feature",
                    "-Xclang", "+m", "-ffreestanding", "-fno-builtin",
                    "-ffunction-sections", "-fdata-sections", "-w",
                    "-include", str(sqlite / "adapted/capstone_sqlite_libc.h")]
    sqlite_flags += ["-I" + str(p) for p in [sqlite / "adapted/stubinc", sqlite / "adapted",
                    sqlite, repo / "capstone/tests/runtime-qemu/sqlite-vfs-skeleton",
                    args.sqlite_source.resolve().parent]] + defines
    for opt in ("O0", "O2"):
        jobs.append(("sqlite/" + opt, args.sqlite_source.resolve(), sqlite_flags + ["-" + opt], opt, False))
    control = repo / "llvm/test/CodeGen/Capstone/recover-provenance.ll"
    for opt in ("O0", "O2"):
        jobs.append(("control/" + opt, control, [], opt, True))

    def measure(job):
        name, source, flags, opt, is_ir = job
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        ir = source if is_ir else dest / "input.ll"
        row = {"source": name, "source_sha256": sha(source), "optimization": opt}
        if not is_ir and run([clang, *flags, "-S", "-emit-llvm", source, "-o", ir], dest / "frontend.log"):
            row["status"] = "frontend-failed"
            return row
        counts, assembly, status = {}, {}, {}
        for arm in ("on", "off"):
            common = [llc, "-mtriple=capstone64", "-mattr=+m,+a", "-" + opt,
                      "-capstone-recover-provenance=" + ("true" if arm == "on" else "false"), ir]
            mir = dest / (arm + ".mir")
            rc = run([*common, "-stop-after=capstone-provenance", "-o", mir], dest / (arm + ".ir.log"))
            if rc:
                row["status"] = arm + "-ir-failed"
                return row
            counts[arm] = ir_functions(mir.read_text())
            asm = dest / (arm + ".s")
            status[arm] = run([*common, "-o", asm], dest / (arm + ".asm.log"))
            if not status[arm]:
                assembly[arm] = asm.read_text()
        row["inttoptr_off"] = sum(counts["off"].values())
        row["inttoptr_on"] = sum(counts["on"].values())
        row["recovered"] = row["inttoptr_off"] - row["inttoptr_on"]
        row["recovered_functions"] = {fn: n - counts["on"].get(fn, 0)
            for fn, n in counts["off"].items() if n != counts["on"].get(fn, 0)}
        row["backend_status"] = status
        if any(status.values()):
            row["status"] = "backend-failed"
        else:
            row["status"] = "ok"
            row["assembly_equal"] = assembly["on"] == assembly["off"]
            row["cincoffset"] = {arm: len(re.findall(r"^\s+cincoffset(?:imm)?\s", text, re.M))
                                  for arm, text in assembly.items()}
        return row

    compiler_hashes = {"clang": sha(clang), "llc": sha(llc)}
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(measure, jobs):
            rows.append(row)
            if row.get("recovered") or row["status"] != "ok":
                print(json.dumps(row), flush=True)
    controls = [r for r in rows if r["source"].startswith("control/")]
    control_ok = len(controls) == 2 and all(r["status"] == "ok" and r["recovered"] > 0
        and not r["assembly_equal"] for r in controls)
    by_name = {r["source"]: r for r in rows}
    control_ok &= by_name["musl/" + musl_survey.CONTROL_MUST_PASS]["status"] == "ok"
    control_ok &= by_name["musl/" + musl_survey.CONTROL_MUST_FAIL]["status"] == "frontend-failed"
    groups = {}
    for group in ("musl", "sqlite", "control"):
        subset = [r for r in rows if r["source"].startswith(group + "/")]
        measured = [r for r in subset if "recovered" in r]
        groups[group] = {"units": len(subset), "compiled": sum(r["status"] == "ok" for r in subset),
            "inttoptr_off": sum(r["inttoptr_off"] for r in measured),
            "inttoptr_on": sum(r["inttoptr_on"] for r in measured),
            "recovered": sum(r["recovered"] for r in measured),
            "units_with_recovery": sum(r["recovered"] > 0 for r in measured),
            "units_with_changed_assembly": sum(r.get("assembly_equal") is False for r in subset)}
    if compiler_hashes != {"clang": sha(clang), "llc": sha(llc)}:
        raise SystemExit("Compiler changed during the measurement")
    result = {"compiler_sha256": compiler_hashes["clang"], "llc_sha256": compiler_hashes["llc"],
              "compiler_version": subprocess.check_output([clang, "--version"], text=True).splitlines()[0],
              "controls_pass": control_ok, "groups": groups, "units": rows}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"controls_pass": control_ok, "groups": groups}, indent=2))
    unexpected = any("-ir-failed" in r["status"] or
        (r["status"] == "backend-failed" and bool(r["backend_status"]["on"]) != bool(r["backend_status"]["off"]))
        for r in rows)
    return 0 if control_ok and not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
