#!/usr/bin/env python3
"""Qualify Capstone malloc/free protection, with an unprotected control.

Requires both host ELFs and their LLD maps, and checks the guest image hashes.
Fault cases must reach the test operation, then fault at its instruction with
the expected cause, or survive it and exit with the control sentinel (90).
See docs/plans/capstone-heap-protection.md for the scope of this qualification.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from datetime import date
from pathlib import Path

HOST = Path(__file__).resolve().parents[2] / "host"
sys.path.insert(0, str(HOST))
from capstone_vm.symbolize import parse as parse_fault

FAULTS = ("fault-stale", "fault-reused", "fault-bounds", "fault-bounds-large",
          "fault-double-free", "fault-double-free-reused")
COMPLETES = ("healthy", "churn", "heap-bounds", "heap-neighbour", "heap-companion")
HEAP_SYMBOLS = ("malloc", "free", "calloc", "realloc", "__libc_malloc", "__libc_free")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_symbols(elf, nm):
    output = subprocess.run([nm, "-S", "--defined-only", str(elf)],
                            text=True, capture_output=True, check=True).stdout
    table = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 4:
            table.setdefault(parts[3], []).append((int(parts[0], 16), int(parts[1], 16)))
    for name in (*HEAP_SYMBOLS, "domain_main", "capstone_heap_fault_load"):
        require(len(table.get(name, [])) == 1, f"{elf.name}: missing or duplicate {name}")
    return {name: entries[0] for name, entries in table.items() if len(entries) == 1}


def check_link_map(contents, table, wanted):
    """Match each allocation symbol's ELF address/size to its LLD input object."""
    owners = {}
    owner = None
    for line in contents.splitlines():
        match = re.fullmatch(r"\s*([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+\d+\s+(.+)", line)
        if not match:
            continue
        address, size, tail = int(match[1], 16), int(match[3], 16), match[4].strip()
        if ":(" in tail:
            owner = (tail.split(":(", 1)[0], address, size)
        elif tail.startswith(".") and not tail.startswith(".L"):
            owner = None
        if tail not in HEAP_SYMBOLS:
            continue
        require(tail not in owners, f"duplicate map symbol: {tail}")
        require(owner is not None, f"no input object for {tail}")
        obj, start, extent = owner
        require(Path(obj).name == wanted, f"{tail}: linked from {Path(obj).name}, expected {wanted}")
        require((address, size) == table[tail], f"{tail}: map does not match ELF")
        require(size > 0 and start <= address and address + size <= start + extent,
                f"{tail}: outside its input section")
        owners[tail] = {"object": wanted, "address": hex(address), "size": size}
    require(set(owners) == set(HEAP_SYMBOLS), "link map is missing allocation symbols")
    return owners


def probe_pc(elf, objdump, table, function, dest):
    start, size = table[function]
    output = subprocess.run([objdump, "-d", f"--start-address={start}",
                             f"--stop-address={start + size}", str(elf)],
                            text=True, capture_output=True, check=True).stdout
    # Fail closed if code generation no longer has the intended byte probe.
    loads = re.findall(rf"^\s*([0-9a-f]+):[^\n]*\blbu\s+{dest},\s*(?:0|0x0)\(a0\)\s*$",
                       output, re.MULTILINE)
    require(len(loads) == 1, f"{function}: expected one identifiable byte probe, got {len(loads)}")
    return int(loads[0], 16)


def check_case(label, mode, result, before, after, evidence):
    matches = re.findall(rf"^application {re.escape(mode)}: (PASS|FAIL) \(wait status=(\d+)\)$",
                         result.stdout, re.MULTILINE)
    require(len(matches) == 1, f"{label} {mode}: missing or ambiguous supervisor verdict")
    verdict, raw_status = matches[0]
    status = int(raw_status)
    control = label == "level0" and mode in FAULTS
    require(verdict == ("FAIL" if control else "PASS"), f"{label} {mode}: wrong supervisor verdict")
    require(result.returncode == (1 if control else 0), f"{label} {mode}: supervisor did not finish as expected")
    nodes = after["nodes_allocated_total"] - before["nodes_allocated_total"]
    live = {k: after[k] for k in ("live_domains", "live_regions", "live_bytes")}
    require(all(v == 0 for v in live.values()), f"{label} {mode}: resources leaked: {live}")
    require(nodes >= 0, f"{label} {mode}: allocation counter went backwards")
    if label == "sublet" and mode in ("churn", "fault-reused"):
        require(nodes >= 200000, f"{label} {mode}: reuse workload did not complete ({nodes} nodes)")
    record = {"supervisor": verdict, "wait_status": status, "nodes_allocated": nodes, "live_after": live}
    if mode not in FAULTS:
        require(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, f"{label} {mode}: abnormal completion")
        return record

    markers = re.findall(rf"^heap evidence {re.escape(mode)}: ready=(\d) survived=(\d) stderr=(\d)$",
                         result.stdout, re.MULTILINE)
    require(markers == [("1", "1" if control else "0", "1")], f"{label} {mode}: missing operation/completion evidence")
    lines = [line for line in result.stdout.splitlines() if "capstone-exec: domain fault" in line]
    if control:
        require(os.WIFEXITED(status) and os.WEXITSTATUS(status) == 90,
                f"{label} {mode}: control did not reach completion sentinel")
        require(not lines, f"{label} {mode}: control faulted")
        record["operation_survived"] = True
    else:
        require(os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGSEGV,
                f"{label} {mode}: expected SIGSEGV")
        require(len(lines) == 1, f"{label} {mode}: missing or ambiguous fault record")
        fault = parse_fault(lines[0])
        require(fault is not None and fault["sha256"] == evidence["sha256"],
                f"{label} {mode}: fault record does not identify the executed ELF")
        # This QEMU's _helper_access_with_cap reports a load bounds violation
        # as LOAD_ACCESS_FAULT (5), not the RTL's capability exception encoding.
        causes = (5,) if mode in ("fault-bounds", "fault-bounds-large") else (24, 25)
        require(fault["cause"] in causes, f"{label} {mode}: unexpected fault cause {fault['cause']}")
        link_pc = fault["pc"] - (fault["entry"] - evidence["entry"])
        site = "sh_free" if mode.startswith("fault-double-free") else "capstone_heap_fault_load"
        require(link_pc == evidence["probes"][site], f"{label} {mode}: fault at wrong instruction {link_pc:#x}")
        record["fault"] = {"cause": fault["cause"], "link_pc": hex(link_pc), "site": site}
    record["operation_reached"] = True
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--sublet-image", required=True, help="guest path of the protected image")
    parser.add_argument("--control-image", required=True, help="guest path of the unprotected image")
    parser.add_argument("--sublet-elf", type=Path, required=True)
    parser.add_argument("--control-elf", type=Path, required=True)
    parser.add_argument("--sublet-map", type=Path, help="default: <sublet-elf>.map")
    parser.add_argument("--control-map", type=Path, help="default: <control-elf>.map")
    parser.add_argument("--nm", default=os.environ.get("CAPSTONE_LLVM_NM", "llvm-nm"))
    parser.add_argument("--objdump", default=os.environ.get("CAPSTONE_LLVM_OBJDUMP", "llvm-objdump"))
    parser.add_argument("--platform", nargs="*", default=[], help="host files to hash in the report")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(HOST))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]

    def call(*words, timeout=90, verdict=False):
        result = subprocess.run([*cli, *words], text=True, capture_output=True, timeout=timeout, env=env)
        if result.returncode and not verdict:
            raise RuntimeError(f"{words[0]} exited {result.returncode}: {result.stderr}\n{result.stdout}")
        return result

    report = {"date": date.today().isoformat(), "platform": "QEMU supervised CALL, one hart",
              "verdicts": {}, "symbols": {}, "sha256": {}}
    for path in args.platform:
        report["sha256"][Path(path).name] = sha256(path)
    evidence = {}
    for label, elf, map_path in (("sublet", args.sublet_elf, args.sublet_map),
                                 ("level0", args.control_elf, args.control_map)):
        map_path = map_path or Path(str(elf) + ".map")
        table = read_symbols(elf, args.nm)
        wanted = "sublet_heap.c.obj" if label == "sublet" else "level0.c.obj"
        owners = check_link_map(map_path.read_text(), table, wanted)
        evidence[label] = {"sha256": sha256(elf), "entry": table["domain_main"][0]}
        if label == "sublet":
            evidence[label]["probes"] = {
                "capstone_heap_fault_load": probe_pc(elf, args.objdump, table, "capstone_heap_fault_load", "a0"),
                # free takes the heap lock and frees in sh_free, whose first act is the
                # stale-pointer probe: with several threads, the probe and the revocation
                # must be one step, or two frees of one pointer could both pass the probe
                "sh_free": probe_pc(elf, args.objdump, table, "sh_free", "zero")}
        report["symbols"][label] = {"entry_points": owners, "sha256": evidence[label]["sha256"],
                                    "map_sha256": sha256(map_path)}
        print(f"{label}: ELF/map agree on allocation entry points from {wanted}: PASS", flush=True)

    boot_id = call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout.strip()
    call("exec", "sh", "-c", "cp /mnt/host/application-supervisor /tmp/application-supervisor && "
         "chmod +x /tmp/application-supervisor")
    for label, image in (("sublet", args.sublet_image), ("level0", args.control_image)):
        guest_hash = call("exec", "sha256sum", image).stdout.split()[0]
        require(guest_hash == evidence[label]["sha256"], f"{label}: guest image differs from checked ELF")
        report["symbols"][label]["guest_sha256"] = guest_hash
        result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec", image, timeout=180, verdict=True)
        require(result.returncode == 0 and "application sequence: PASS" in result.stdout,
                f"{label}: application sequence failed: {result.stdout}\n{result.stderr}")
        report["verdicts"][label] = {"sequence": "PASS"}
        print(f"{label}: application sequence: PASS", flush=True)
        for mode in FAULTS + COMPLETES:
            before = json.loads(call("exec", "capstone-exec", "--stats").stdout)
            result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec",
                          image, "--mode", mode, timeout=200, verdict=True)
            after = json.loads(call("exec", "capstone-exec", "--stats").stdout)
            try:
                record = check_case(label, mode, result, before, after, evidence[label])
            except ValueError:
                print(result.stdout, file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                raise
            report["verdicts"][label][mode] = record
            print(f"{label} {mode}: checked operation, completion and cleanup: PASS", flush=True)
    require(call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout.strip() == boot_id, "VM rebooted during qualification")
    report["boot_id"] = boot_id
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print("heap qualification: PASS")


if __name__ == "__main__":
    main()
