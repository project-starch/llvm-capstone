#!/usr/bin/env python3
"""Decide one run.sh attempt from its evidence and print result.json.

PASS needs all of: runcpu exited 0; SPEC's log reports "Success: 1x<bench>" and no error for it;
the invocation records exist and none was refused; the benchmark's own executable ran in the
guest (route "guest") and every guest invocation exited 0 without a domain fault.
usage: verdict.py OUT BENCH SIZE RUNCPU_RC START END [LABEL]
Also records, without judging them: SPEC's reported seconds, the launcher's CAPSTONE_VM_STATS counters
of every guest invocation (from the run directory's .err files) and the QEMU process's usage."""
import glob, json, os, re, sys

out, bench, size, rc, start, end = sys.argv[1:7]
label = sys.argv[7] if len(sys.argv) > 7 else "capstone-virtual"
reasons = []
log = open(os.path.join(out, "runcpu.log"), errors="replace").read()
spec_logs = sorted(glob.glob(os.path.join(out, "spec", "CPU2017.*.log")))
spec_log = open(spec_logs[-1], errors="replace").read() if spec_logs else ""
if int(rc):
    reasons.append(f"runcpu exited {rc}")
success = re.search(rf"^Success: 1x{re.escape(bench)}\b", log + spec_log, re.M)
error = re.search(rf"^Error: .*{re.escape(bench)}\b", log + spec_log, re.M)
if not success:
    reasons.append("SPEC reports no success")
if error:
    reasons.append(f"SPEC reports {error.group(0).strip()!r}")
records = []
for f in sorted(glob.glob(os.path.join(out, "invocations", "[0-9][0-9][0-9][0-9].json"))):
    records.append(json.load(open(f)))
routes = {}
for r in records:
    routes[r["route"]] = routes.get(r["route"], 0) + 1
if not records:
    reasons.append("no invocation records")
if routes.get("refused"):
    reasons.append(f"{routes['refused']} invocation(s) refused")
# Each benchmark's executable name (its Spec/object.pm $exename) and the build suffix.
EXE = {"500.perlbench_r": "perlbench_r", "502.gcc_r": "cpugcc_r", "505.mcf_r": "mcf_r",
       "519.lbm_r": "lbm_r", "525.x264_r": "x264_r", "538.imagick_r": "imagick_r",
       "544.nab_r": "nab_r", "557.xz_r": "xz_r", "997.specrand_fr": "specrand_fr",
       "999.specrand_ir": "specrand_ir"}
guest = [r for r in records if r["route"] == "guest"]
mine = f"{EXE[bench]}_base.{label}"
if not any(os.path.basename(r["exe"]) == mine for r in guest):
    reasons.append(f"{mine} never ran in the guest")
for r in guest:
    g = (r.get("detail") or {}).get("guest") or {}
    if r["status"] != 0 or g.get("kind") != "exit" or g.get("value") != 0 or g.get("fault"):
        reasons.append(f"guest invocation {os.path.basename(r['exe'])} ended {g or r['status']}")
stub = os.path.join(out, "perl-pack-stub.jsonl")
# Measurements, recorded as read.
reported = None
for rsf in glob.glob(os.path.join(out, "spec", "CPU2017.*.rsf")):
    for line in open(rsf, errors="replace"):
        m = re.match(rf"spec\.cpu2017\.results\.{re.escape(bench.replace('.', '_'))}\.base\.\d+\.reported_time: ([0-9.]+)", line)
        if m: reported = float(m.group(1))
stats = []
for err in sorted(glob.glob(os.path.join(out, "spec", "*.err"))):
    for line in open(err, errors="replace"):
        if line.startswith("CAPSTONE_VM_STATS "):
            rec = {"file": os.path.basename(err)}
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                rec[k] = int(v) if v.isdigit() else v
            stats.append(rec)
usage_path = os.path.join(out, "qemu-usage.json")
usage = json.load(open(usage_path)) if os.path.exists(usage_path) else None
qemu_cpu = None
if usage and "before" in usage and "after" in usage and "utime_ticks" in usage["after"] and "utime_ticks" in usage["before"]:
    qemu_cpu = ((usage["after"]["utime_ticks"] + usage["after"]["stime_ticks"]) -
                (usage["before"]["utime_ticks"] + usage["before"]["stime_ticks"])) / usage["clk_tck"]
result = {
    "benchmark": bench, "size": size, "label": label, "verdict": "PASS" if not reasons else "FAIL",
    "reported_seconds": reported, "qemu_cpu_seconds": qemu_cpu,
    "qemu_vmhwm_kb": int(usage["after"]["VmHWM"].split()[0]) if usage and "VmHWM" in usage.get("after", {}) else None,
    "vm_stats": stats,
    "reasons": reasons, "start": start, "end": end, "routes": routes,
    "guest_invocations": len(guest),
    "spec_reports": sorted(os.path.basename(p) for p in glob.glob(os.path.join(out, "spec", "CPU2017.*"))),
    "perl_pack_stub": [json.loads(l) for l in open(stub)] if os.path.exists(stub) else [],
}
print(json.dumps(result, indent=2))
