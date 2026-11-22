"""readings.py <run> ...: per twin run of perf/mrhwm/out, the sums over every `@@NATIVE-CNT <n> end` line of the
serial log (k1+k3+k4 and service_other), the invocation count, and the summary's specdiff and @@MEM-PEAK
(used_kB, VmHWM kB). A run with no serial log or no end line is reported and makes the exit status 2."""
import json, os, re, sys
O = os.environ.get("MRHWM_OUT", os.path.join(os.environ.get("CAPSTONE_WORK", f"/mnt/hdd/{os.environ.get('USER', '')}/capstone"), "perf/mrhwm/out"))
rc = 0
for n in sys.argv[1:]:
    p = f"{O}/{n}/vm/serial.log"
    ends = re.findall(r"^@@NATIVE-CNT \d+ end (.*)$", open(p, errors="replace").read(), re.M) if os.path.exists(p) else []
    if not ends:
        print(f"{n}: NO DATA ({p})"); rc = 2; continue
    s = {}
    for e in ends:
        for k, v in re.findall(r"(\w+)=(\d+)", e): s[k] = s.get(k, 0) + int(v)
    j = json.loads(open(f"{O}/{n}.log").readline())
    v = open(f"{O}/{n}.validate").read().strip()
    print(f"{n:18s} invocations {len(ends):2d} k1+k3+k4 {s['k1'] + s['k3'] + s['k4']:7d} service_other {s['service_other']:6d} "
          f"mem_peak {j.get('mem_peak')} oom {j.get('oom')} {v}")
sys.exit(rc)
