#!/usr/bin/env python3
"""One binary, one run, two instruments: the stock cache is ON, so ASan can only report a
use-after-free on a block that actually reached libc free(). The probe says, in the same run
and for the same address, whether that block was parked or passed through. Each confirms the
other; neither is inferred across builds."""
import collections, json, os, re, subprocess, sys
from pathlib import Path
B = "/tmp/capstone/php-both/php-5.0.0/sapi/cli/php"
C = Path("/home/zephyr/php-corpus-wt/capstone/bug-corpora/php/engine-repros")
OUT = Path("/tmp/capstone/both-run"); (OUT/"raw").mkdir(parents=True, exist_ok=True)
REGION = re.compile(r"is located \d+ bytes inside of (\d+)-byte region \[(0x[0-9a-f]+),")
HDR = 24
res = {}
for d in sorted(x for x in C.iterdir() if (x/"trigger.php").is_file()):
    raw = OUT/"raw"/f"{d.name}.tsv"
    if raw.exists(): raw.unlink()
    env = dict(PATH="/usr/bin:/bin", ASAN_OPTIONS="detect_leaks=0:abort_on_error=0",
               ZPROBE_OUT=str(raw))
    try:
        p = subprocess.run([B, "-n", "trigger.php"], cwd=str(d), env=env,
                           capture_output=True, timeout=180)
        rc, err = p.returncode, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        rc, err = -999, "TIMEOUT"
    (OUT/"raw"/f"{d.name}.asan").write_text(err)
    kind = (re.search(r"AddressSanitizer: (\S+)", err) or [None, None])[1] if "AddressSanitizer" in err else None
    m = REGION.search(err)
    payload = int(m.group(2), 16) + HDR if m else None
    probe = []
    tot = collections.Counter()
    if raw.exists():
        for line in raw.read_text(errors="replace").splitlines():
            f = line.split("\t")
            if f[0] == "ZP" and len(f) == 7:
                tot[f[1]] += 1
                if payload is not None and int(f[5], 16) == payload:
                    probe.append((f[1], int(f[2])))
    res[d.name] = {"rc": rc, "asan": kind, "payload": hex(payload) if payload else None,
                   "probe_at_payload": probe, "probe_totals": dict(tot)}
    print(f"  {d.name:<44} rc={rc:<5} {str(kind):<22} probe@addr={probe}", flush=True)
(OUT/"both.json").write_text(json.dumps(res, indent=1))
print(f"\nwritten {OUT/'both.json'}")
