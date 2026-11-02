#!/usr/bin/env python3
"""ASan as a LOCATOR, not a classifier: run every trigger on the detector build (cache
disabled, so every efree reaches free() and ASan can see it) and record WHERE the block the
bug later uses was released -- frame #1 of the `freed by` stack above _efree, which is exactly
the return address the probe logs. The classification still comes from the probe."""
import json, re, subprocess, sys
from pathlib import Path
B = "/tmp/capstone/php-asan/php-5.0.0/sapi/cli/php"
C = Path("/home/zephyr/php-corpus-wt/capstone/bug-corpora/php/engine-repros")
OUT = Path("/tmp/capstone/asan-locator"); OUT.mkdir(parents=True, exist_ok=True)
FRAME = re.compile(r"^\s*#(\d+) 0x[0-9a-f]+ in (\S+) (\S+?):(\d+)")
cases = sorted(d for d in C.iterdir() if (d / "trigger.php").is_file())
res = {}
for d in cases:
    try:
        p = subprocess.run([B, "-n", "trigger.php"], cwd=str(d), capture_output=True, timeout=120,
                           env={"PATH": "/usr/bin:/bin",
                                "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=0"})
        err = p.stderr.decode("utf-8", "replace"); rc = p.returncode
    except subprocess.TimeoutExpired:
        err, rc = "TIMEOUT", -999
    (OUT / f"{d.name}.txt").write_text(err)
    kind = None
    m = re.search(r"ERROR: AddressSanitizer: (\S+)", err)
    if m:
        kind = m.group(1)
    # the `freed by` stack: skip the interceptor and _efree itself, take the first caller
    site, chain = None, []
    seg = err.split("freed by thread")
    if len(seg) > 1:
        for line in seg[1].splitlines():
            fm = FRAME.match(line)
            if not fm:
                if chain:
                    break
                continue
            _, func, f, ln = fm.groups()
            f = f.replace("/tmp/capstone/php-asan/php-5.0.0/", "")
            chain.append(f"{func} ({f}:{ln})")
            if func not in ("__interceptor_free", "free", "_efree", "_efree_rel"):
                if site is None:
                    site = f"{f}:{ln}"
    res[d.name] = {"rc": rc, "asan": kind, "free_site": site, "free_chain": chain[:5]}
    print(f"  {d.name:<44} rc={rc:<5} {str(kind):<24} free@ {site}", flush=True)
(OUT / "locator.json").write_text(json.dumps(res, indent=1))
got = sum(1 for v in res.values() if v["free_site"])
print(f"\n{got} of {len(res)} cases produced a free site")
