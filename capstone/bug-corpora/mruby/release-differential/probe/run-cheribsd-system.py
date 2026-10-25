#!/usr/bin/env python3
"""The system-allocator cases on stock CheriBSD, with where each fault lies and what the
quarantine held. One boot, revocation at the platform default.

    run-cheribsd-system.py OUT --cheri CHERI_OUTPUT --bins DIR --cases N [N ...]

CHERI_OUTPUT holds sdk/, rootfs-riscv64-purecap/ and cheribsd-riscv64-purecap.img; DIR is
probe/build-cheribsd-extra.sh's output plus the interpreter it relinked (DIR/mruby, which must
equal DIR/mruby-relink). Each case runs twice:

  plain    the interpreter (or the case's C-API driver) as the 2026-10-06 arm ran it: exit
           status, the harness's lines, and the kernel's log line for a CHERI exception, whose
           pc the image's own symbols resolve to a function
  probed   the same with the quarantine probe wrapped around the allocator: frees, how many
           were quarantined, how many allocations came back still quarantined, and how many
           sweeps completed. The probe changes sizes and addresses, so a probed run scores
           nothing; it reads the state of the mechanism

Controls, in the same boot and before any case: the sysctls read back (any value but the
default aborts the run), the interpreter evaluates, 40- and 500-frame recursion, the platform's
revocation control (a stale read after a forced sweep must take SIGPROT at mc_defect_read), and
the probe's sweep counter both ways (a churn of allocations must advance it, a quiet run must
not). Writes OUT/cheribsd.json and prints one line per run.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/ports/common/host/cheribsd"))
from guest import Guest  # noqa: E402

CHURN = 'a = nil; 200_000.times { |i| a = "x" * 64 + i.to_s }; puts "CHURN done"\n'
# The 2026-10-06 arm's own depth controls, verbatim. FMT_DEPTH is not a control: the same
# recursion printed through String#% took SIGPROT here at 40 frames on 2026-10-11, so it runs
# after the controls as an observation, with its fault located like a case's.
DEPTH = 'def deep(n) = n == 0 ? 0 : 1 + deep(n - 1)\nputs "DEEP {n} #{{deep({n})}}"\n'
FMT_DEPTH = 'def d(n) n == 0 ? 0 : 1 + d(n - 1) end; puts "DEEP %d %d" % [40, d(40)]\n'


def symbolize(nm, image, pc):
    out = subprocess.run([str(nm), "-n", "--defined-only", str(image)], capture_output=True, text=True).stdout
    best = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1].lower() in "tw":
            a = int(parts[0], 16)
            if a <= pc:
                best = (a, parts[2])
    return f"{best[1]}+{pc - best[0]:#x}" if best else "?"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--cheri", type=Path, required=True)
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--port", type=int, default=10093)
    p.add_argument("--nm", type=Path, help="llvm-nm for the symbols (default: the SDK's)")
    a = p.parse_args()
    out = a.output / time.strftime("run-%H%M%S")
    out.mkdir(parents=True)
    if (a.bins / "mruby").read_bytes() != (a.bins / "mruby-relink").read_bytes():
        sys.exit("DIR/mruby is not the relinked interpreter")
    nm = a.nm or a.cheri / "sdk/bin/llvm-nm"
    cases = []
    for n in a.cases:
        d = next(CORPUS.glob(f"{int(n):02d}_*"))
        claims = json.loads((d / "case.json").read_text())
        cases.append((d.name, claims["trigger"], d, claims.get("fault_sites", [])))
    tar = out / "kit.tar.gz"   # static images with debug info: 121 MB raw, past scp's 120 s in the guest
    with tarfile.open(tar, "w:gz") as t:
        for f in ("mruby", "mruby-qprobe", "revocation-control"):
            t.add(a.bins / f, arcname=f)
        for name, trig, d, _ in cases:
            if trig.endswith(".c"):
                for f in (f"capi-{name[:2]}", f"capi-{name[:2]}-qprobe"):
                    t.add(a.bins / f, arcname=f)
            else:
                t.add(d / trig, arcname=f"cases/{name[:2]}.rb")
        for f, text in (("churn.rb", CHURN), ("d40.rb", DEPTH.format(n=40)), ("d500.rb", DEPTH.format(n=500)),
                        ("fmt-d40.rb", FMT_DEPTH)):
            (out / f).write_text(text)
            t.add(out / f, arcname=f)

    g = Guest(a.cheri / "sdk", a.cheri / "rootfs-riscv64-purecap", a.cheri / "cheribsd-riscv64-purecap.img",
              out, a.port)
    rec = {"runs": [], "controls": {}}
    print("BOOT", flush=True)
    g.start()
    try:
        sh = lambda c, t=180: g.ssh(c, timeout=t).stdout
        sysctls = {k: sh(f"sysctl -n {k} 2>/dev/null || echo absent").strip() for k in (
            "security.cheri.runtime_revocation_default", "security.cheri.runtime_revocation_async",
            "security.cheri.runtime_revocation_every_free_default")}
        rec["sysctls"] = sysctls
        print("sysctls", sysctls, flush=True)
        if sysctls["security.cheri.runtime_revocation_default"] != "1" or \
           sysctls["security.cheri.runtime_revocation_async"] != "1":
            raise SystemExit("revocation is not at the platform default; refusing to score")
        knobs = [l.split(":")[0] for l in sh("sysctl -a 2>/dev/null | grep -i 'log_user_cheri'").splitlines() if ":" in l]
        for k in knobs:
            sh(f"sysctl {k}=1")
        rec["exception_logging"] = {k: sh(f"sysctl -n {k}").strip() for k in knobs}
        print("exception logging", rec["exception_logging"], flush=True)
        print(f"copying {tar.stat().st_size >> 20} MiB", flush=True)
        rec["uname"] = sh("uname -rm").strip()
        sh("rm -rf /root/s && mkdir -p /root/s")
        subprocess.run(["scp", "-O", *g.ssh_options, "-P", str(a.port), str(tar), "root@127.0.0.1:/root/s/kit.tar.gz"],
                       check=True, capture_output=True, timeout=1800)
        sh("cd /root/s && tar xzf kit.tar.gz && chmod +x mruby* capi-* revocation-control 2>/dev/null; true")

        def run(label, cmd, image, timeout=60):
            sh("dmesg -c > /dev/null 2>&1 || true")
            r = sh(f"cd /root/s && (timeout {timeout} {cmd} > /tmp/o 2>&1; echo RC=$? >> /tmp/o); cat /tmp/o | head -c 6000",
                   t=timeout + 60)
            log = sh("dmesg 2>/dev/null | tail -20")
            m = re.search(r"RC=(\d+)", r)
            rc = int(m.group(1)) if m else -1
            lines = [l for l in r.splitlines() if not l.startswith("RC=")]
            pcs = [int(x, 16) for x in re.findall(r"(?:pc|sepc|PC)[ =:]+(?:0x)?([0-9a-fA-F]{4,16})", log)]
            fault = [symbolize(nm, a.bins / image, pc) for pc in pcs[:2]]
            row = {"label": label, "cmd": cmd, "rc": rc,
                   "harness": next((l for l in lines if l.startswith("[")), "")[:400],
                   "case_lines": [l[:200] for l in lines if l.startswith(("CASE", "REVOCATION", "DEEP", "CHURN", "QUARANTINE"))][:8],
                   "first_line": (lines[0] if lines else "")[:200],
                   "kernel_log": [l for l in log.splitlines() if "cheri" in l.lower() or "pid" in l.lower()][-4:],
                   "fault_function": fault}
            rec["runs"].append(row)
            print(f"  {label:34} rc={rc:<4} {fault} {row['harness'][:60] or row['case_lines'][:2]}", flush=True)
            return row

        ev = run("control:eval", "./mruby -e 'puts 6*7'", "mruby")
        ok = "42" in ev["first_line"]
        for dd in ("d40", "d500"):
            r = run(f"control:depth-{dd}", f"./mruby {dd}.rb", "mruby")
            ok = ok and any(l.startswith("DEEP") for l in r["case_lines"])
        rc_ = run("control:revocation", "./revocation-control", "revocation-control")
        rec["controls"]["revocation_faulted"] = rc_["rc"] == 162
        ok = ok and rc_["rc"] == 162
        sw = run("control:sweep (probed churn)", "./mruby-qprobe churn.rb", "mruby-qprobe", timeout=600)
        ns = run("control:no-sweep (probed eval)", "./mruby-qprobe -e 'puts 6*7'", "mruby-qprobe")
        rec["controls"]["ok"] = ok
        if not ok:
            raise SystemExit("a control failed; refusing to score")
        run("observation:fmt-depth40", "./mruby fmt-d40.rb", "mruby")
        for name, trig, d, sites in cases:
            nn = name[:2]
            if trig.endswith(".c"):
                plain, probed, img = f"./capi-{nn}", f"./capi-{nn}-qprobe", f"capi-{nn}"
            else:
                plain, probed, img = f"./mruby cases/{nn}.rb", f"./mruby-qprobe cases/{nn}.rb", "mruby"
            r = run(f"{name[:30]} plain", plain, img)
            r["case"], r["fault_sites"] = name, sites
            r = run(f"{name[:30]} probed", probed, img + "-qprobe")
            r["case"] = name
    finally:
        (out / "cheribsd.json").write_text(json.dumps(rec, indent=1) + "\n")
        g.close()
    print(f"WROTE {out / 'cheribsd.json'}")


if __name__ == "__main__":
    main()
