#!/usr/bin/env python3
"""Random-program differential testing of the Capstone compiler (Tier 2c).

    run-fuzz-campaign.py --gen csmith --count N [--start S] [--levels O0,O2] --out DIR
                         [--tsv results.tsv] [--no-qemu] [--control]

Per program: generate; compile and run it NATIVELY at -O0 and -O2 (the reference;
a disagreement between the two native runs means generator UB and the seed is
discarded as GEN-UB, a native timeout as GEN-SLOW); build a Capstone domain at
each requested level (build-fuzz-program.sh, which reports BUILD-ERROR /
BUILD-CRASH / LINK-ERROR / SIZE-SKIP); then run every domain in ONE QEMU boot
(run-domain-batch.py) and compare each returned checksum with the native one:

  MATCH      the domain returned the native checksum
  MISMATCH   it returned a different value -- a miscompile candidate
  WEDGE      the guest hung on it; NOT a compiler verdict until the same image wedges as the
             FIRST domain of a fresh boot (2026-09-05: a nondeterministic per-boot wedge hit
             any image as early as the fifth domain; F-04 was retracted on that)
  FAULT      the domain faulted (QEMU aborted and was rebooted)
  TIMEOUT    the domain did not return
  ERROR      the loader could not run it

Every row names the generator, seed, level, native checksum, domain value, domain
sha256 and QEMU binary.  --control appends the positive controls: the first seed's
FUZZ_XOR build, which MUST read MISMATCH, and fault_domain (a deliberate capability
fault) placed BEFORE the last item, which MUST read FAULT while the item after it
still runs.  Exit 0 when no MISMATCH/FAULT/TIMEOUT/BUILD-CRASH and the controls
behaved; 1 otherwise; 2 when nothing ran.
"""
import argparse
import hashlib
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOCK = HERE / "TOOLS.lock"
CSMITH_FLAGS = ["--no-argc", "--no-float", "--no-unions", "--no-packed-struct", "--max-array-dim", "2", "--max-funcs", "6"]


def lock_value(key):
    for l in open(LOCK):
        if l.startswith(key + "="):
            # TOOLS.lock keeps paths relative to a literal $HOME so it carries no user name.
            return os.path.expandvars(l.split("=", 1)[1].strip())
    raise SystemExit(f"{LOCK} lacks {key}; run setup-fuzz-tools.sh")


def sh(cmd, timeout=None, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return None, "", "timeout"


def native_checksum(src, level, inc, workdir, seed):
    exe = workdir / f"native-{seed}-{level}"
    rc, _, err = sh(["cc", f"-{level}", "-w", "-I", inc, "-o", str(exe), str(src)], timeout=120)
    if rc != 0:
        return None, "native compile failed: " + err.strip().splitlines()[0][:160] if err.strip() else "native compile failed"
    rc, out, _ = sh([str(exe)], timeout=2)
    if rc is None:
        return None, "native timeout"
    m = re.search(r"checksum = ([0-9A-Fa-f]+)", out)
    if not m:
        return None, "no checksum line"
    return int(m.group(1), 16), ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", default="csmith", choices=["csmith"])
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--levels", default="O0,O2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tsv", default=str(HERE / "results" / (time.strftime("%Y-%m-%d") + ".tsv")))
    ap.add_argument("--no-qemu", action="store_true", help="generate, reference and build only")
    ap.add_argument("--control", action="store_true", help="append the XOR and fault positive controls")
    ap.add_argument("--per-item-timeout", type=float, default=90.0)
    a = ap.parse_args()
    levels = [l.strip() for l in a.levels.split(",") if l.strip()]
    out = pathlib.Path(a.out).resolve()
    (out / "src").mkdir(parents=True, exist_ok=True)
    (out / "doms").mkdir(exist_ok=True)
    (out / "findings").mkdir(exist_ok=True)
    csmith = lock_value("csmith_bin")
    inc = lock_value("csmith_include")
    qemu = os.environ.get("CAPSTONE_QEMU_BINARY", "")
    qid = "qemu=?"
    if qemu and os.path.exists(qemu):
        qid = f"qemu={hashlib.sha256(open(qemu, 'rb').read()).hexdigest()[:12]}@{time.strftime('%Y-%m-%dT%H:%M', time.localtime(os.path.getmtime(qemu)))}"
    build = str(HERE / "build-fuzz-program.sh")

    rows = []          # (gen, seed, level, verdict, native, value, dom_sha, note)
    manifest = []      # (name, path, seed, level, native, kind)
    for seed in range(a.start, a.start + a.count):
        src = out / "src" / f"cs{seed}.c"
        flags = CSMITH_FLAGS + (["--builtins"] if seed % 4 == 0 else [])
        rc, _, err = sh([csmith, "--seed", str(seed)] + flags + ["-o", str(src)], timeout=120)
        if rc != 0:
            rows.append(("csmith", seed, "*", "GEN-ERROR", "", "", "", err.strip()[:160])); continue
        n0, why0 = native_checksum(src, "O0", inc, out / "src", seed)
        n2, why2 = native_checksum(src, "O2", inc, out / "src", seed)
        if n0 is None or n2 is None:
            rows.append(("csmith", seed, "*", "GEN-SLOW" if "timeout" in (why0 + why2) else "GEN-ERROR", "", "", "", (why0 or why2)[:160])); continue
        if n0 != n2:
            rows.append(("csmith", seed, "*", "GEN-UB", f"{n0:X}", f"{n2:X}", "", "native -O0 and -O2 disagree")); continue
        for lvl in levels:
            dom = out / "doms" / f"cs{seed}-{lvl}.dom"
            rc, o, e = sh(["bash", build, str(src), str(dom), f"-{lvl}"], timeout=600)
            line = next((l for l in (o + e).splitlines() if re.match(r"(BUILT|BUILD-|LINK-|SIZE-)", l)), (o + e).strip()[:200])
            if rc != 0:
                verdict = {10: "BUILD-ERROR", 11: "BUILD-CRASH", 12: "LINK-ERROR", 13: "SIZE-SKIP"}.get(rc, "BUILD-ERROR")
                rows.append(("csmith", seed, lvl, verdict, f"{n0:X}", "", "", line[:200]))
                if verdict == "BUILD-CRASH":
                    keep_finding(out, seed, lvl, src, None, line)
                continue
            sha = hashlib.sha256(open(dom, "rb").read()).hexdigest()[:12]
            manifest.append((f"cs{seed}-{lvl}", str(dom), seed, lvl, n0, sha))
        if a.control and seed == a.start and manifest:
            dom = out / "doms" / f"cs{seed}-{levels[-1]}-xor.dom"
            rc, o, e = sh(["bash", build, str(src), str(dom), f"-{levels[-1]}", "xor"], timeout=600)
            if rc != 0:
                # A positive control that does not build is a harness error, never a skip.
                print("ERROR: the XOR control did not build:\n" + (o + e)[-1500:], file=sys.stderr)
                sys.exit(2)
            manifest.append((f"cs{seed}-{levels[-1]}-xor", str(dom), seed, levels[-1] + "-xor", n0,
                             hashlib.sha256(open(dom, "rb").read()).hexdigest()[:12]))

    if a.control and manifest:
        fdom = out / "doms" / "fault.dom"
        # "bare": fault_domain.c defines domain_main itself; linking the csmith entry as
        # well was a duplicate-symbol link error that the first campaign skipped SILENTLY
        # (2026-09-05), so the run had one control instead of two.  Never skip a control.
        rc, o, e = sh(["bash", build, str(HERE / "csmith-rt" / "fault_domain.c"), str(fdom), "-O0", "bare"], timeout=300)
        if rc != 0:
            print("ERROR: the fault control did not build:\n" + (o + e)[-1500:], file=sys.stderr)
            sys.exit(2)
        # Before the LAST item, so the run must survive the fault and still run one more.
        manifest.insert(len(manifest) - 1, ("fault-control", str(fdom), 0, "control", -1,
                                            hashlib.sha256(open(fdom, "rb").read()).hexdigest()[:12]))

    if manifest and not a.no_qemu:
        mpath = out / "manifest.tsv"
        with open(mpath, "w") as f:
            for name, path, *_ in manifest:
                f.write(f"{name}\t{path}\n")
        share = out / "share"
        res = out / "batch-results.tsv"
        # One lock for the machine, independent of CAPSTONE_TMP_ROOT (see run-twin-suite.sh).
        lock = pathlib.Path(os.environ.get("CAPSTONE_QEMU_LOCK", "/tmp/capstone/nightly-qemu.lock"))
        cmd = ["python3", str(HERE / "run-domain-batch.py"), "--manifest", str(mpath),
               "--share", str(share), "--log", str(out / "batch.log"), "--out", str(res),
               "--per-item-timeout", str(a.per_item_timeout)]
        # CAPSTONE_QEMU_LOCK_HELD=1: the caller (the nightly) already holds the rootfs
        # lock; a nested flock on the same file would wait for it forever.
        if not os.environ.get("CAPSTONE_QEMU_LOCK_HELD"):
            cmd = ["flock", str(lock)] + cmd
        rc, o, e = sh(cmd, timeout=None)
        print(o.strip()); print(e.strip()[-2000:] if e.strip() else "", file=sys.stderr)
        got = {}
        if res.exists():
            for l in open(res):
                p = l.rstrip("\n").split("\t")
                if len(p) >= 3:
                    got[p[0]] = (p[1], p[2], p[3] if len(p) > 3 else "")
        for name, path, seed, lvl, native, sha in manifest:
            v = got.get(name)
            if v is None:
                rows.append(("csmith", seed, lvl, "ERROR", f"{native:X}", "", sha, "no row from the batch runner")); continue
            kind, val, note = v
            if kind == "RET":
                value = int(val)
                if lvl == "control":
                    verdict = "CONTROL-FAILED" if True else ""
                    rows.append(("control", 0, "fault", "CONTROL-FAILED", "", str(value), sha, "fault_domain returned instead of faulting"))
                    continue
                if lvl.endswith("-xor"):
                    verdict = "CONTROL-OK" if value != native else "CONTROL-FAILED"
                    rows.append(("control", seed, lvl, verdict, f"{native:X}", f"{value:X}", sha, "XOR build must mismatch"))
                    continue
                verdict = "MATCH" if value == native else "MISMATCH"
                rows.append(("csmith", seed, lvl, verdict, f"{native:X}", f"{value:X}", sha, note))
                if verdict == "MISMATCH":
                    keep_finding(out, seed, lvl, out / "src" / f"cs{seed}.c", path, f"native {native:X} domain {value:X}")
            else:
                if lvl == "control":
                    rows.append(("control", 0, "fault", "CONTROL-OK" if kind == "FAULT" else "CONTROL-FAILED", "", "", sha, f"fault_domain read {kind}"))
                    continue
                rows.append(("csmith", seed, lvl, kind, f"{native:X}", "", sha, note))
                if kind in ("FAULT", "TIMEOUT", "WEDGE"):
                    keep_finding(out, seed, lvl, out / "src" / f"cs{seed}.c", path, f"{kind}: {note}")
    elif manifest:
        for name, path, seed, lvl, native, sha in manifest:
            rows.append(("csmith", seed, lvl, "BUILT", f"{native:X}", "", sha, "--no-qemu"))

    tsv = pathlib.Path(a.tsv)
    tsv.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    with open(tsv, "a") as f:
        for gen, seed, lvl, verdict, native, value, sha, note in rows:
            f.write("\t".join(["fuzz", stamp, gen, str(seed), str(lvl), verdict, native, value, sha, qid, note]) + "\n")
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    print(f"fuzz-campaign {a.gen} seeds {a.start}..{a.start + a.count - 1} levels {','.join(levels)} {qid}: "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"; rows -> {tsv}")
    if not rows:
        print("ERROR: nothing ran", file=sys.stderr); sys.exit(2)
    bad = sum(counts.get(k, 0) for k in ("MISMATCH", "FAULT", "TIMEOUT", "WEDGE", "BUILD-CRASH", "CONTROL-FAILED", "ERROR"))
    if a.control and not a.no_qemu:
        if counts.get("CONTROL-OK", 0) < 2:
            print("ERROR: a positive control did not fire (XOR must MISMATCH, fault_domain must FAULT)", file=sys.stderr)
            sys.exit(1)
        bad -= 0
    sys.exit(1 if bad else 0)


def keep_finding(out, seed, lvl, src, dom, note):
    d = out / "findings" / f"cs{seed}"
    d.mkdir(exist_ok=True)
    if src and os.path.exists(src):
        with open(src, "rb") as s, open(d / os.path.basename(src), "wb") as t:
            t.write(s.read())
    if dom and os.path.exists(dom):
        with open(dom, "rb") as s, open(d / os.path.basename(dom), "wb") as t:
            t.write(s.read())
    with open(d / "NOTES.txt", "a") as f:
        f.write(f"{lvl}: {note}\n")


if __name__ == "__main__":
    main()
