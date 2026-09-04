#!/usr/bin/env python3
"""IR-level crash fuzzing of the Capstone backend with llvm-stress (Tier 3).

For each seed: generate a module with llvm-stress, retarget it to capstone64 (the
pure-capability datalayout is prepended and every `ptr` becomes `ptr addrspace(200)`,
because A200/P200 make an address-space-0 pointer meaningless here), then run llc at
each requested level with -verify-machineinstrs, and optionally opt -O2 first.

Each run is classified:
  OK            llc exited 0
  CRASH <sig>   llc died with a stack dump / assertion / unreachable
  ERROR <msg>   llc exited non-zero with a diagnostic and no crash
  HANG          llc exceeded --timeout

A crash SIGNATURE is the first diagnostic line with numbers and SDNode ids stripped,
plus the first Capstone-specific frame of the stack trace when there is one, so the
same defect groups together across seeds and a target-specific crash is told apart
from a generic legalizer one.  --known lists signatures already filed (one per line:
`substring | ID | note`); a crash matching one is reported but does not fail the run.

Exit 0 when every non-OK result matches a known signature; 1 when an unknown crash,
error or hang was seen; 2 on a tool problem.  Crashing modules are kept under
<out>/crashes/<sig-hash>/ for reduction (capstone/tests/reduce.sh).
"""
import argparse
import concurrent.futures as cf
import hashlib
import os
import re
import subprocess
import sys
import time

DL = ('target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"\n'
      'target triple = "capstone64"\n')
SIG_LINE = re.compile(r"(Assertion `.*' failed|LLVM ERROR: .*|Cannot select: .*|UNREACHABLE executed.*|"
                      r"fatal error: .*|error: .*|.*: Assertion .*|.*Segmentation fault.*)")
CAP_FRAME = re.compile(r"#\d+ .*?(llvm::Capstone[A-Za-z_]*::[A-Za-z_0-9]+|\bCapstone[A-Za-z_0-9]+\b|\(anonymous namespace\)::Capstone[A-Za-z_0-9]+)")


def normalize(s):
    s = re.sub(r"0x[0-9a-fA-F]+", "0x?", s)
    s = re.sub(r"\bt\d+\b", "t?", s)
    s = re.sub(r"\b\d+\b", "?", s)
    s = re.sub(r"/[^ ]*/", "", s)
    s = re.sub(r"^(error: )?llc: [^ ]*\.(ll|bc):\?:\?: ", "", s)
    return s.strip()[:200]


def classify(rc, err, timed_out):
    if timed_out:
        return "HANG", "timeout"
    if rc == 0:
        return "OK", ""
    lines = err.replace("\r", "\n").split("\n")
    crash = any(("PLEASE submit a bug report" in l) or ("Stack dump" in l) or ("Assertion" in l)
                or ("UNREACHABLE executed" in l) or ("Segmentation fault" in l) for l in lines)
    sig = ""
    for l in lines:
        m = SIG_LINE.search(l)
        if m:
            sig = normalize(m.group(1))
            break
    if not sig:
        sig = normalize(next((l for l in lines if l.strip()), f"rc={rc}"))
    if crash:
        frame = ""
        for l in lines:
            m = CAP_FRAME.search(l)
            if m:
                frame = m.group(1)
                break
        return "CRASH", (sig + (" @ " + frame if frame else " @ generic"))
    return "ERROR", sig


def retarget(text):
    # The textual IR parser gives a bare `alloca` address space 0 regardless of the
    # datalayout's A200, so the address space must be spelled out on every alloca as
    # well as on every pointer type.
    text = re.sub(r"\bptr\b(?! addrspace)", "ptr addrspace(200)", text)
    text = re.sub(r"^(\s*%[\w.]+ = alloca [^\n]*?)(?<!addrspace\(200\))$", r"\1, addrspace(200)", text, flags=re.M)
    return DL + text


def run(cmd, timeout):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stderr, False, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return -1, (e.stderr or b"").decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or ""), True, timeout


def one_seed(a, seed):
    out = os.path.join(a.out, "seeds")
    if isinstance(seed, str):
        # --files mode: an existing module, run as it is (it carries its own triple).
        ll = seed
        seed = os.path.basename(seed)
    else:
        ll = os.path.join(out, f"seed-{seed}.ll")
        rc, err, to, _ = run([a.bin + "/llvm-stress", f"-seed={seed}", f"-size={a.size}", "-o", ll + ".raw"], 60)
        if rc != 0:
            return [(seed, "gen", "ERROR", "llvm-stress failed: " + normalize(err))]
        text = open(ll + ".raw").read()
        os.remove(ll + ".raw")
        has_fp = bool(re.search(r"\b(float|double|half|bfloat|fp128)\b", text))
        has_vec = "<" in text and " x " in text
        if a.skip_fp and has_fp:
            return [(seed, "gen", "SKIP", "fp")]
        if a.skip_vec and has_vec:
            return [(seed, "gen", "SKIP", "vector")]
        open(ll, "w").write(retarget(text))
    results = []
    common = ["-mtriple=capstone64", "-mattr=+m", "-verify-machineinstrs", "-o", "/dev/null"]
    for lvl in a.levels:
        rc, err, to, dt = run([a.bin + "/llc"] + common + [f"-{lvl}", ll], a.timeout)
        verdict, sig = classify(rc, err, to)
        results.append((seed, lvl, verdict, sig))
        if verdict != "OK":
            keep(a, seed, lvl, sig, ll, err)
    if a.opt:
        # Always under --out: in --files mode `ll` is someone else's file.
        bc = os.path.join(out, os.path.basename(ll) + ".opt.bc")
        # -mtriple so a module without a datalayout gets the target's (functions and
        # globals in address space 200), as llc gives it.
        rc, err, to, dt = run([a.bin + "/opt", "-mtriple=capstone64", "-O2", ll, "-o", bc], a.timeout)
        if rc != 0 or to:
            verdict, sig = classify(rc, err, to)
            results.append((seed, "opt-O2", verdict, sig))
            keep(a, seed, "opt-O2", sig, ll, err)
        else:
            rc, err, to, dt = run([a.bin + "/llc"] + common + ["-O2", bc], a.timeout)
            verdict, sig = classify(rc, err, to)
            results.append((seed, "opt-O2+llc-O2", verdict, sig))
            if verdict != "OK":
                keep(a, seed, "opt-O2+llc-O2", sig, bc, err)
    return results


def keep(a, seed, lvl, sig, path, err):
    h = hashlib.sha1(sig.encode()).hexdigest()[:10]
    d = os.path.join(a.out, "crashes", h)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, f"seed-{seed}-{lvl}")
    ext = ".bc" if path.endswith(".bc") else ".ll"
    if not os.path.exists(base + ext):
        with open(path, "rb") as s, open(base + ext, "wb") as t:
            t.write(s.read())
        with open(base + ".stderr", "w") as f:
            f.write(err[:20000])
        with open(os.path.join(d, "SIGNATURE"), "w") as f:
            f.write(sig + "\n")


def load_known(path):
    known = []
    if path and os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            parts = [p.strip() for p in l.split("|")]
            known.append((parts[0], parts[1] if len(parts) > 1 else "?"))
    return known


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--levels", default="O0,O2", help="comma-separated llc levels")
    ap.add_argument("--opt", action="store_true", help="also run opt -O2 then llc -O2")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per llc/opt run")
    ap.add_argument("--skip-fp", action="store_true", help="skip modules that use floating point")
    ap.add_argument("--skip-vec", action="store_true", help="skip modules that use vectors")
    ap.add_argument("--bin", default=os.environ.get("CAPSTONE_LLVM_BIN", ""), help="LLVM bin dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--known", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "known-signatures.txt"))
    ap.add_argument("--files", nargs="*", help="classify these existing .ll files instead of generating seeds")
    a = ap.parse_args()
    a.levels = [l.strip() for l in a.levels.split(",") if l.strip()]
    if not a.bin or not os.path.exists(os.path.join(a.bin, "llc")) or not os.path.exists(os.path.join(a.bin, "llvm-stress")):
        print(f"ERROR: --bin/CAPSTONE_LLVM_BIN must hold llc and llvm-stress (got {a.bin!r})", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(a.out, "seeds"), exist_ok=True)
    known = load_known(a.known)

    rows = []
    work = list(a.files) if a.files is not None else list(range(a.start, a.start + a.seeds))
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for res in ex.map(lambda s: one_seed(a, s), work):
            rows.extend(res)
    rows.sort(key=lambda r: (str(r[0]), r[1]))
    tsv = os.path.join(a.out, "results.tsv")
    with open(tsv, "w") as f:
        for seed, lvl, verdict, sig in rows:
            f.write(f"{seed}\t{lvl}\t{verdict}\t{sig}\n")

    groups = {}
    for seed, lvl, verdict, sig in rows:
        if verdict in ("CRASH", "ERROR", "HANG"):
            groups.setdefault((verdict, sig), []).append((seed, lvl))
    n_ok = sum(1 for r in rows if r[2] == "OK")
    n_skip = sum(1 for r in rows if r[2] == "SKIP")
    unknown = 0
    what = f"{len(a.files)} file(s)" if a.files is not None else f"seeds {a.start}..{a.start + a.seeds - 1} size {a.size}"
    print(f"llc-stress: {what} levels {','.join(a.levels)}"
          f"{' +opt' if a.opt else ''}: {n_ok} OK, {n_skip} skipped, {len(groups)} distinct failure signature(s)")
    for (verdict, sig), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ids = [i for sub, i in known if sub and sub in sig]
        tag = f"KNOWN {ids[0]}" if ids else "UNKNOWN"
        unknown += 0 if ids else 1
        h = hashlib.sha1(sig.encode()).hexdigest()[:10]
        print(f"  {verdict:5} x{len(members):<4} {tag:14} [{h}] {sig}   e.g. seed {members[0][0]} {members[0][1]}")
    print(f"  results: {tsv}")
    if not rows or all(r[2] == "SKIP" for r in rows):
        print("ERROR: no seed was run -- nothing was tested", file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if unknown else 0)


if __name__ == "__main__":
    main()
