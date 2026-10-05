#!/usr/bin/env python3
"""Run several real programs in one capstone VM, concurrently or one after another.

    run-mix.py --state <vm state> --bench <prepare-bench.py out dir> --mode par|seq
               --mruby <guest path of the mruby image> --speedtest1 <guest path> PROGRAM...

PROGRAM is a benchmark name from prepare-bench.py or `speedtest1`, or, with --programs, a
name from that manifest (ptrbench.py's programs.json: guest cwd, command, stdin and the native
reference output, relative to --bench). In `par` mode every
program is started at once, each as its own `capstone_vm exec` (its own Linux process
and domain); in `seq` mode the same list runs one after another -- the control that
differs from `par` only in the interleaving. Each program runs in its own guest
directory. Every output is compared with the native reference; the exit status is 0
only if every program exited 0 and matched.
"""
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


def command(args, i, prog):
    if prog in args.manifest:
        m = args.manifest[prog]
        redirect = f" < {m['stdin']}" if m.get("stdin") else ""
        # a domain fault's record goes to stderr only when asked for
        return f"cd {m['cwd']} && export CAPSTONE_EXEC_DIAGNOSTICS=1 && {m['cmd']}{redirect}"
    work = f"mkdir -p /tmp/mix{i} && cd /tmp/mix{i} && "
    if prog == "speedtest1":
        return work + f"rm -f st.db st.db-journal && {args.speedtest1} --size 1 st.db"
    return work + f"{args.mruby} /mnt/host/bench/{prog}.rb"


def check(args, prog, out):
    if prog in args.manifest:
        return out == (args.bench / args.manifest[prog]["native_out"]).read_bytes()
    want = (args.bench / "native" / f"{prog}.out").read_bytes()
    if prog == "speedtest1":
        got = "\n".join(l for l in out.decode(errors="replace").splitlines()
                        if l.startswith("STUDY-ORACLE")) + "\n"
        return got.encode() == want
    return out == want


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--mode", choices=["par", "seq"], required=True)
    p.add_argument("--mruby", required=True)
    p.add_argument("--speedtest1", required=True)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--programs", dest="manifest_path", type=Path, help="ptrbench.py's programs.json")
    p.add_argument("--save-dir", type=Path, help="keep every program's guest stdout here, for diffing a FAIL")
    p.add_argument("programs", nargs="+")
    args = p.parse_args()
    args.manifest = json.load(open(args.manifest_path)) if args.manifest_path else {}
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state), "exec", "sh", "-c"]
    start = time.monotonic()
    results = []
    if args.mode == "par":
        # one waiter per program, so each program's own end time is recorded
        def one(i, prog):
            t0 = time.monotonic()
            r = subprocess.run(cli + [command(args, i, prog)], capture_output=True, timeout=args.timeout)
            return (i, prog, r.returncode, r.stdout, r.stderr, time.monotonic() - t0)
        with concurrent.futures.ThreadPoolExecutor(len(args.programs)) as pool:
            results = list(pool.map(one, range(len(args.programs)), args.programs))
    else:
        for i, prog in enumerate(args.programs):
            t0 = time.monotonic()
            r = subprocess.run(cli + [command(args, i, prog)], capture_output=True, timeout=args.timeout)
            results.append((i, prog, r.returncode, r.stdout, r.stderr, time.monotonic() - t0))
    ok = True
    for i, prog, rc, out, err, secs in results:
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            (args.save_dir / f"{args.mode}-{i}-{prog}.out").write_bytes(out)
        match = rc == 0 and check(args, prog, out)
        ok &= match
        print(f"{args.mode} #{i} {prog}: {'PASS' if match else 'FAIL'} rc={rc} {secs:.1f}s")
        if not match:
            print("   stderr: " + " | ".join(err.decode(errors='replace').strip().splitlines()[-3:]))
    print(f"mix {args.mode} n={len(args.programs)}: {'PASS' if ok else 'FAIL'} wall={time.monotonic() - start:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
