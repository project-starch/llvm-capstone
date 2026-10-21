#!/usr/bin/env python3
"""perl-lib-requires.py OUT --state VM --arm ARM --perl PERLD_ROOT --llvm-bin BIN [--raw DIR] [--only NN,NN]: which
library files the perl release-differential cases load, measured in the guest, for a board boot whose payload cannot
carry the whole lib (perl-lib-closure.py turns this into the closure).

It is a MEASUREMENT, never a reading. Each case and both controls run as perl/release-differential/runners/virtual/
run-virtual.py runs them (the same image, staged library, environment and trigger file), except that the script is
entered through `-e` with an @INC hook in front that prints `@@REQ <name>` for every require that searches @INC, the
core's own implicit loads included (Tie::Hash::NamedCapture, attributes, PerlIO layers, charnames), and then `do`es
the trigger. The hook changes the heap, so a case's outcome here may differ from its reading; each case's outcome is
recorded beside its names, and a closure made from a run whose outcomes differ from the reference's is suspect.
Runs through tier3-appvm-arm.sh like any runner (collect, one boot, read). Writes OUT/requires.json.
Exit: 0 when both controls produced their marker and every case ran, 75 otherwise.
"""
import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
RUNNER = REPO / "capstone/bug-corpora/perl/release-differential/runners/virtual/run-virtual.py"
spec = importlib.util.spec_from_file_location("perl_run_virtual", RUNNER)
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)              # puts tools/ on sys.path; its own imports run once
appvm, v, sdk_builtins = rv.appvm, rv.v, rv.sdk_builtins

HOOK = 'BEGIN { unshift @INC, sub { print STDERR "\\@\\@REQ $_[1]\\n"; return } } '
REQ = re.compile(r"^@@REQ (\S+)$", re.M)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=("virtual-malloc", "virtual-nested-pools"))
    p.add_argument("--perl", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path)
    p.add_argument("--only")
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()
    sdk_builtins.gate([a.perl], who="perl-lib-requires.py")
    if appvm.profile(a.state) != "virtual":
        p.error(f"{a.state} is not a virtual-profile VM")
    image = a.perl / "src/perl-5.36.3/perl"
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    rv.stage_library(a.perl, share)
    cases_dir = share / "perl-cases"
    cases_dir.mkdir(exist_ok=True)
    env = ("-e", "PERL5LIB=/mnt/host/perl-lib:/mnt/host/perl-harness", "--cwd", "/mnt/host/perl-cases")
    raw = a.raw or a.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    symbols = v.Symbols(a.llvm_bin, image)
    runs, ok = {}, True

    def one(name, body, marker=None):
        text, result = appvm.run_app(a.state, image, ["-e", HOOK + body], raw / name, run_args=env, timeout=a.timeout)
        kind = rv.outcome(text, result, symbols)[0]
        runs[name] = {"outcome": kind, "requires": REQ.findall(text),
                      "marker": None if marker is None else marker in text}
        return runs[name]

    collect = (appvm.session(a.state) or {}).get("phase") == "collect"      # nothing runs in a session's collect
    for name, body, marker in (("control-eval", 'print 6*7, "\\n"', "42"),
                               ("control-shim", 'require "shim.pl"; ok(1); print "SHIMOK\\n"', "SHIMOK")):
        seen = one(name, body, marker)["marker"]           # every phase calls it, so the collect records it
        ok &= collect or bool(seen)
    cases = sorted(rv.CORPUS.glob("[0-9][0-9]_*"))
    if a.only:
        cases = [d for d in cases if d.name[:2] in set(a.only.split(","))]
    for d in cases:
        claims = json.loads((d / "case.json").read_text())
        shutil.copy(d / claims["trigger"], cases_dir / f"{d.name[:2]}.pl")
        r = one(d.name[:2], f'do "/mnt/host/perl-cases/{d.name[:2]}.pl"; print STDERR "\\@\\@DOERR $@\\n" if $@;')
        print(f"  {d.name[:44]:44} {r['outcome']:9} {len(r['requires']):3} requires", flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "requires.json").write_text(json.dumps({
        "arm": a.arm, "image_sha256": v.sha256(image), "runner": v.sha256(HERE), "hook": HOOK, "runs": runs,
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)}, indent=1) + "\n")
    names = sorted({n for r in runs.values() for n in r["requires"]})
    print(f"--- {a.arm}: {len(runs)} runs, {len(names)} distinct requires")
    return 0 if ok and len(runs) == len(cases) + 2 else 75


if __name__ == "__main__":
    sys.exit(main())
