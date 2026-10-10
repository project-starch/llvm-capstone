#!/usr/bin/env python3
"""Run Perl's own test files on one virtual image, and record how each ended.

    qualify.py OUT --state VM --perl PERLD_ROOT --llvm-bin BIN [--timeout S] TEST...

TEST is a path under the release's t/ (op/sub.t). The build's src/perl-5.36.3/t and lib are
staged in the VM's share, and each file runs as `perl TEST` from t/, as `make test` runs it. One
row per file: completed or exitN with the count of `ok` and `not ok` lines, or a fault with its
cause and function. This is the false-positive check for PERLD_SUBLET=1: a file that runs on the
stock image and faults on the protected one names a read of a freed SV head that the patch does
not route through its arena. Compare two runs file by file; the counts alone are not the check.

The virtual profile starts no subprocesses (system() returns -1, a piped open is ENOSYS,
backticks fault), so the staged t/test.pl has its child-perl helpers replaced by stubs, on both
images alike, as the corpus's own harness does: a test that needs a child then reports `not ok`
instead of ending the file.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402


NO_CHILDREN = '''
# qualify.py: no subprocesses on the virtual profile; the child-perl helpers return empty.
no warnings 'redefine';
sub runperl { '' }
*run_perl = \\&runperl;
sub fresh_perl { '' }
sub fresh_perl_is { 1 }
sub fresh_perl_like { 1 }
sub run_multiple_progs { 1 }
1;
'''


def stage(perl_root, share):
    src = perl_root / "src/perl-5.36.3"
    for name in ("t", "lib"):
        home = share / f"perl-q/{name}"
        stamp = home / ".staged-from"
        # The stub is part of what was staged: a path alone kept a tree staged without it.
        want = str(src / name) + ("\n" + NO_CHILDREN if name == "t" else "")
        if stamp.is_file() and stamp.read_text() == want:
            continue
        shutil.rmtree(home, ignore_errors=True)
        shutil.copytree(src / name, home, ignore=shutil.ignore_patterns("*.o", "*.a"), symlinks=True)
        if name == "t":
            with open(home / "test.pl", "a") as f:
                f.write(NO_CHILDREN)
        stamp.write_text(want)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--perl", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("tests", nargs="+")
    a = p.parse_args()
    image = a.perl / "src/perl-5.36.3/perl"
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    stage(a.perl, share)
    symbols = v.Symbols(a.llvm_bin, image)
    raw = a.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    env = ("-e", "PERL5LIB=/mnt/host/perl-q/lib", "--cwd", "/mnt/host/perl-q/t")
    rows = ["test\toutcome\tcause\tfunction\tok\tnot_ok\n"]
    for test in a.tests:
        text, result = appvm.run_app(a.state, image, [test], raw / test.replace("/", "_"),
                                     run_args=env, timeout=a.timeout)
        ok = len(re.findall(r"^ok \d", text, re.M))
        notok = len(re.findall(r"^not ok \d", text, re.M))
        fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
        if "[runner] TIMEOUT" in text:
            kind, cause, func = "timeout", "", ""
        elif fault:
            kind, cause, func = "fault", str(fault.cause), fault.symbol or "?"
        else:
            kind = "completed" if result.get("value") == 0 else f"{result.get('kind')}{result.get('value')}"
            cause = func = ""
        rows.append(f"{test}\t{kind}\t{cause}\t{func}\t{ok}\t{notok}\n")
        print(f"  {test:28} {kind:10} {cause:>3} {func[:28]:28} ok={ok} not_ok={notok}", flush=True)
    (a.output / "qualify.tsv").write_text("".join(rows))
    (a.output / "inputs.json").write_text(json.dumps({
        "image_sha256": v.sha256(image), "tests": a.tests, "timeout_s": a.timeout,
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
