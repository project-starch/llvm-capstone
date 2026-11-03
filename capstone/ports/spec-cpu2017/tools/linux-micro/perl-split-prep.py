#!/usr/bin/env python3
"""perl-split-prep.py SPEC_ROOT LABEL OUTDIR [PER_GROUP]: 500.perlbench_r's test input split into
groups that each run in their own guest (perl-split-one.sh), so the 2-4 h Capstone run becomes
~30 parallel runs of minutes.

The test input is `perlbench -I. -I./lib test.pl`; test.pl is `do 't/TEST'`, and t/TEST runs the 317
test scripts of its hard-coded list one after another, each as a child perlbench (`../$^X`), and
prints one line per script. A group is the same harness over PER_GROUP consecutive scripts of that
list (default 10): OUTDIR/gNN/ is a run directory as runcpu builds it (data/all/input +
data/test/input + the executable, pack.t stubbed by the kit's perl-pack-stub.py), plus
t/TEST-gNN = t/TEST with only the group's scripts in the list and test-gNN.pl = test.pl naming it,
and expected.out = the lines of SPEC's test.out for those scripts and "All tests successful.".
Nothing in the SPEC tree is written. Each group repeats the harness's own start (one perlbench
start and the compile of t/TEST), which the sum over groups therefore counts once per group."""
import json, os, re, shutil, subprocess, sys

spec, label, out = sys.argv[1:4]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 10
kit = os.environ.get("KIT", os.path.expanduser("~/prof-dev/kit"))
B = os.path.join(spec, "benchspec/CPU/500.perlbench_r")
exe = os.path.join(B, "exe", f"perlbench_r_base.{label}")
test_dir = os.path.join(B, "data/test/input")
TEST = open(os.path.join(test_dir, "t/TEST")).read()
m = re.search(r"(if \(1\) \{ # SPEC CPU\n.*?push @ARGV, \(qw\(\n)(.*?)(\n\s*\)\);)", TEST, re.S)
if not m:
    sys.exit("perl-split-prep: t/TEST's hard-coded list not found")
tests = m.group(2).split()
expected = open(os.path.join(B, "data/test/output/test.out")).read().splitlines()
lines = {}
for l in expected:
    mm = re.match(r"t/(\S+) \.+ ", l)
    if mm:
        lines[mm.group(1)] = l
missing = [t for t in tests if t[:-2] not in lines]
if len(tests) != 317 or missing or expected[-1] != "All tests successful.":
    sys.exit(f"perl-split-prep: {len(tests)} tests, {len(missing)} without an expected line, last line {expected[-1]!r}")
os.makedirs(out)
template = os.path.join(out, "template")
os.makedirs(template)
for src in (os.path.join(B, "data/all/input"), test_dir):
    subprocess.run(["cp", "-a", src + "/.", template], check=True)
shutil.copy2(exe, template)
stub = subprocess.run([sys.executable, os.path.join(kit, "tools/perl-pack-stub.py"),
                       os.path.join(template, "t/op/pack.t")], capture_output=True, text=True, check=True).stdout
groups = [tests[i:i + per] for i in range(0, len(tests), per)]
manifest = {"exe": exe, "exe_sha256": subprocess.run(["sha256sum", exe], capture_output=True, text=True).stdout[:64],
            "tests": len(tests), "per_group": per, "pack_stub": json.loads(stub), "groups": []}
for g, members in enumerate(groups):
    name = f"g{g:02d}"
    d = os.path.join(out, name)
    subprocess.run(["cp", "-a", template, d], check=True)
    body = TEST[:m.start(2)] + "\n".join(members) + TEST[m.end(2):]
    open(os.path.join(d, "t", f"TEST-{name}"), "w").write(body)
    open(os.path.join(d, f"test-{name}.pl"), "w").write(f"$::verbose = 0;\ndo 't/TEST-{name}';\n")
    open(os.path.join(d, "expected.out"), "w").write(
        "".join(lines[t[:-2]] + "\n" for t in members) + "All tests successful.\n")
    manifest["groups"].append({"name": name, "tests": members})
shutil.rmtree(template)
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(f"{len(groups)} groups of up to {per} in {out}")
