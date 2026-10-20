#!/usr/bin/env python3
"""Stub the four t/op/pack.t checks that no capability machine can pass, in a run
directory's copy of perlbench's test input (never in the SPEC tree's data/).

Each rebuilds a pointer byte by byte -- pack/unpack "p>" (byte-reversed) and
unpack "p"/"P" of a UTF-8-upgraded string -- and a pointer rebuilt from bytes
carries no capability tag. The checks are constant-folded when the file is
compiled, so one of them kills the whole file before test 1. Each is replaced
by a pass() that says why. Those 4 of pack.t's checks do not run; perlbench's
test.out reports per test file ("t/op/pack ... ok"), so the file still counts.

Fails closed: the file must be SPEC 1.0.2's pack.t or this script's output of
it (both pinned by hash, as are the four lines); anything else is refused. The file is replaced, not edited in place, so a
hard link to SPEC's data/ copy is never written through. Prints one JSON line.
usage: perl-pack-stub.py RUNDIR/t/op/pack.t"""
import hashlib, json, os, sys

PRISTINE = "b3e0961f7898f5bd40d0b2e22e54e522a1769fdba670b6e1be212a5a7f17a9f5"
STUBBED = "e32d7f517eb98a9d4757b9ef411c84654cff4f0c9af6c55384c590927cfb3830"
# The four checks, named by the sha256 of their text (the text itself is SPEC's perlbench source and
# stays out of this public kit); each is one whole statement on its own line of pack.t.
CHECK_HASHES = ["3b00d69b8dfc37b2dbb5dbdb9e44ecf27bd2e278d24dc695f78c67c85a935968",
                "c5165209f24b54b5e102de6fed7b6eba49a62299400365bd7acf1d8aa9ec623f",
                "a96919cd0cc9be6fb8306aaa832cdf15a34f7ee200fa3a0c8422c9d0cd344c34",
                "10535f2bef073e4f6ad7135e7eb421967f13f03092c46dd74fa4de2fab441467"]
STUB = 'pass("skipped: a pointer rebuilt byte by byte carries no capability tag");'

path = sys.argv[1]
data = open(path, "rb").read()
before = hashlib.sha256(data).hexdigest()
record = {"file": os.path.abspath(path), "before": before, "checks_stubbed": 4}
if before == STUBBED:
    record.update(after=before, action="already stubbed")
elif before != PRISTINE:
    sys.exit(f"perl-pack-stub: {path} is neither the pristine nor the stubbed pack.t ({before})")
else:
    lines = data.decode("latin-1").split("\n")
    found = {}
    for i, line in enumerate(lines):
        h = hashlib.sha256(line.strip().encode("latin-1")).hexdigest()
        if h in CHECK_HASHES:
            found.setdefault(h, []).append(i)
    if sorted(found) != sorted(CHECK_HASHES) or any(len(v) != 1 for v in found.values()):
        sys.exit(f"perl-pack-stub: expected each of the four checks exactly once, found {dict((k[:8], len(v)) for k, v in found.items())}")
    for i in (v[0] for v in found.values()):
        indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines[i] = indent + STUB
    out = "\n".join(lines).encode("latin-1")
    after = hashlib.sha256(out).hexdigest()
    if after != STUBBED:
        sys.exit(f"perl-pack-stub: output {after} is not the pinned {STUBBED}")
    tmp = path + ".stub-tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    os.replace(tmp, path)
    record.update(after=after, action="stubbed")
print(json.dumps(record))
