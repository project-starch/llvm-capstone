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
it; anything else is refused. The file is replaced, not edited in place, so a
hard link to SPEC's data/ copy is never written through. Prints one JSON line.
usage: perl-pack-stub.py RUNDIR/t/op/pack.t"""
import hashlib, json, os, sys

PRISTINE = "b3e0961f7898f5bd40d0b2e22e54e522a1769fdba670b6e1be212a5a7f17a9f5"
STUBBED = "e32d7f517eb98a9d4757b9ef411c84654cff4f0c9af6c55384c590927cfb3830"
CHECKS = ['is(unpack("p>",pack("p>","foo")), "foo");',
          'is(unpack("p>",pack("p>",239)), 239);',
          'is(unpack("p", $up), "abc\\xa5", "unpack p upgraded");',
          'is(unpack("P7", $up),   "abc\\xa5\\x00\\xfed", "unpack P upgraded");']
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
    s = data.decode("latin-1")
    for check in CHECKS:
        if s.count(check) != 1:
            sys.exit(f"perl-pack-stub: expected exactly one {check!r}")
        s = s.replace(check, STUB)
    out = s.encode("latin-1")
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
