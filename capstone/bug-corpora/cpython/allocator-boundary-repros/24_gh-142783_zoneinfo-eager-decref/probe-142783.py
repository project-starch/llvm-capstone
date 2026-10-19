#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

Which allocator layer owns this case's object ON THE TARGET?

case.json records `L1-pymalloc-pool` from the host measurement: ASan reports a
328-byte region, under the 512 B SMALL_REQUEST_THRESHOLD. But a purecap ABI
inflates every object that holds pointers -- case 06 is the proof that host and
target object sizes are not the same number -- and the freed object here is a
ZoneInfo subclass instance, whose C struct is pointer-dense. On the target that
struct measures 0x190 = 400 bytes in the binary's DWARF against 208 on the
host, so the request may well cross 512 and leave pymalloc entirely.

That decides how this case's base-arm reading should be read. A pymalloc block
is freed to pymalloc's own list and the base arm never sees a free, so a
cause=25 there would need explaining. An allocation over 512 goes to the system
allocator, whose free revokes immediately on Capstone, and cause=25 on BOTH
virtual arms is then just the lifetime check doing its job.

sys.getallocatedblocks() counts pymalloc blocks only, so the delta across a
batch of allocations says which layer served them. bytes(n) is the calibration:
its request is the PyBytesObject header plus n plus one, so the delta collapses
once that crosses the threshold. ZoneInfo.no_cache avoids the module's instance
cache, so each call really allocates.

No defect is performed: this subclasses ZoneInfo without the trigger's
BombDescriptor, so nothing is freed early.
"""
import os
import sys

from zoneinfo import ZoneInfo

_UTC_TZIF = (
    "VFppZjIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAQAAAAAAAAAAAAAAAQAAAAQAAAAA"
    "AABVVEMAAABUWmlmMgAAAAAAAAAAAAAAAAAAAAAAAAEAAAABAAAAAAAAAAEAAAAB"
    "AAAABPgAAAAAAAAAAAAAAAAAAFVUQwAAAApVVEMwCg=="
)


def vendor_utc():
    import base64
    import tempfile
    import zoneinfo
    try:
        ZoneInfo("UTC")
        return "system"
    except Exception:                                      # noqa: BLE001
        pass
    d = tempfile.mkdtemp(prefix="tzdata-")
    with open(os.path.join(d, "UTC"), "wb") as f:
        f.write(base64.b64decode("".join(_UTC_TZIF)))
    zoneinfo.reset_tzpath([d])
    ZoneInfo("UTC")
    return "vendored"


def say(line):
    os.write(1, ("PROBE-142783 " + line + "\n").encode())


class Plain(ZoneInfo):
    pass


N = 200


def delta(make):
    keep = []
    before = sys.getallocatedblocks()
    for _ in range(N):
        keep.append(make())
    after = sys.getallocatedblocks()
    del keep
    return (after - before) / float(N)


say("tzdata=%s" % vendor_utc())
say("ZoneInfo.__basicsize__=%d Plain.__basicsize__=%d flags=%#x"
    % (ZoneInfo.__basicsize__, Plain.__basicsize__, Plain.__flags__))

inst = Plain.no_cache("UTC")
say("sys.getsizeof(instance)=%d" % sys.getsizeof(inst))
say("SMALL_REQUEST_THRESHOLD=512")
say("host reference: basicsize=208, getsizeof=%s, ASan total request=328"
    % "240 expected")
say("DONE")
