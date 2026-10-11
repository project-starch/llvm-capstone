#!/usr/bin/env python3
"""Trigger for gh142783 -- the reproducer from upstream issue #142783.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (328-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""

# The Capstone domain's image has _zoneinfo built in but no tzdata, so
# ZoneInfo("UTC") died in zoneinfo/_common.py load_tzdata before this case
# reached its defect, and the row read as the arm staying quiet. The UTC TZif
# is 127 bytes, so it is carried here rather than installed: the real tzdata is
# used wherever it exists and this runs only when it does not, leaving the host
# and CheriBSD on exactly the path they took before.
_UTC_TZIF = (
    "VFppZjIAAAAAAAAAAAAAAAAAAAAAAAABAAAAAQAAAAAAAAAAAAAAAQAAAAQAAAAA"
    "AABVVEMAAABUWmlmMgAAAAAAAAAAAAAAAAAAAAAAAAEAAAABAAAAAAAAAAEAAAAB"
    "AAAABPgAAAAAAAAAAAAAAAAAAFVUQwAAAApVVEMwCg=="
)

def _vendor_utc():
    import base64, os, tempfile, zoneinfo
    try:
        zoneinfo.ZoneInfo("UTC")
        return "system"
    except Exception:
        pass
    d = tempfile.mkdtemp(prefix="tzdata-")
    with open(os.path.join(d, "UTC"), "wb") as f:
        f.write(base64.b64decode("".join(_UTC_TZIF)))
    zoneinfo.reset_tzpath([d])
    zoneinfo.ZoneInfo("UTC")          # fail here rather than inside the case
    return "vendored"

# Recorded so a probe can see which branch ran without calling it a second
# time, which would report "system" once the vendored copy is already in place.
_TZDATA_SOURCE = _vendor_utc()
from zoneinfo import ZoneInfo

class Cache:
    def get(self, *args, **kwargs):
        return None
    def setdefault(self, *args, **kwargs):
        return None
    def clear(self, *args, **kwargs):
        pass

class BombDescriptor:
    def __get__(self, obj, owner):
        return Cache()

class EvilZoneInfo(ZoneInfo):
    pass

EvilZoneInfo._weak_cache = BombDescriptor()

EvilZoneInfo("UTC")

