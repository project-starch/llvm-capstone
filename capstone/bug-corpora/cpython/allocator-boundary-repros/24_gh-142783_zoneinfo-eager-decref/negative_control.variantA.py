"""Negative control for case 24: the descriptor hands back a stable cache.

The defect is a descriptor that returns a FRESH Cache() each time _weak_cache is
read, so the eager decref drops the last reference to an object still in use.
Here the descriptor is still a descriptor, still called on every read, and still
returns a Cache -- the same one each time, so nothing loses its last reference
while in use. The ZoneInfo construction and its allocations are unchanged.
"""
from zoneinfo import ZoneInfo

# Lifted verbatim from trigger.py. The first version of this control dropped
# it and died in load_tzdata inside the domain, before reaching the defect
# site at all -- the evidence gate caught that and voided the run.

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

class Cache:
    def get(self, *args, **kwargs):
        return None
    def setdefault(self, *args, **kwargs):
        return None
    def clear(self, *args, **kwargs):
        pass

_stable = Cache()

class StableDescriptor:
    def __get__(self, obj, owner):
        # The defect returns Cache() -- a new object whose only reference the
        # caller drops. This returns the one held above.
        return _stable

class BenignZoneInfo(ZoneInfo):
    pass

BenignZoneInfo._weak_cache = StableDescriptor()

BenignZoneInfo("UTC")
print("NEGATIVE-CONTROL no defect performed")
