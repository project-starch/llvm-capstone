"""Negative control for case 24, variant B: _weak_cache is not touched at all.

The defect is a descriptor on _weak_cache that returns a FRESH Cache() on every
read, so the eager decref drops the last reference to an object still in use.
Variant A kept the descriptor and made it return one stable Cache(); this
variant removes it entirely. Same ZoneInfo subclass, same construction, same
tzdata, same allocation traffic, nothing abnormal installed on the class.
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

# Variant A kept the descriptor and returned a stable Cache(). In the domain it
# faulted with cause 7, a STORE bounds fault -- a different mechanism from the
# cause 24 the real trigger produces -- so the subversion itself may be the
# problem rather than the allocation pattern. This variant removes the
# descriptor entirely: the same subclass, the same construction, the same
# tzdata, nothing abnormal on _weak_cache.
#
# If B is silent and A faults, the fault belongs to A and case 24's detection is
# neither confirmed nor refuted by A. If B faults too, the base arm faults on
# this case's ordinary traffic and that detection is in question.

class BenignZoneInfo(ZoneInfo):
    pass

BenignZoneInfo("UTC")
print("NEGATIVE-CONTROL no defect performed")
