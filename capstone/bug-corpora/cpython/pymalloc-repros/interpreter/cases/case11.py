#!/usr/bin/env python3
"""Python-level trigger for case 11 / gh-142783.

Use-after-free in the ``zoneinfo`` module (``Modules/_zoneinfo.c``).
``get_weak_cache`` did ``Py_XDECREF(cache)`` on the line after
``PyObject_GetAttrString(type, "_weak_cache")`` and returned the value as a
"borrowed reference", assuming the type kept it alive.  A ``_weak_cache``
attribute that is a data descriptor (or otherwise not type-owned) returns a
fresh object whose only reference is dropped by that ``Py_XDECREF`` -- freed and
then used on the next line.  No re-entrancy: the free and the use are adjacent.

Reduced from CPython's own regression test,
``Lib/test/test_zoneinfo/test_zoneinfo.py::test_weak_cache_descriptor_use_after_free``.
Uses the C ``zoneinfo.ZoneInfo``.  Standalone.  A ``--without-pymalloc`` ASan
build reports the heap-use-after-free on pinned v3.13.7.
"""
import zoneinfo

KEY = "America/Los_Angeles"


def main():
    class BombDescriptor:
        def __get__(self, obj, owner):
            return {}                   # fresh cache dict, not type-owned

    class EvilZoneInfo(zoneinfo.ZoneInfo):
        pass

    # Must be set after class creation so ZoneInfo's own cache machinery is set.
    EvilZoneInfo._weak_cache = BombDescriptor()

    zone1 = EvilZoneInfo(KEY)           # get_weak_cache frees the fresh dict...
    assert str(zone1) == KEY

    EvilZoneInfo.clear_cache()
    zone2 = EvilZoneInfo(KEY)           # ...and reads it again here
    assert str(zone2) == KEY
    assert zone2 is not zone1


if __name__ == "__main__":
    try:
        main()
    except zoneinfo.ZoneInfoNotFoundError as exc:
        # The guest must ship the IANA database for this key; the defect itself
        # is in get_weak_cache, reached as soon as a ZoneInfo can be built.
        raise SystemExit("case11 gh-142783: tzdata for %r unavailable: %s"
                         % (KEY, exc))
    print("case11 gh-142783: sequence completed")
