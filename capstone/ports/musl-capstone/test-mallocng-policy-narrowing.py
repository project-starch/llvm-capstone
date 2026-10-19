#!/usr/bin/env python3
"""Negative-test the ONE narrowing in check-mallocng-policy.py.

The guard now accepts a size_classes table that differs from upstream musl 1.2.5 in exactly seven
entries at exactly seven values. A gate that has never refused anything is not a passing gate, so
this exercises it in both directions: the authorised table must be accepted, and six other tables
-- including the UNEDITED upstream one, since the edits are required rather than merely tolerated
-- must be refused.

    python3 test-mallocng-policy-narrowing.py        # exits non-zero on any wrong verdict
"""
import importlib.util, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("policy", HERE / "check-mallocng-policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)

UPSTREAM = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 31, 36, 42, 50, 63,
    72, 84, 102, 127, 146, 170, 204, 255, 292, 340, 409, 511,
    584, 682, 818, 1023, 1169, 1364, 1637, 2047, 2340, 2730, 3276, 4095,
    4680, 5460, 6552, 8191,
]
EDITS = policy.ALLOWED_SIZE_CLASS_EDITS

def table(values):
    return "const uint16_t size_classes[] = {\n\t" + ", ".join(str(v) for v in values) + ",\n};\n"

def verdict(values):
    return policy.size_classes_ok(table(UPSTREAM), table(values))

def apply(edits):
    out = list(UPSTREAM)
    for i, v in edits.items():
        out[i] = v
    return out

fails = []
def expect(want, values, what):
    got = verdict(values)
    print("  %-7s %s" % ("accept" if got else "REFUSE", what))
    if got != want:
        fails.append(what)

# the authorised table
expect(True, apply(EDITS), "the seven authorised entries at their authorised values")

# and every near miss
expect(False, UPSTREAM, "upstream unedited (the edits are REQUIRED, not tolerated)")
eighth = dict(EDITS); eighth[44] = 4681
expect(False, apply(eighth), "an EIGHTH edited entry")
six = {k: v for k, v in list(EDITS.items())[:6]}
expect(False, apply(six), "only six of the seven")
wrong = dict(EDITS); wrong[43] = 4097
expect(False, apply(wrong), "a seventh entry at the WRONG value (4097 for 4096)")
lowered = dict(EDITS); lowered[36] = 1168
expect(False, apply(lowered), "an authorised entry LOWERED below upstream")
short = apply(EDITS)[:-1]
expect(False, short, "a table of the wrong length")

# and that it pins the values rather than just the indices
shuffled = {36: 1638, 38: 1170, 39: 2048, 41: 2732, 43: 4096, 45: 5464, 47: 8192}
expect(False, apply(shuffled), "the right indices with two values swapped")

if fails:
    print("\n%d wrong verdict(s): %s" % (len(fails), "; ".join(fails)))
    sys.exit(1)
print("\nthe narrowing accepts exactly one table and refuses every near miss")
