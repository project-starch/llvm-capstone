#!/usr/bin/env python3
"""Which sizes does fix (a) actually penalise? No allocation trace is recorded anywhere in
runtime/virtual/results (only aggregate live_bytes/cached_bytes/node_bytes), so instead of
synthesising a distribution, characterise the penalised SET and check the sizes applications
actually use."""
import sweep as S

promoted = {}
for n in range(1, S.MMAP_THRESHOLD):
    r = S.repr_round(n)
    a, b = S.size_to_class(n), S.size_to_class(r)
    if b is None or b == a:
        continue
    s0, s1 = S.UNIT*S.SIZE_CLASSES[a], S.UNIT*S.SIZE_CLASSES[b]
    promoted[n] = (s1 - s0, 100.0*(s1 - s0)/n)

print("penalised sizes: %d of %d" % (len(promoted), S.MMAP_THRESHOLD - 1))
print("every penalised n sits in the last %d bytes below its class's slot end?" % S.UNIT)
# a penalised n is one where repr_round(n) crosses out of the class -- i.e. n is within
# (granule-1) of the class's own capacity
edge = all(S.UNIT*S.SIZE_CLASSES[S.size_to_class(n)] - S.IB - n < S.repr_granule(n)
           for n in promoted)
print("  -> %s" % edge)

print("\npenalty distribution (%% of the request):")
buckets = {}
for n, (extra, pct) in promoted.items():
    k = int(pct // 5) * 5
    buckets[k] = buckets.get(k, 0) + 1
for k in sorted(buckets):
    print("   %2d-%2d%%: %5d sizes" % (k, k+5, buckets[k]))

print("\nsizes applications actually request -- are they penalised?")
common = []
for e in range(4, 18):
    common.append(1 << e)
for k in (1, 2, 3, 4, 8, 16, 32):
    common.append(k*4096)
common += [1000, 10000, 100000, 65536, 131072//2]
common = sorted(set(c for c in common if 0 < c < S.MMAP_THRESHOLD))
pen = [c for c in common if c in promoted]
print("  checked %d common sizes (powers of two, page multiples, round decimals)" % len(common))
print("  penalised: %s" % (pen if pen else "NONE"))

print("\nworst 6 penalties:")
for n, (extra, pct) in sorted(promoted.items(), key=lambda kv: -kv[1][1])[:6]:
    print("   n=%-7d +%-6d bytes  (%.1f%%)   granule %d" % (n, extra, pct, S.repr_granule(n)))
