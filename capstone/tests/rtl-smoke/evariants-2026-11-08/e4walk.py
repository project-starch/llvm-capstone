import re, sys, statistics
# walks = contiguous runs of switcher "Save idx"/"Restore idx" dprints; timestamp = last numeric-prefixed line before
for f in sys.argv[1:]:
    t = 0; walks = []; cur = None
    for ln in open(f, errors="replace"):
        m = re.match(r"(\d+) ", ln)
        if m: t = int(m.group(1)); continue
        if ln.startswith(("Save idx", "Restore idx")):
            if cur and t - cur[1] <= 60: cur[1] = t; cur[2] += 1
            else:
                if cur: walks.append(cur)
                cur = [t, t, 1]
    if cur: walks.append(cur)
    d = [w[1] - w[0] for w in walks]; n = [w[2] for w in walks]
    print(f.split("/")[-1][:30], "walks", len(walks), "dur median", statistics.median(d) if d else None,
          "min", min(d) if d else None, "max", max(d) if d else None, "ids median", statistics.median(n) if n else None)
    from collections import Counter
    print("  (ids,dur) top:", Counter(zip(n, [x//10*10 for x in d])).most_common(6))
