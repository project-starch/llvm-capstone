#!/usr/bin/env python3
"""tl-verdict.py <transcript>: the tag-leak probe's verdict (results/2026-11-05-nosweep/PREREG.md, N1/N2).
Reads the probe's lines after the transcript's last OpenSBI banner (a board console replays the previous boot first).
N1 (gates the no-sweep runs): in the sweep=2 pair, the fill read back a tag in every sampled granule, the scan's block
   is loaded (recycled >= half its pages, no PFN unread), the scan read no tag (tagged = tagged_recycled = 0), and its
   own tag reads back (ctl = 1). The sweep=3 pair must hold the same, as the reference arm.
N2 (reported, gates nothing): ETXPROBE running=DENIED and idle=WRITABLE in both pairs.
Exit 0 if N1 holds, 1 if it fails, 2 if the probe's lines are missing."""
import re, sys
t = open(sys.argv[1], errors="replace").read()
if "OpenSBI" in t: t = t[t.rfind("OpenSBI"):]
modes = re.split(r"TAGLEAK-MODE ", t)[1:]
res = {}
for blk in modes:
    m = re.match(r"sweep=(\d+)", blk); s = m and m[1]
    f = re.search(r"TAGLEAK fill mb=\d+ pages=(\d+) sampled_tagged=(\d+) of (\d+) nopfn=(\d+)", blk)
    c = re.search(r"TAGLEAK scan mb=\d+ pages=(\d+) tagged=(\d+) tagged_pages=\d+ fill_pfns=\d+ recycled=(\d+) tagged_recycled=(\d+) nopfn=(\d+) ctl=(\d+)", blk)
    run = re.search(r"ETXPROBE running=(\S+)", blk); idle = re.search(r"ETXPROBE idle=(\S+)", blk)
    if not (s and f and c): continue
    fp, fs, ft, fn = map(int, f.groups()); cp, ct, cr, ctr, cn, cc = map(int, c.groups())
    ok = fs == ft and fn == 0 and cn == 0 and cr >= cp // 2 and ct == 0 and ctr == 0 and cc == 1
    res[s] = ok
    print(f"sweep={s}: fill tagged {fs}/{ft} nopfn {fn} | scan pages {cp} recycled {cr} tagged {ct} tagged_recycled {ctr} nopfn {cn} ctl {cc}"
          f" -> N1 {'holds' if ok else 'FAILS'} | ETX running={run and run[1]} idle={idle and idle[1]}"
          f" -> N2 {'holds' if run and idle and run[1] == 'DENIED' and idle[1] == 'WRITABLE' else 'fails'}")
if set(res) != {"2", "3"}: print(f"NO DATA: pairs read {sorted(res)}"); sys.exit(2)
print("N1 " + ("HOLDS" if all(res.values()) else "FAILS")); sys.exit(0 if all(res.values()) else 1)
