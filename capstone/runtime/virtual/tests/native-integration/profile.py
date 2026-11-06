#!/usr/bin/env python3
"""Per-function attribution of capprof windows (kernel System.map, module ELF, user ELF).

usage: profile.py OUTDIR KERNEL_SYSTEM_MAP MODULE.ko ARM BYTES [USER_ELF]
Sums attempts over the repetitions of one arm/size, divides by calls and the number of
repetitions, and prints the functions in decreasing order. Unknown PCs are listed as such."""
import collections, pathlib, struct, sys

def elf_syms(path):
    d = pathlib.Path(path).read_bytes()
    shoff, = struct.unpack_from("<Q", d, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", d, 0x3a)
    secs = []
    for i in range(shnum):
        n, ty, fl, addr, off, size, link, info, al, es = struct.unpack_from("<IIQQQQIIQQ", d, shoff + i * shentsize)
        secs.append((n, ty, addr, off, size, link, es))
    stroff = secs[shstrndx][3]
    def name(tab_off, n):
        e = d.index(b"\0", tab_off + n); return d[tab_off + n:e].decode()
    names = [name(stroff, s[0]) for s in secs]
    syms = []
    for i, s in enumerate(secs):
        if s[1] != 2: continue
        so = secs[s[5]][3]
        for j in range(s[4] // 24):
            nm, info, other, shndx, val, size = struct.unpack_from("<IBBHQQ", d, s[3] + j * 24)
            if info & 15 == 2 and shndx < len(secs):  # STT_FUNC
                syms.append((names[shndx], val, size, name(so, nm)))
    return syms, names, secs

class Resolver:
    def __init__(self, sysmap, ko, sections, user):
        self.k = []
        for line in open(sysmap):
            a, t, n = line.split()[:3]
            if t in "tTwW": self.k.append((int(a, 16), n))
        self.k.sort()
        self.m = []
        if ko:
            syms, _, _ = elf_syms(ko)
            base = {}
            for line in open(sections):
                p = line.split()
                if len(p) == 2: base[p[0]] = int(p[1], 16)
            for sec, val, size, n in syms:
                if sec in base: self.m.append((base[sec] + val, base[sec] + val + max(size, 1), "module:" + n))
            self.m.sort()
        self.u = []
        if user:
            syms, _, _ = elf_syms(user)
            for sec, val, size, n in syms: self.u.append((val, val + max(size, 1), "user:" + n))
            self.u.sort()
    @staticmethod
    def find(tab, a):
        lo, hi = 0, len(tab)
        while lo < hi:
            mid = (lo + hi) // 2
            if tab[mid][0] <= a: lo = mid + 1
            else: hi = mid
        return tab[lo - 1] if lo else None
    def __call__(self, a):
        if a >= 0xffffffff80000000:
            r = self.find(self.k, a)
            return "kernel:" + r[1] if r else "kernel:?"
        if a >= 0xffffffff00000000 or a >= 0xffffffc000000000:
            r = self.find(self.m, a)
            if r and r[0] <= a < r[1]: return r[2]
            return "module:?"
        r = self.find(self.u, a) if self.u else None
        if r and r[0] <= a < r[1]: return r[2]
        return "user:?"

def main():
    out = pathlib.Path(sys.argv[1]); sysmap, ko, arm, size = sys.argv[2:6]
    user = sys.argv[6] if len(sys.argv) > 6 else None
    res = Resolver(sysmap, ko, out / "sections.txt", user)
    arms = [l.split() for l in (out / "arms.tsv").read_text().splitlines()]
    sel = [a for a in arms if a[1] == arm and a[2] == size]
    calls = 1 if size == "65536" else 100
    tot = collections.Counter()
    for idx, *_ in sel:
        for row in (out / "capprof" / f"window-{idx}.tsv").read_text().splitlines():
            if row.startswith("#"): continue
            f = row.split()
            tot[res(int(f[0], 16))] += int(f[3])
    n = len(sel) * calls
    allsum = sum(tot.values())
    print(f"# arm={arm} bytes={size} windows={len(sel)} calls/window={calls} total/call={allsum / n:.1f}")
    for k, v in tot.most_common(int(__import__('os').environ.get("TOP", "40"))):
        print(f"{v / n:10.1f}  {k}")
main()
