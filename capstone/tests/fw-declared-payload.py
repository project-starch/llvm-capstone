#!/usr/bin/env python3
"""fw-declared-payload.py <fw_payload.bin> <declared-dir>: the preflight's C10 check for a DECLARED Tier 3 bake.

Extracts the initramfs from the firmware itself (the gzip'd newc cpio inside the Linux Image), lists every regular file
with its sha256, and requires each (path, sha256) to be in the clean base (<declared-dir>/base.txt) or in the bake's
declared manifest (<declared-dir>/manifest.txt, plus replaced.txt if present: lines '<sha256> <path>'). Every declared
file must also be present. So no undeclared file, stale domain or otherwise, can ride along, which is C10's purpose;
the size rule for such a boot is checked by the caller. Exit 0 = GO; 1 = BLOCKED (each offending file printed);
2 = no data (no cpio found in the firmware, or an empty base or manifest)."""
import sys, zlib, hashlib
from pathlib import Path

def initramfs(fw):
    b = Path(fw).read_bytes(); i = 0
    while (i := b.find(b'\x1f\x8b\x08', i)) >= 0:
        try:
            out = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(b[i:i + 64 * 1024 * 1024])
            if out[:6] == b'070701' and b'TRAILER!!!' in out:
                return out
        except Exception:
            pass
        i += 3
    return None

def files(c):
    """Every regular file across ALL the newc archives concatenated in the initramfs (2026-11-27: the board's payloads
    carry the rootfs and the overlay as two archives in one gzip stream; stopping at the first TRAILER checked only the
    rootfs). Between archives only zero padding is allowed; anything else is malformed. Sets files.archives to the
    per-archive regular-file counts."""
    p, d, counts, dups = 0, {}, [], []
    while True:
        while p < len(c) and c[p] == 0: p += 1
        if p >= len(c): break
        k = 0; seen_here = set()
        while True:
            h = c[p:p + 110]
            if h[:6] != b'070701': print(f"NO DATA: malformed cpio at {p}"); sys.exit(2)
            f = [int(h[6 + 8 * k2:14 + 8 * k2], 16) for k2 in range(13)]
            mode, fsize, nsize = f[1], f[6], f[11]
            name = c[p + 110:p + 110 + nsize - 1].decode(); p = (p + 110 + nsize + 3) & ~3
            data = c[p:p + fsize]; p = (p + fsize + 3) & ~3
            if name == 'TRAILER!!!': break
            if (mode & 0o170000) == 0o100000:
                key, h2 = name.lstrip('./'), hashlib.sha256(data).hexdigest()
                if key in d and len(counts) > 0 and key not in seen_here: dups.append((key, d[key], h2))
                d[key] = h2; seen_here.add(key); k += 1
        counts.append(k)
    if not counts: print("NO DATA: no cpio archive in the initramfs"); sys.exit(2)
    files.archives = counts
    files.dups = dups   # (path, earlier sha, later sha): a later archive overrides an earlier one (the kernel keeps the last)
    return d

def listing(p):
    out = {}
    if Path(p).is_file():
        for l in Path(p).read_text().splitlines():
            if l.strip(): h, path = l.split(None, 1); out[path.strip().lstrip('./')] = h
    return out

if __name__ == '__main__':
    fw, dd = sys.argv[1], Path(sys.argv[2])
    c = initramfs(fw)
    if c is None: print(f"NO DATA: no initramfs cpio found inside {fw}"); sys.exit(2)
    have = files(c)
    base = listing(dd / 'base.txt'); repl = listing(dd / 'replaced.txt'); decl = {**listing(dd / 'manifest.txt'), **repl}
    if not base or not decl: print(f"NO DATA: empty base ({len(base)}) or manifest ({len(decl)}) in {dd}"); sys.exit(2)
    bad = [f"UNDECLARED {k} {v[:16]}" for k, v in sorted(have.items()) if base.get(k) != v and decl.get(k) != v]
    bad += [f"MISSING declared {k} {v[:16]}" for k, v in sorted(decl.items()) if have.get(k) != v]
    # an override across archives is checked by its effective (last) content above AND must be declared as one in replaced.txt
    bad += [f"OVERRIDE undeclared {k} {a[:16]} -> {b[:16]}" for k, a, b in getattr(files, 'dups', []) if a != b and repl.get(k) != b]
    for l in bad: print("  " + l)
    print(f"{len(files.archives)} archive(s) of {files.archives} regular files; {len(have)} files in the firmware's initramfs: {sum(base.get(k) == v for k, v in have.items())} clean base, "
          f"{sum(decl.get(k) == v and base.get(k) != v for k, v in have.items())} declared, {len(bad)} offending")
    sys.exit(1 if bad else 0)
