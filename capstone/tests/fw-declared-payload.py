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
    p, d = 0, {}
    while True:
        h = c[p:p + 110]
        if h[:6] != b'070701': raise SystemExit(f"NO DATA: malformed cpio at {p}")
        f = [int(h[6 + 8 * k:14 + 8 * k], 16) for k in range(13)]
        mode, fsize, nsize = f[1], f[6], f[11]
        name = c[p + 110:p + 110 + nsize - 1].decode(); p = (p + 110 + nsize + 3) & ~3
        data = c[p:p + fsize]; p = (p + fsize + 3) & ~3
        if name == 'TRAILER!!!': return d
        if (mode & 0o170000) == 0o100000: d[name.lstrip('./')] = hashlib.sha256(data).hexdigest()

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
    base = listing(dd / 'base.txt'); decl = {**listing(dd / 'manifest.txt'), **listing(dd / 'replaced.txt')}
    if not base or not decl: print(f"NO DATA: empty base ({len(base)}) or manifest ({len(decl)}) in {dd}"); sys.exit(2)
    bad = [f"UNDECLARED {k} {v[:16]}" for k, v in sorted(have.items()) if base.get(k) != v and decl.get(k) != v]
    bad += [f"MISSING declared {k} {v[:16]}" for k, v in sorted(decl.items()) if have.get(k) != v]
    for l in bad: print("  " + l)
    print(f"{len(have)} files in the firmware's initramfs: {sum(base.get(k) == v for k, v in have.items())} clean base, "
          f"{sum(decl.get(k) == v and base.get(k) != v for k, v in have.items())} declared, {len(bad)} offending")
    sys.exit(1 if bad else 0)
