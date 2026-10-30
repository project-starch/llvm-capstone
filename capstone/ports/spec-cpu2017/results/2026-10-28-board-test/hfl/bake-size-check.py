#!/usr/bin/env python3
"""bake-size-check.py --ref-image IMAGE --batch DIR [--recompress-only]: the firmware a board bake of DIR would make,
without the buildroot or the board.

A bake puts a batch directory (boardvm's: spec/ = the make-stage stage, control/, capstone-vexec, capstone_vm.ko,
gate.sh, ctl.sh) under test-domains/vbatch/ of the board rootfs, and the kernel build embeds the rootfs as a gzip'd
newc cpio in the Image. This script takes a REAL Image of the same kit (a past boot's guest/image/Image), extracts its
initramfs, removes that boot's test-domains/vbatch/ entries, inserts DIR's files in their place (same position, same
newc format, mode 0755 for an executable file as board-run.sh does), recompresses with gzip -9 -n, and applies the
Image layout: Image = 2 MiB * ceil((6,349,144 + gz + 35,320) / 2 MiB) + 4,884,992; firmware = Image + 2,097,160.
--recompress-only recompresses the reference initramfs unchanged: its length must equal the original's for the method
to be exact (the positive control). Exit 0; 2 on no data.
"""
import argparse
import gzip
import os
import sys
import zlib
from pathlib import Path

MiB2 = 2 * 1024 * 1024


def find_initramfs(b):
    i = 0
    while (i := b.find(b"\x1f\x8b\x08", i)) >= 0:
        try:
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            out = d.decompress(b[i:i + 96 * 1024 * 1024])
            if out[:6] == b"070701" and b"TRAILER!!!" in out:
                return i, len(b[i:i + 96 * 1024 * 1024]) - len(d.unused_data), out
        except Exception:
            pass
        i += 3
    return None


def entries(c):
    p, out = 0, []
    while True:
        h = c[p:p + 110]
        if h[:6] != b"070701":
            sys.exit(f"NO DATA: malformed cpio at {p}")
        f = [int(h[6 + 8 * k:14 + 8 * k], 16) for k in range(13)]
        fsize, nsize = f[6], f[11]
        name = c[p + 110:p + 110 + nsize - 1].decode()
        end = (((p + 110 + nsize + 3) & ~3) + fsize + 3) & ~3
        out.append((name, c[p:end], f))
        p = end
        if name == "TRAILER!!!":
            return out


def newc(name, data, mode, ino, mtime):
    nb = name.encode() + b"\0"
    hdr = b"070701" + b"".join(b"%08X" % v for v in (ino, mode, 0, 0, 1, mtime, len(data), 0, 0, 0, 0, len(nb), 0))
    rec = hdr + nb
    rec += b"\0" * ((4 - len(rec) % 4) % 4) + data
    return rec + b"\0" * ((4 - len(rec) % 4) % 4)


def layout(gz):
    img = MiB2 * -(-(6_349_144 + gz + 35_320) // MiB2) + 4_884_992
    return img, img + 2_097_160


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-image", type=Path, required=True)
    ap.add_argument("--batch", type=Path)
    ap.add_argument("--recompress-only", action="store_true")
    a = ap.parse_args()
    b = a.ref_image.read_bytes()
    r = find_initramfs(b)
    if not r:
        print(f"NO DATA: no initramfs in {a.ref_image}")
        return 2
    off, glen, cpio = r
    if a.recompress_only:
        gz = len(gzip.compress(cpio, compresslevel=9, mtime=0))
        img, fw = layout(glen)
        print(f"reference {a.ref_image}: Image {len(b):,} (layout {img:,}), initramfs gz {glen:,} at {off:,}; "
              f"recompressed -9: {gz:,} ({gz - glen:+,})")
        return 0
    es = entries(cpio)
    keep = [e for e in es if not (e[0] == "test-domains/vbatch" or e[0].startswith("test-domains/vbatch/"))]
    first = next(i for i, e in enumerate(es) if e[0].startswith("test-domains/vbatch"))
    pos = sum(1 for e in es[:first] if e in keep)
    mtime = es[first][2][5]
    ino = max(e[2][0] for e in es) + 1
    new, dirs = [], set()
    for f in sorted(a.batch.rglob("*")):
        rel = "test-domains/vbatch/" + f.relative_to(a.batch).as_posix()
        if f.is_dir():
            continue
        parts = rel.split("/")
        for k in range(2, len(parts)):
            d = "/".join(parts[:k])
            if d not in dirs and d != "test-domains":
                dirs.add(d)
                new.append(newc(d, b"", 0o040755, ino, mtime)); ino += 1
        data = f.read_bytes()
        mode = 0o100755 if (f.name.endswith(".sh") or f.name == "capstone-vexec" or os.access(f, os.X_OK)) else 0o100644
        new.append(newc(rel, data, mode, ino, mtime)); ino += 1
    body = b"".join(e[1] for e in keep[:pos]) + b"".join(new) + b"".join(e[1] for e in keep[pos:])
    gz = len(gzip.compress(body, compresslevel=9, mtime=0))
    img, fw = layout(gz)
    print(f"batch {a.batch}: {len(new)} entries, cpio {len(body):,}, initramfs gz {gz:,} -> Image {img:,}, "
          f"firmware {fw:,}; next 2 MiB step at gz {MiB2 * -(-(6_349_144 + gz + 35_320) // MiB2) - 6_349_144 - 35_320:,} "
          f"({MiB2 * -(-(6_349_144 + gz + 35_320) // MiB2) - 6_349_144 - 35_320 - gz:,} B of room)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
