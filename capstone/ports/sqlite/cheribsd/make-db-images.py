#!/usr/bin/env python3
"""Write the corrupt-database images this arm's cases open, from the corpus sources.

    make-db-images.py [--work DIR]

WHY THIS EXISTS. The Capstone arm embeds each corrupt image in the case source and serves
it through an in-domain memfs. CheriBSD has a real VFS, so its copy of the same case opens
an ordinary file instead -- `#define DBPATH "/root/corpus/<name>.db"`. Those files lived
ONLY in the guest, and nothing in the repository created them, so `deploy-corpus.sh`'s
`rm -rf /root/corpus` destroyed them and every affected case then ran against a database
sqlite3_open had just created EMPTY. That does not fail: open returns SQLITE_OK, the first
statement returns "no such table", the case prints RETURNED rc=0, and the run is scored a
clean completion. Found 2026-10-16, when a probe on case 26 reported hits=0 -- the defect
site had never been reached on that arm.

So the images are generated here, from the byte arrays in the corpus case sources, which
are in git and are the same bytes the Capstone arm uses. They are written into $WORK/out,
which deploy-corpus.sh already ships, so a wiped /root/corpus is restored by a redeploy.
"""
import argparse
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
CORPUS = HERE.parents[2] / "bug-corpora/sqlite/engine-repros"
IMG = re.compile(r"dbimg\[\]\s*=\s*\{(.*?)\};", re.S)
TAG = re.compile(r'REPRO322_MAIN\("([^"]+)"\)')
DBPATH = re.compile(r'#define\s+DBPATH\s+"([^"]+)"')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", type=pathlib.Path,
                   default=pathlib.Path.home() / "arms/sqlite/cheribsd")
    a = p.parse_args()
    out = a.work / "out"
    if not out.is_dir():
        sys.exit(f"{out} does not exist; build the corpus first")

    # What this arm's cases open, by file name.
    wanted = {}
    for src in sorted((HERE / "cases").glob("*.c")):
        m = DBPATH.search(src.read_text())
        if m:
            wanted[pathlib.PurePosixPath(m.group(1)).name] = src.name

    written, missing = [], dict(wanted)
    for case in sorted(CORPUS.glob("[0-9][0-9]_*/case.c")):
        text = case.read_text()
        img, tag = IMG.search(text), TAG.search(text)
        if not (img and tag):
            continue
        data = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", img.group(1)))
        for name in list(missing):
            if name.startswith(tag.group(1) + "."):  # <tag>.db
                pass
            elif not name.startswith(tag.group(1)):  # <tag>_r2.db and friends
                continue
            (out / name).write_bytes(data)
            written.append((name, len(data), case.parent.name))
            del missing[name]

    for name, n, src in written:
        print(f"wrote   {name:20s} {n:7d} bytes  (from {src})")
    for name, src in sorted(missing.items()):
        print(f"NO IMAGE {name:19s} -- {src} opens it, no corpus case embeds it")
    print(f"{len(written)} written, {len(missing)} without a source")
    return 1 if not written else 0


if __name__ == "__main__":
    sys.exit(main())
