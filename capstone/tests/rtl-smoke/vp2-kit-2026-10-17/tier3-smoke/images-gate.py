#!/usr/bin/env python3
"""images-gate.py <bin dir> <reference dir>... [--rows 1,2,3]: before a board boot of a runner that BUILDS its images at run
time, check that the images it built are the reference's. Every case of each reference's verdicts.jsonl that records an
image_sha256 (cut to --rows: cases named row<N>_..., for capi-style --only boots) must have that sha256 among the .dom files
in <bin dir> (normally a measure-only bake's raw/bin). Prints each missing case and a last line "<m> of <n> reference images
built". Exit 0 when all are built, 1 when any is missing, 2 when there is nothing to compare (no reference image or no built
image). Written 2026-10-22 after the compiler lane found its -g builds path-dependent: a board boot of such a build is VOID
by compare-pilot's image rule, and this gate refuses it before the board is spent."""
import hashlib
import json
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    rows = None
    if "--rows" in args:
        i = args.index("--rows")
        rows = {f"row{r}_" for r in args[i + 1].split(",")}
        del args[i:i + 2]
    if len(args) < 2:
        print(__doc__); sys.exit(2)
    bindir, refs = Path(args[0]), [Path(a) for a in args[1:]]
    built = {hashlib.sha256(p.read_bytes()).hexdigest(): p.name for p in sorted(bindir.glob("*.dom"))} if bindir.is_dir() else {}
    want = []
    for ref in refs:
        f = ref / "verdicts.jsonl"
        if not f.is_file():
            print(f"NO DATA: {f} missing"); sys.exit(2)
        for line in f.open():
            if not line.strip():
                continue
            o = json.loads(line).get("observation", {})
            case, sha = o.get("case"), o.get("image_sha256")
            if rows is not None and not any(str(case).startswith(r) for r in rows):
                continue
            if sha:
                want.append((ref.parent.parent.parent.name if ref.name == "virtual-malloc" else ref.name, case, sha))
    if not want or not built:
        print(f"NO DATA: {len(want)} reference images, {len(built)} built images in {bindir}"); sys.exit(2)
    missing = [w for w in want if w[2] not in built]
    for corpus, case, sha in missing:
        print(f"MISSING {corpus} {case}: reference image {sha[:16]} not built")
    print(f"{len(want) - len(missing)} of {len(want)} reference images built")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
