#!/usr/bin/env python3
"""Widen mruby's EMBEDDED-STRING length field so it fits 16-byte capabilities.

THE PROBLEM. mruby stores a short string inside the object header and keeps its
length in a bitfield of the object's `flags`. The capacity is derived from
`sizeof(void*)`, so it grows with the pointer while the bitfield does not:

    #define RSTRING_EMBED_LEN_MAX ((mrb_int)(sizeof(void*) * 3 - 1))   /* 47 */
    mrb_static_assert(RSTRING_EMBED_LEN_MAX < (1 << 5), "pointer size too big
                                                         for embedded string");

At 16-byte capabilities that is 47 in a five-bit field, and the build stops.
Measured on this target: pointer 16, mrb_int 4, RSTRING_EMBED_LEN_MAX 47,
sizeof(struct RString) 96 -- so 47 is the RIGHT capacity (it matches the heap
arm of the union exactly) and only the field is too narrow.

WHY WIDEN RATHER THAN SHRINK. Capping the capacity at 31 would also compile and
needs no bit reallocation, but it moves every 32..47 byte string from the object
header to the heap -- a change to the ALLOCATION PATTERN, which is the thing
being measured. Widening is also what upstream itself did: mruby 3.x replaced
the literal with `MRB_STR_EMBED_LEN_BIT`, and the CHERI port sets that to 6.

SAFE IN EVERY PINNED TREE, checked rather than assumed: `MRB_OBJECT_HEADER`
declares `uint32_t flags:21`, and in each tree the embed-length field is the
HIGHEST-numbered RString flag, with nothing above it. The script re-checks that
below and refuses if some other flag has moved in.

TWO SCHEMES, because the corpus pins nine versions spanning 2017-2026:
  new  `#define MRB_STR_EMBED_LEN_BIT 5`  -- one number, header only
  old  `MRB_STR_EMBED_LEN_MASK 0x3e0` + `SHIFT 5`, with the width ALSO written
       out as a literal in the static assert in src/string.c, so that file has
       to be shadowed too

NOTHING IS WRITTEN INTO THE MRUBY TREE. `xlang/repro/<n>/mruby` is the corpus's
pinned artefact and must stay byte-identical; patch-parser.py works the same way.
The header copy is made by the caller, this only edits it.
"""

import pathlib
import re
import sys


def fail(msg: str) -> "NoReturn":
    print(f"patch-embed-len: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if len(sys.argv) != 4:
        fail("usage: patch-embed-len.py <mruby_src> <shadow_string_h> <gen_dir>")
    mruby_src = pathlib.Path(sys.argv[1])
    shadow_h = pathlib.Path(sys.argv[2])
    gen_dir = pathlib.Path(sys.argv[3])

    if not shadow_h.is_file():
        fail(f"no shadowed header at {shadow_h}")
    text = shadow_h.read_text()

    # ---- newest scheme: MRB_STR_EMBED_LEN_BITS (plural) + explicit SHIFT ----
    # The mask is computed from it and the static assert names it symbolically,
    # so the header alone is enough here -- no shadowed src/string.c.
    mb = re.search(r"^#define MRB_STR_EMBED_LEN_BITS\s+(\d+)\s*$", text, re.M)
    if mb:
        bits = int(mb.group(1))
        if bits >= 6:
            return 0
        if bits != 5:
            fail(f"MRB_STR_EMBED_LEN_BITS is {bits}, expected 5 or 6")
        msh = re.search(r"^#define MRB_STR_EMBED_LEN_SHIFT\s+(\d+)\s*$", text, re.M)
        if not msh:
            fail("MRB_STR_EMBED_LEN_BITS without a matching SHIFT")
        shift = int(msh.group(1))
        new_bit = 1 << (shift + bits)
        for name, val in re.findall(r"^#define (MRB_STR_[A-Z_]+)\s+(\d+)\s*$", text, re.M):
            if int(val) & new_bit:
                fail(f"bit {shift + bits} is already taken by {name}")
        if shift + bits >= 21:
            fail(f"bit {shift + bits} is outside the 21-bit flags field")
        shadow_h.write_text(text.replace(mb.group(0),
                                         "#define MRB_STR_EMBED_LEN_BITS 6"))
        print(f"embed-len: MRB_STR_EMBED_LEN_BITS 5 -> 6 (claims bit "
              f"{shift + bits})", file=sys.stderr)
        return 0

    # ---- 3.x scheme: one macro, header only ---------------------------------
    m = re.search(r"^#define MRB_STR_EMBED_LEN_BIT\s+(\d+)\s*$", text, re.M)
    if m:
        bit = int(m.group(1))
        if bit >= 6:
            return 0                      # already wide enough (mruby-purecap)
        if bit != 5:
            fail(f"MRB_STR_EMBED_LEN_BIT is {bit}, expected 5 or 6")
        shadow_h.write_text(text.replace(m.group(0),
                                         "#define MRB_STR_EMBED_LEN_BIT 6"))
        print("embed-len: MRB_STR_EMBED_LEN_BIT 5 -> 6", file=sys.stderr)
        return 0

    # ---- old scheme: hex mask + shift, plus a literal in the assert ---------
    mm = re.search(r"^#define MRB_STR_EMBED_LEN_MASK\s+0x([0-9a-fA-F]+)\s*$", text, re.M)
    ms = re.search(r"^#define MRB_STR_EMBED_LEN_SHIFT\s+(\d+)\s*$", text, re.M)
    if not mm or not ms:
        fail("neither MRB_STR_EMBED_LEN_BIT nor a MASK/SHIFT pair found; the "
             "embedded-string layout is a third shape and needs re-deriving")

    mask, shift = int(mm.group(1), 16), int(ms.group(1))
    width = bin(mask >> shift).count("1")
    if mask != ((1 << width) - 1) << shift:
        fail(f"MASK 0x{mask:x} is not {width} contiguous bits at shift {shift}")
    if width >= 6:
        return 0
    if width != 5:
        fail(f"embed-length field is {width} bits, expected 5 or 6")

    # The bit about to be claimed must not belong to another flag.
    new_bit = 1 << (shift + width)
    for name, val in re.findall(r"^#define (MRB_STR_[A-Z_]+)\s+(\d+)\s*$", text, re.M):
        if int(val) & new_bit:
            fail(f"bit {shift + width} is already taken by {name}; widening "
                 f"the embed length here would corrupt it")
    if shift + width >= 21:
        fail(f"bit {shift + width} is outside the 21-bit flags field")

    new_mask = mask | new_bit
    text = text.replace(mm.group(0),
                        f"#define MRB_STR_EMBED_LEN_MASK 0x{new_mask:x}")
    shadow_h.write_text(text)

    # The width is written out a second time, as a literal, in the static
    # assert. Miss it and the header is wider while the build still stops.
    src_c = mruby_src / "src" / "string.c"
    if not src_c.is_file():
        fail(f"no {src_c}")
    ctext = src_c.read_text()
    old_assert = f"RSTRING_EMBED_LEN_MAX < (1 << {width})"
    if ctext.count(old_assert) != 1:
        fail(f"expected exactly one {old_assert!r} in {src_c}, found "
             f"{ctext.count(old_assert)}")
    gen_dir.mkdir(parents=True, exist_ok=True)
    out = gen_dir / "string.c"
    out.write_text(ctext.replace(old_assert,
                                 f"RSTRING_EMBED_LEN_MAX < (1 << {width + 1})"))

    print(f"embed-len: MASK 0x{mask:x} -> 0x{new_mask:x} ({width} -> {width + 1} "
          f"bits), assert widened in a shadowed src/string.c", file=sys.stderr)
    print(f"SHADOW_SRC={out}")           # stdout: consumed by the build script
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
