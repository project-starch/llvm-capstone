#!/usr/bin/env python3
"""Make a layout-controlled view of an arm's SDK: the same SDK, linked with heap.c's code LAST, in fixed regions.

  layout-sdk.py --sdk SDK --out DIR [--plant-shift N]

The arms differ only in the runtime's heap.c.obj (recipes/sdks.sh proves it). capstone-cc links the runtime archive
whole and FIRST, and link.ld places every .text input in link order, so heap.c's code sits early in .text and every
benchmark function after it moves by the difference of the two heap.c sizes (505.mcf_r's main: 0x212ec in one arm,
0x21414 in the other). That is a code-placement difference between the arms, not the heap's.

DIR gets:
- link.ld: the SDK's own link.ld with every allocated input section of heap.c.obj moved out of its usual place into a
  region of FIXED size at the end of its output section: text at the end of .text, data at the end of .data, bss at
  the end of .bss, and rodata in an output section of its own (.rodata.heap, right after .rodata: lld merges
  SHF_MERGE strings across a whole output section, so heap.c's strings inside .rodata would change the size of the
  shared string pool). Every other input section keeps its order and so sits at the same address in both arms; the
  regions start at the same address in both arms and only heap.c's code inside them differs. Each region ends with a
  link-time ASSERT, so a heap.c that outgrows it fails the link instead of moving what follows.
- sdk.json: the SDK's, with only `script` changed to DIR/link.ld;
- capstone-cc: a symlink to the SDK's. capstone-cc reads the sdk.json of the directory it is invoked from
  (Path(argv[0]).absolute(), which keeps the symlink's directory), so DIR/capstone-cc compiles exactly as the SDK
  does and links with DIR/link.ld;
- layout.json: the regions, the heap.c.obj hash and its section sizes, and the sha256 of every file above.

The regions are constants, not computed from a pair: every arm built this way gets the same layout around its heap,
whatever SDK it is built against, as long as heap.c fits (heap.c.obj's sizes are checked here, and again by the
ASSERTs at link time).

--plant-shift N inserts N bytes after .text.entry. It exists only for layout-check.py's negative control: an image
linked with it must FAIL the check against an unshifted one.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HEAP_MEMBER = "heap.c.obj"
RUNTIME_ARCHIVE = "libapplication-runtime.a"
PATTERN = f"*{RUNTIME_ARCHIVE}:{HEAP_MEMBER}"
ALIGN = 64  # each region starts on a cache line
# Region sizes in bytes. heap.c.obj measured 2026-10-30: text 12668 (sdk-T1), 13060 (sdk-T3), 14536 (sdk-heapfix-T1),
# 14928 (sdk-heapfix-T3); bss 65744 in all four; rodata 0 (T1, T3) and 75 (heapfix); no data.
REGIONS = {"text": 0x8000, "rodata": 0x1000, "data": 0x1000, "bss": 0x18000}
CLASSES = {
    "text": (".text",),
    "rodata": (".rodata", ".srodata"),
    "data": (".data", ".sdata"),
    "bss": (".bss", ".sbss"),
}


def archive_member(path, member):
    """The one member of an ar archive with this name (GNU or BSD long-name table)."""
    data = Path(path).read_bytes()
    if data[:8] != b"!<arch>\n":
        sys.exit(f"layout-sdk: {path} is not an ar archive")
    pos, names, found = 8, b"", []
    while pos + 60 <= len(data):
        hdr = data[pos:pos + 60]
        size = int(hdr[48:58].decode().strip())
        body = data[pos + 60:pos + 60 + size]
        name = hdr[:16].decode().strip()
        pos += 60 + size + (size & 1)
        if name in ("/", "/SYM64/"):
            continue
        if name == "//":
            names = body
            continue
        if name.startswith("/") and name[1:].isdigit():
            off = int(name[1:])
            name = names[off:names.index(b"\n", off)].decode()
        if name.rstrip("/") == member:
            found.append(body)
    if len(found) != 1:
        sys.exit(f"layout-sdk: {path} has {len(found)} members named {member}, expected exactly one")
    return found[0]


def classify(name):
    for cls, prefixes in CLASSES.items():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            return cls
    return None


def heap_sections(obj, llvm_bin):
    """Allocated sections of heap.c.obj by class, with their packed sizes; refuses any it has no region for."""
    with tempfile.NamedTemporaryFile(suffix=".o") as f:
        f.write(obj)
        f.flush()
        out = subprocess.run([str(llvm_bin / "llvm-readelf"), "-S", "--wide", f.name], capture_output=True,
                             text=True, check=True).stdout
        nm = subprocess.run([str(llvm_bin / "llvm-nm"), f.name], capture_output=True, text=True, check=True).stdout
    if any(line.split()[-2:-1] == ["C"] for line in nm.splitlines() if line.split()):
        sys.exit("layout-sdk: heap.c.obj has COMMON symbols, which no region holds")
    sizes, other = {c: 0 for c in CLASSES}, []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("[") or "]" not in line:
            continue
        rest = line.split("]", 1)[1].split()
        # Name Type Address Off Size ES Flg Lk Inf Al (Flg may be empty)
        if len(rest) < 9 or not rest[0].startswith("."):
            continue
        name, size, align = rest[0], int(rest[4], 16), int(rest[-1])
        flags = rest[6] if len(rest) == 10 else ""
        if "A" not in flags or size == 0:
            continue
        cls = classify(name)
        if cls is None:
            other.append(name)
            continue
        sizes[cls] = (sizes[cls] + align - 1) // align * align + size
    if other:
        sys.exit(f"layout-sdk: heap.c.obj has allocated sections no region holds: {other}")
    return sizes


def replace_once(text, old, new):
    n = text.count(old)
    if n != 1:
        sys.exit(f"layout-sdk: the SDK's link.ld has {n} copies of the block to replace, expected 1:\n{old}")
    return text.replace(old, new)


def region(cls, inputs):
    sym = f"__layout_heap_{cls}"
    return (f"    . = ALIGN({ALIGN});\n"
            f"    {sym}_start = .;\n"
            f"    {PATTERN}({inputs})\n"
            f"    {sym}_end = .;\n"
            f"    . = {sym}_start + LAYOUT_HEAP_{cls.upper()}_REGION;\n")


def script(link_ld, plant):
    t = Path(link_ld).read_text()
    x = f"EXCLUDE_FILE({PATTERN}) "
    t = replace_once(t, "    *(.text.entry)\n    *(.text .text.*)\n  } :text\n",
                     "    *(.text.entry)\n" + (f"    . = . + {plant};\n" if plant else "") +
                     f"    *({x}.text .text.*)\n" + region("text", ".text .text.*") + "  } :text\n")
    t = replace_once(t, "    *(.srodata .srodata.*)\n    *(.rodata .rodata.*)\n  } :text\n",
                     f"    *({x}.srodata .srodata.*)\n    *({x}.rodata .rodata.*)\n  }} :text\n\n"
                     "  /* heap.c's read-only data, in an output section of its own: lld merges SHF_MERGE strings\n"
                     "     across one output section, so inside .rodata they would change the shared pool's size.\n"
                     "     BYTE(0) keeps the section loaded data (PROGBITS) when heap.c has none. */\n"
                     "  .rodata.heap : ALIGN(" + str(ALIGN) + ") {\n    BYTE(0)\n" +
                     region("rodata", ".srodata .srodata.* .rodata .rodata.*") + "  } :text\n")
    t = replace_once(t, "    *(.sdata .sdata.*)\n    *(.data .data.*)\n  } :data\n",
                     f"    *({x}.sdata .sdata.*)\n    *({x}.data .data.*)\n" +
                     region("data", ".sdata .sdata.* .data .data.*") + "  } :data\n")
    t = replace_once(t, "    *(.sbss .sbss.*)\n    *(.bss .bss.*)\n    *(COMMON)\n    . = ALIGN(16);\n  } :data\n",
                     f"    *({x}.sbss .sbss.*)\n    *({x}.bss .bss.*)\n    *(COMMON)\n" +
                     region("bss", ".sbss .sbss.* .bss .bss.*") + "    . = ALIGN(16);\n  } :data\n")
    head = (f"/* Generated by capstone/ports/spec-cpu2017/tools/layout-sdk.py from {link_ld}:\n"
            f"   heap.c.obj last in .text, .data and .bss and alone in .rodata.heap, each in a fixed region"
            f"{f'; PLANTED SHIFT {plant} bytes after .text.entry (negative control only)' if plant else ''}. */\n" +
            "".join(f"LAYOUT_HEAP_{c.upper()}_REGION = {v:#x};\n" for c, v in REGIONS.items()))
    tail = "\n" + "".join(
        f"ASSERT(__layout_heap_{c}_end <= __layout_heap_{c}_start + LAYOUT_HEAP_{c.upper()}_REGION, "
        f"\"heap.c {c} exceeds its layout region (tools/layout-sdk.py REGIONS)\")\n" for c in REGIONS)
    return head + t + tail


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdk", type=Path, required=True, help="the arm's SDK directory (sdk.json, capstone-cc)")
    p.add_argument("--out", type=Path, required=True, help="new directory")
    p.add_argument("--plant-shift", type=int, default=0, help="negative control only: N bytes after .text.entry")
    a = p.parse_args()
    sdk = a.sdk.absolute()
    cfg = json.loads((sdk / "sdk.json").read_text())
    if Path(cfg["runtime"]).name != RUNTIME_ARCHIVE:
        sys.exit(f"layout-sdk: the runtime is {cfg['runtime']}, the pattern {PATTERN} would not match it")
    llvm_bin = Path(os.environ.get("CAPSTONE_LLVM_BIN") or Path(cfg["ld"]).parent)
    obj = archive_member(cfg["runtime"], HEAP_MEMBER)
    sizes = heap_sections(obj, llvm_bin)
    over = {c: (sizes[c], REGIONS[c]) for c in REGIONS if sizes[c] > REGIONS[c]}
    if over:
        sys.exit(f"layout-sdk: heap.c.obj outgrows its regions (size, region): {over}")
    if a.out.exists():
        sys.exit(f"layout-sdk: {a.out} exists")
    a.out.mkdir(parents=True)
    out = a.out.absolute()
    (out / "link.ld").write_text(script(cfg["script"], a.plant_shift))
    derived = dict(cfg, script=str(out / "link.ld"))
    (out / "sdk.json").write_text(json.dumps(derived, indent=2) + "\n")
    os.symlink(sdk / "capstone-cc", out / "capstone-cc")
    rec = {
        "sdk": str(sdk), "sdk_json_sha256": sha(sdk / "sdk.json"), "sdk_script": cfg["script"],
        "sdk_script_sha256": sha(cfg["script"]), "capstone_cc_sha256": sha(sdk / "capstone-cc"),
        "heap_obj_sha256": hashlib.sha256(obj).hexdigest(), "heap_obj_bytes": sizes,
        "regions": REGIONS, "region_align": ALIGN, "plant_shift": a.plant_shift,
        "link_ld_sha256": sha(out / "link.ld"), "derived_sdk_json_sha256": sha(out / "sdk.json"),
    }
    (out / "layout.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"layout-sdk {out}: heap.c.obj {rec['heap_obj_sha256'][:16]} "
          + " ".join(f"{c} {sizes[c]}/{REGIONS[c]}" for c in REGIONS)
          + f" link.ld {rec['link_ld_sha256'][:16]}" + (f" PLANT {a.plant_shift}" if a.plant_shift else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
