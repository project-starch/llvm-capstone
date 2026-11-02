#!/usr/bin/env python3
"""Page-volume census: classify every resident page of a profiled run by mapping class.

usage: prof-pages.py capstone DUMP READELF [--phase exit|first-service] [--index N] [--json]
       prof-pages.py plain    DUMP READELF [--index N] [--json]

A dump holds one block per snapshot; a run with several guest invocations (538.imagick_r: convert,
then validate) appends them in order, and --index N (from 0) picks the N-th block of the phase,
the last one by default.

DUMP is the file the launcher wrote with CAPSTONE_VM_SMAPS set (capstone/runtime/virtual/exec.c:
mapping table, minor-fault ledger, present-page bitmaps and /proc/self/smaps, at the first delegate
round and at exit) or the one profrun wrote with PROFRUN_SMAPS set (tools/capprof-validation/
profrun.c: the child's stat, per-VMA present bitmaps and smaps at its exit stop). READELF is the
output of `llvm-readelf -l -S -W` on the program's image, which places the image's pages in their
segments and sections.

Attribution is by the PRESENT-PAGE BITMAPS (/proc/<pid>/pagemap bit 63), not by VMA: adjacent
anonymous mappings with the same flags are one VMA to the kernel (the launcher's 145 heap arenas of
a gcc run are 30 VMAs, and the domain stack merges with its neighbours), and one anonymous VMA
spans the image's bss and the grant's padding. A present page is a page in this process's page
table, which is what smaps counts as Rss, so the bitmap counts and the smaps Rss sum agree; the
launcher's own pages are the Rss not covered by any registered mapping.

Classes (disjoint):
  Capstone: image-rx (code, rodata, capability tables), image-rw-file (.data, file-backed),
    image-bss, image-pad (the grant's power-of-two rounding beyond the ELF extent), heap
    (CV_MAP_HEAP arenas: the allocator's), app-mapping (CV_MAP_APPLICATION: the program's own
    mmap), alloc-meta (CV_MAP_METADATA), nested-heap, runtime-fixed:{stack,meta,exchange,startup},
    launcher-self (everything else). Plus the retired arenas' pages at their retirement (gone
    from the exit picture, counted by minflt) and the launcher's minflt ledger.
  Plain: image-r, image-rx, image-rw-file, image-bss, heap ([heap] and anonymous mappings: the
    allocator's and the program's mmap, not separable), stack, other ([vdso], [vvar]).
Nothing here counts a grant's or VMA's length as pages.
"""
import json, re, sys

PAGE = 4096

def parse_readelf(path):
    loads, sections = [], []
    for line in open(path):
        m = re.match(r"\s*LOAD\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+([RWE ]+)\s+0x", line)
        if m:
            loads.append({"offset": int(m[1], 16), "vaddr": int(m[2], 16), "filesz": int(m[4], 16),
                          "memsz": int(m[5], 16), "flags": m[6].replace(" ", "")})
            continue
        m = re.match(r"\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]{16})\s+([0-9a-f]+)\s+([0-9a-f]+)", line)
        if m and m[2] != "NULL":
            sections.append({"name": m[1], "type": m[2], "addr": int(m[3], 16), "size": int(m[5], 16)})
    return loads, sections

def parse_smaps(text):
    vmas = []
    for line in text.splitlines():
        m = re.match(r"([0-9a-f]+)-([0-9a-f]+) (\S+) (\S+) (\S+) (\S+)\s*(.*)", line)
        if m:
            vmas.append({"lo": int(m[1], 16), "hi": int(m[2], 16), "perms": m[3], "path": m[7].strip()})
            continue
        m = re.match(r"(\w+):\s+(\d+) kB", line)
        if m and vmas:
            vmas[-1][m[1]] = int(m[2]) * 1024 // PAGE
    return vmas

class Present:
    """Present pages by address, from the bitmaps (one per mapping, page i = bit i)."""
    def __init__(self, bitmaps):
        self.maps = [(base, pages, hexmap) for base, pages, hexmap in bitmaps]
    def count(self, lo, hi):
        n = 0
        for base, pages, hexmap in self.maps:
            a, b = max(lo, base), min(hi, base + pages * PAGE)
            if a >= b: continue
            for p in range((a - base) // PAGE, (b - base) // PAGE):
                if (int(hexmap[p // 4], 16) >> (p % 4)) & 1: n += 1
        return n
    def covered(self, lo, hi):
        return any(max(lo, b) < min(hi, b + n * PAGE) for b, n, _ in self.maps)

def split_dump(text, begin, end):
    i, j = text.index(begin), text.index(end)
    return text[:i], text[i + len(begin):j]

def kv_of(line):
    return dict(w.split("=", 1) for w in line.split()[1:] if "=" in w)

def capstone(dump, readelf, phase, index=-1):
    text = open(dump).read()
    blocks = ["CAPSTONE_VM_SMAPS phase=" + b for b in text.split("CAPSTONE_VM_SMAPS phase=")[1:]]
    block = [b for b in blocks if b.startswith("CAPSTONE_VM_SMAPS phase=" + phase + " ")]
    if not block or index >= len(block):
        sys.exit(f"no snapshot {index} of phase {phase} in {dump} (phases: {[b.split()[0] for b in blocks]})")
    block = block[index]
    head, smaps = split_dump(block, "CAPSTONE_VM_SMAPS_BEGIN\n", "CAPSTONE_VM_SMAPS_END\n")
    named, maps, bitmaps, ledger, header = {}, [], [], {}, {}
    retired = {"arenas": 0, "resident_pages": 0, "sizes": []}
    for line in head.splitlines():
        kv = kv_of(line)
        if line.startswith("CAPSTONE_VM_SMAPS_MINFLT"): ledger = {k: int(v) for k, v in kv.items()}
        elif line.startswith("CAPSTONE_VM_SMAPS_NAMED"): named[kv["name"]] = (int(kv["base"], 16), int(kv["bytes"]))
        elif line.startswith("CAPSTONE_VM_SMAPS_MAP "): maps.append({"base": int(kv["base"], 16), "bytes": int(kv["bytes"]), "visible": int(kv["visible"]), "kind": int(kv["kind"]), "heap": int(kv["heap"])})
        elif line.startswith("CAPSTONE_VM_SMAPS_PRESENT"): bitmaps.append((int(kv["base"], 16), int(kv["pages"]), kv["present"]))
        elif line.startswith("CAPSTONE_VM_SMAPS_RETIRED_SIZE"): retired["sizes"].append({k: int(v) for k, v in kv.items()})
        elif line.startswith("CAPSTONE_VM_SMAPS_RETIRED"): retired.update({k: int(v) for k, v in kv.items() if k != "sizes"})
        elif line.startswith("CAPSTONE_VM_SMAPS "): header = kv
    vmas = parse_smaps(smaps)
    present = Present(bitmaps)
    loads, sections = parse_readelf(readelf)
    text_seg = [l for l in loads if "E" in l["flags"]][0]
    data_seg = [l for l in loads if "W" in l["flags"]][0]
    image_lo, rx_bytes = named["image-text"]
    data_lo, data_bytes = named["image-data"]
    whole = data_seg["filesz"] & ~(PAGE - 1)                   # exec.c load(): file-backed whole pages
    data_extent = (data_seg["memsz"] + PAGE - 1) & ~(PAGE - 1)  # image_layout(): data_bytes, page-rounded
    bss_lo, pad_lo, grant_hi = data_lo + whole, data_lo + data_extent, data_lo + data_bytes
    classes = {}
    def put(cls, pages, span):
        c = classes.setdefault(cls, {"mappings": 0, "span": 0, "pages": 0})
        c["mappings"] += 1; c["span"] += span; c["pages"] += pages
    put("image-rx", present.count(image_lo, image_lo + rx_bytes), rx_bytes // PAGE)
    put("image-rw-file", present.count(data_lo, bss_lo), whole // PAGE)
    put("image-bss", present.count(bss_lo, pad_lo), (pad_lo - bss_lo) // PAGE)
    put("image-pad", present.count(pad_lo, grant_hi), (grant_hi - pad_lo) // PAGE)
    for name in ("stack", "meta", "exchange", "startup"):
        if name in named:
            b, n = named[name]; put("runtime-fixed:" + name, present.count(b, b + n), n // PAGE)
    if "nested-heap" in named:
        b, n = named["nested-heap"]; put("nested-heap", present.count(b, b + n), n // PAGE)
    arenas = []
    for m in maps:
        if not m["heap"]: continue   # the image and the fixed grants are named above
        p = present.count(m["base"], m["base"] + m["bytes"])
        cls = {0: "heap", 1: "app-mapping", 2: "alloc-meta"}.get(m["kind"], "grant-other")
        put(cls, p, m["bytes"] // PAGE)
        arenas.append({"base": m["base"], "bytes": m["bytes"], "kind": m["kind"], "present": p})
    registered = sum(c["pages"] for c in classes.values())
    total_rss = sum(v.get("Rss", 0) for v in vmas)
    # the launcher's own: Rss of the VMAs no registered mapping covers, plus whatever the covered
    # VMAs hold beyond the registered ranges (their merged neighbours)
    own_vmas = [v for v in vmas if not present.covered(v["lo"], v["hi"])]
    classes["launcher-self"] = {"mappings": len(own_vmas), "span": sum((v["hi"] - v["lo"]) // PAGE for v in own_vmas),
                                "pages": total_rss - registered}
    rx_parts = {}
    for s in sections:
        if text_seg["vaddr"] <= s["addr"] < text_seg["vaddr"] + text_seg["memsz"] and s["size"]:
            lo = image_lo + (s["addr"] - text_seg["vaddr"])
            rx_parts[s["name"]] = (present.count(lo, lo + s["size"]), (lo + s["size"] + PAGE - 1) // PAGE - lo // PAGE)
    smaps_image = {}
    for v in vmas:
        if image_lo <= v["lo"] < grant_hi:
            smaps_image[f"0x{v['lo']:x}-0x{v['hi']:x} {v['perms']}"] = {k: v.get(k, 0) for k in ("Rss", "Anonymous", "Private_Dirty", "Private_Clean", "Referenced")}
    return {"phase": phase, "header": header, "ledger": ledger, "classes": classes, "rx_sections": rx_parts,
            "retired": retired, "arenas": arenas, "named": {k: list(v) for k, v in named.items()},
            "image": {"grant_pages": (grant_hi - image_lo) // PAGE, "rx_pages": rx_bytes // PAGE, "rw_file_pages": whole // PAGE,
                      "bss_pages": (pad_lo - bss_lo) // PAGE, "pad_pages": (grant_hi - pad_lo) // PAGE},
            "image_vmas": smaps_image, "total_rss": total_rss, "vmas": len(vmas), "registered_pages": registered}

def plain(dump, readelf, index=-1):
    text = open(dump).read()
    blocks = ["PROFRUN_SMAPS phase=" + b for b in text.split("PROFRUN_SMAPS phase=")[1:]]
    if not blocks or index >= len(blocks): sys.exit(f"no snapshot {index} in {dump} ({len(blocks)} blocks)")
    head, smaps = split_dump(blocks[index], "PROFRUN_SMAPS_BEGIN\n", "PROFRUN_SMAPS_END\n")
    bitmaps, stat = [], None
    for line in head.splitlines():
        if line.startswith("PROFRUN_PRESENT"):
            kv = kv_of(line); bitmaps.append((int(kv["base"], 16), int(kv["pages"]), kv["present"]))
        elif line.startswith("PROFRUN_STAT "):
            fields = line[line.rindex(")") + 2:].split()
            stat = {"minflt": int(fields[7]), "majflt": int(fields[9])}   # state ppid pgrp session tty tpgid flags minflt cminflt majflt
    vmas = parse_smaps(smaps)
    present = Present(bitmaps)
    loads, sections = parse_readelf(readelf)
    classes = {}
    def put(cls, pages, span):
        c = classes.setdefault(cls, {"mappings": 0, "span": 0, "pages": 0})
        c["mappings"] += 1; c["span"] += span; c["pages"] += pages
    for l in loads:
        lo, file_hi, mem_hi = l["vaddr"] & ~(PAGE - 1), l["vaddr"] + l["filesz"], l["vaddr"] + l["memsz"]
        if "E" in l["flags"]: put("image-rx", present.count(lo, mem_hi), (mem_hi - lo + PAGE - 1) // PAGE)
        elif "W" not in l["flags"]: put("image-r", present.count(lo, mem_hi), (mem_hi - lo + PAGE - 1) // PAGE)
        else:
            file_end = (file_hi + PAGE - 1) & ~(PAGE - 1)
            put("image-rw-file", present.count(lo, file_end), (file_end - lo) // PAGE)
            if mem_hi > file_end: put("image-bss", present.count(file_end, mem_hi), (mem_hi - file_end + PAGE - 1) // PAGE)
    image_hi = max(l["vaddr"] + l["memsz"] for l in loads)
    for v in vmas:
        if v["lo"] < image_hi: continue   # the program's segments, attributed above
        if v["path"] == "[stack]": put("stack", v.get("Rss", 0), (v["hi"] - v["lo"]) // PAGE)
        elif v["path"] in ("", "[heap]"): put("heap", v.get("Rss", 0), (v["hi"] - v["lo"]) // PAGE)
        else: put("other:" + v["path"], v.get("Rss", 0), (v["hi"] - v["lo"]) // PAGE)
    # the bss shares its first VMA with the brk heap when brk starts right after it; the
    # segment attribution above took the bss part, so take it out of the heap VMA's Rss
    for v in vmas:
        if v["path"] == "[heap]" and v["lo"] < image_hi:
            put("heap", v.get("Rss", 0) - present.count(v["lo"], image_hi), (v["hi"] - image_hi) // PAGE)
    sec_detail = {}
    for s in sections:
        if s["size"] and s["type"] in ("PROGBITS", "NOBITS") and any(l["vaddr"] <= s["addr"] < l["vaddr"] + s["size"] + l["memsz"] for l in loads):
            sec_detail[s["name"]] = (present.count(s["addr"], s["addr"] + s["size"]), (s["addr"] + s["size"] + PAGE - 1) // PAGE - s["addr"] // PAGE)
    smaps_image = {}
    for v in vmas:
        if v["lo"] < image_hi:
            smaps_image[f"0x{v['lo']:x}-0x{v['hi']:x} {v['perms']} {v['path']}"] = {k: v.get(k, 0) for k in ("Rss", "Anonymous", "Private_Dirty", "Private_Clean", "Referenced")}
    return {"phase": "exit", "stat": stat, "classes": classes, "sections": sec_detail, "image_vmas": smaps_image,
            "total_rss": sum(v.get("Rss", 0) for v in vmas), "vmas": len(vmas)}

def table(out):
    print(f"phase {out['phase']}  vmas {out['vmas']}  total Rss pages {out['total_rss']}")
    if out.get("ledger"): print("launcher minflt ledger", out["ledger"])
    if out.get("stat"): print("child stat", out["stat"])
    print(f"{'class':24} {'maps':>5} {'span':>7} {'pages':>6}")
    for k in sorted(out["classes"]):
        c = out["classes"][k]
        print(f"{k:24} {c['mappings']:5} {c['span']:7} {c['pages']:6}")
    print(f"{'sum':24} {'':5} {'':7} {sum(c['pages'] for c in out['classes'].values()):6}")
    if out.get("retired", {}).get("arenas"):
        r = out["retired"]; print(f"retired arenas {r['arenas']} resident pages at retire {r['resident_pages']}")
        for s in sorted(r["sizes"], key=lambda s: s["bytes"]): print(f"  bytes={s['bytes']:8} count={s['count']:4} resident_pages={s['resident_pages']}")
    for key in ("rx_sections", "sections"):
        if out.get(key):
            print(f"{key} (present pages / span pages):")
            for name, (p, n) in out[key].items(): print(f"  {name:24} {p:6} / {n}")
    if out.get("image"): print("image geometry (pages)", out["image"])
    if out.get("image_vmas"):
        print("image VMAs (smaps, pages):")
        for k, v in out["image_vmas"].items(): print(f"  {k:40} {v}")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3: sys.exit(__doc__)
    kind, dump, readelf = args[:3]
    phase = args[args.index("--phase") + 1] if "--phase" in args else "exit"
    index = int(args[args.index("--index") + 1]) if "--index" in args else -1
    out = capstone(dump, readelf, phase, index) if kind == "capstone" else plain(dump, readelf, index)
    if "--json" in args: print(json.dumps(out, indent=1))
    else: table(out)
