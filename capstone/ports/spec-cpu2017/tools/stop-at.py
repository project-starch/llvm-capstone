#!/usr/bin/env python3
"""Work-bounded profiling windows: resolve a function's guest address for capprof, and judge
the run that stopped there (tools/profile.md, "Work-bounded slices").

usage: stop-at.py resolve --nm NM --kind capstone|plain --exe EXE --symbol NAME --count K
                          [--start-symbol NAME --start-count K]
         prints the JSON record (OUT/stop-at.json): the symbol's link address and first
         instruction bytes, and "plugin_args", the capprof arguments that stop at its K-th entry
       stop-at.py finish OUT
         after the run: checks OUT/capprof/stopped against OUT/stop-at.json, writes OUT/slice.json,
         appends the stopped invocation's image record (capstone arm), and turns OUT/result.json's
         verdict into SLICE or FAIL; prints the verdict and the reasons

Where the instruction is in the guest:
  plain     a static ET_EXEC runs at its link addresses: the site is the symbol's address.
  capstone  the launcher maps the domain image at a base it reserves at run time (exec.c
            reserve(): a fresh mapping of the image's grant size, rounded to a power of two R,
            aligned to R), so the instruction is at base + (link - low) with base a multiple of R;
            capprof gets the offset link - low with slide_align=R and matches the bytes. The
            stopped address then gives the base, and with it the image record the launcher would
            have written at exit (it never exits: the run ends inside the domain).
"""
import json, os, struct, subprocess, sys


def die(msg):
    print("stop-at:", msg, file=sys.stderr)
    sys.exit(2)


def elf(path):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        die(f"{path} is not a 64-bit ELF")
    e_type, = struct.unpack_from("<H", data, 16)
    phoff, = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    loads = []
    for i in range(phnum):
        p_type, p_flags, p_offset, p_vaddr, _, p_filesz, p_memsz, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1:
            loads.append((p_vaddr, p_memsz, p_flags, data[p_offset:p_offset + p_filesz]))
    return e_type, loads


def symbol(nm, exe, name, size=False):
    out = subprocess.run([nm, "-n", "-S", "--defined-only", exe], capture_output=True, text=True, check=True).stdout
    found = {}
    for p in (l.split() for l in out.splitlines()):
        if len(p) == 4 and p[2] in "TtWw" and p[3] == name:
            found[int(p[0], 16)] = int(p[1], 16)
        elif len(p) == 3 and p[1] in "TtWw" and p[2] == name:
            found.setdefault(int(p[0], 16), 0)
    if len(found) != 1:
        die(f"{exe}: {len(found)} text symbols named {name} ({[hex(a) for a in found]}); need exactly one")
    (addr, sz), = found.items()
    return (addr, sz) if size else addr


def first_insn(loads, va):
    for v, m, flags, b in loads:
        if flags & 1 and v <= va < v + len(b):
            o = va - v
            length = 2 if b[o] & 3 != 3 else 4
            return length, int.from_bytes(b[o:o + length], "little")
    die(f"{va:#x} is in no executable segment")


def transfers_control(w, length):
    """As capprof.c: a branch, jump, call, return, ecall/ebreak or Capstone jump has no
    sequential successor, so its entries cannot be told from fault retries."""
    if length == 2:
        q, f3 = w & 3, (w >> 13) & 7
        return (q == 1 and f3 in (5, 6, 7)) or (q == 2 and f3 == 4 and not (w >> 2) & 31)
    op, f3, f7 = w & 0x7f, (w >> 12) & 7, w >> 25
    return op in (0x63, 0x67, 0x6f, 0x73) or (op == 0x5b and (f3 in (5, 6) or (f3 == 1 and f7 in (0x20, 0x21, 0x0d, 0x22, 0x23, 0x24))))


def rounded(n):
    size = 4096
    while size < n:
        size <<= 1
    return size


def resolve(a):
    e_type, loads = elf(a.exe)
    rec = {"exe": a.exe, "kind": a.kind, "sites": {}}
    sites = [("stop", a.symbol, a.count)]
    if a.start_symbol:
        sites.insert(0, ("start", a.start_symbol, a.start_count))
    if a.kind == "plain" and e_type != 2:
        die("a plain program must be ET_EXEC (static, no PIE)")
    if a.kind == "capstone":
        if len(loads) != 2 or loads[0][2] != 5 or loads[1][2] != 6:
            die("a domain image has two PT_LOADs, text (R+X) then data (R+W)")
        low = loads[0][0]
        region = rounded(loads[0][1] + ((loads[1][1] + 4095) & ~4095))
        rec.update(low=low, region=region, domain_main=symbol(a.nm, a.exe, "domain_main"))
    # The anchor: the image's entry instruction, whose arrival arms the image's base (the plain
    # program's e_entry; the domain's domain_main, where the launcher enters it).
    anchor = rec["domain_main"] if a.kind == "capstone" else struct.unpack_from("<Q", open(a.exe, "rb").read(), 24)[0]
    low = rec.get("low", 0)
    # The text segment, in the plugin's vaddr space: every instruction in it counts on its image's
    # counter, which tells a completed site from one reached by another path.
    tseg = [l for l in loads if l[2] & 1 and l[0] <= anchor < l[0] + l[1]]
    if len(tseg) != 1:
        die(f"the entry {anchor:#x} is in {len(tseg)} executable segments")
    rec["text"] = {"vaddr": tseg[0][0], "size": tseg[0][1]}
    args = []
    for which, name, count in sites:
        if count is None or count < 1:
            die(f"{which}: a count K >= 1 is needed")
        link, size = symbol(a.nm, a.exe, name, size=True)
        length, word = first_insn(loads, link)
        if transfers_control(word, length):
            die(f"{name}'s first instruction {word:0{2 * length}x} transfers control: its entries cannot be "
                "told from fault retries; pick another function")
        if link in (anchor, anchor - length) or link + length == anchor:
            die(f"{name} at {link:#x} is the image's entry {anchor:#x} or next to it: the anchor's arrival would "
                "reset the site's own; pick another function")
        if not tseg[0][0] <= link < link + length < tseg[0][0] + tseg[0][1]:
            die(f"{name} and its successor are not in the entry's text segment")
        nlen, nword = first_insn(loads, link + length)
        site = {"symbol": name, "count": count, "link": link, "size": size, "len": length, "bytes": word,
                "next_len": nlen, "next_bytes": nword}
        args += [f"{which}_vaddr={link - low:x}", f"{which}_count={count}", f"{which}_bytes={word:x}",
                 f"{which}_next_bytes={nword:x}"]
        rec["sites"][which] = site
    length, word = first_insn(loads, anchor)
    rec["anchor"] = {"link": anchor, "len": length, "bytes": word}
    args += [f"anchor_vaddr={anchor - low:x}", f"anchor_bytes={word:x}",
             f"text_vaddr={tseg[0][0] - low:x}", f"text_size={tseg[0][1]:x}"]
    if a.kind == "capstone":
        args.append(f"slide_align={rec['region']:x}")
    if "start" in rec["sites"] and rec["sites"]["start"]["count"] >= rec["sites"]["stop"]["count"] \
            and rec["sites"]["start"]["symbol"] == a.symbol:
        die("start_count must be below stop_count for one symbol")
    rec["plugin_args"] = args
    print(json.dumps(rec, indent=1))


def foreign(wfile, loads, slide, lo, hi):
    """(attempts, addresses) of a window's EXECUTED records in [lo, hi) (link addresses; run-time =
    link + slide) whose bytes are not the image's; None when there are none."""
    att = n = 0
    for line in (open(wfile) if os.path.exists(wfile) else []):
        if line[0] == "#":
            continue
        rva, ln, b, a, _ = line.split("\t")
        link, ln = int(rva, 16) - slide, int(ln)
        if not int(a):
            continue    # translated in the window, never executed there (prof-report skips these too)
        if lo <= link < hi:
            seg = [l for l in loads if l[0] <= link < l[0] + len(l[3])]
            if not seg or int.from_bytes(seg[0][3][link - seg[0][0]:link - seg[0][0] + ln], "little") != int(b, 16):
                att += int(a); n += 1
    return (att, n) if n else None


def footer(path):
    """The '# key=value ...' lines of a window file, by key."""
    got = {}
    for line in open(path):
        if line.startswith("# ") and "=" in line.split()[1]:
            k, _, v = line[2:].split()[0].partition("=")
            got[k] = dict([("value", v)] + [tuple(w.split("=", 1)) for w in line.split()[2:] if "=" in w])
        elif line.startswith("# stop_candidates") or line.startswith("# start_candidates"):
            w = line.split()
            got[w[1]] = [tuple(x.split(":")) for x in w[2:]]
    return got


def finish(out):
    want = json.load(open(os.path.join(out, "stop-at.json")))
    result_path = os.path.join(out, "result.json")
    result = json.load(open(result_path))
    stop = want["sites"]["stop"]
    reasons = []
    stopped_path = os.path.join(out, "capprof", "stopped")
    if not os.path.exists(stopped_path):
        # The site's K-th entry never came. Without a start site the run is a full run that did not
        # stop: its verdict stands, and the record says how far the site got. With one, the plugin
        # dropped every count before the start site's entry, so the windows are neither a full run
        # nor the slice asked for: FAIL.
        result["slice"] = {"requested": stop, "start": want["sites"].get("start"), "stopped": False}
        if want["sites"].get("start"):
            result["runcpu_reasons"] = result.get("reasons", [])
            result["reasons"] = ["a start site trimmed the windows and the stop site's K-th entry never came"]
            result["verdict"] = "FAIL"
        json.dump(result, open(result_path, "w"), indent=1)
        print(result["verdict"], "(not stopped: the site's K-th entry never came)", result.get("reasons"))
        return
    line = open(stopped_path).read().split()
    kv = dict(w.split("=", 1) for w in line)
    va_s, _, k_s = kv["stopped_at"].partition(":")
    va, k, window = int(va_s, 16), int(k_s), int(kv["window"])
    if k != stop["count"]:
        reasons.append(f"stopped at entry {k}, asked for {stop['count']}")
    if int(kv["bytes"], 16) != stop["bytes"] or int(kv.get("next_bytes", "-1"), 16) != stop["next_bytes"]:
        reasons.append(f"stopped on bytes {kv['bytes']} / {kv.get('next_bytes')}, the symbol's are "
                       f"{stop['bytes']:08x} / {stop['next_bytes']:08x}")
    wfile = os.path.join(out, "capprof", f"window-{window}.tsv")
    ft = footer(wfile) if os.path.exists(wfile) else {}
    if "stopped_at" not in ft:
        reasons.append(f"window {window} has no stopped_at footer")
    # capprof lists one candidate per (address, physical page): vaddr:page:entries:window entries:unarmed.
    # The stop must have been fed by ONE of them in the stopped window: two images with the site's
    # bytes at the armed address (another static program at the same link address) would both count.
    cands = [c for c in ft.get("stop_candidates", []) if int(c[3])]
    if len(cands) != 1 or int(cands[0][0], 16) != va:
        reasons.append(f"{len(cands)} (address, page) candidates entered the stop site in window {window} ({cands}): not one image")
    start = want["sites"].get("start")
    if start:
        st = ft.get("started_at")
        if not st or int(st["value"].split(":")[1]) != start["count"]:
            reasons.append(f"no started_at for entry {start['count']} in window {window}")
        scands = [c for c in ft.get("start_candidates", []) if int(c[2])]
        if len(scands) != 1:
            reasons.append(f"{len(scands)} addresses entered as the start site ({scands})")
    image_record = None
    anchored = int(kv["base"], 16) if "base" in kv else None
    base = va - (stop["link"] - want.get("low", 0))
    if anchored != base:
        reasons.append(f"stopped at {va:#x}, i.e. base {base:#x}, but the plugin's armed base is {kv.get('base')}")
    if want["kind"] == "plain" and base != 0:
        reasons.append(f"stopped at {va:#x}, the symbol is at {stop['link']:#x}")
    if want["kind"] == "capstone" and base % want["region"]:
        reasons.append(f"base {base:#x} is not a multiple of the grant size {want['region']:#x}")
    # Identity of what executed: the stopped window's records in the image's text at that base (a
    # domain: the whole text, which a full domain run never shows foreign bytes in) or in the stop
    # function's own range (plain: other static programs share the link addresses and run in
    # windows, but not inside the site's function) must be the image's own bytes.
    _, loads = elf(want["exe"])
    seg = [l for l in loads if l[2] & 1 and l[0] <= stop["link"] < l[0] + l[1]][0]
    lo, hi = (seg[0], seg[0] + seg[1]) if want["kind"] == "capstone" else (stop["link"], stop["link"] + max(stop["size"], stop["len"] + stop["next_len"]))
    bad = foreign(wfile, loads, base - want.get("low", 0), lo, hi)
    if bad:
        reasons.append(f"{bad[0]} attempts at {bad[1]} addresses in {'the text' if want['kind'] == 'capstone' else stop['symbol']} "
                       f"at base {base:#x} are not the image's bytes")
    if want["kind"] == "capstone":
        entry = base + want["domain_main"] - want["low"]
        image_record = (f"capprof-slice: domain image entry={entry:#x} code={base:#x}-{base + want['region']:#x} "
                        f"(derived from stopped_at; the launcher never exited)")
        # Earlier invocations of the same image wrote their own records at exit: they must agree.
        rp = os.path.join(out, "image-records.txt")
        for l in (open(rp).read().splitlines() if os.path.exists(rp) else []):
            if os.path.basename(want["exe"]).split(".")[0] in l and f"entry={entry:#x} " not in l:
                reasons.append(f"an earlier record of this image disagrees: {l}")
    # The invocations: every guest invocation before the stopped one passed; the stopped one is the
    # last guest invocation and owns window `window`; everything after it was refused as after-stop.
    inv = [json.load(open(os.path.join(out, "invocations", f))) for f in
           sorted(os.listdir(os.path.join(out, "invocations"))) if f[:4].isdigit() and f.endswith(".json") and len(f) == 9]
    guest = [d for d in inv if d.get("route") in ("guest", "guest-plain")]
    if not guest or guest[-1].get("detail", {}).get("window") != window:
        reasons.append(f"the last guest invocation does not own the stopped window {window}")
    elif os.path.basename(guest[-1]["exe"]) != os.path.basename(want["exe"]):
        reasons.append(f"stopped in {os.path.basename(guest[-1]['exe'])}, the symbol was resolved in {want['exe']}")
    for d in guest[:-1]:
        g = (d.get("detail") or {}).get("guest") or {}
        if d["status"] != 0 or g.get("kind", "exit") != "exit" or g.get("value", 0) != 0 or g.get("fault"):
            reasons.append(f"guest invocation {os.path.basename(d['exe'])} (window {d['detail'].get('window')}) ended {g or d['status']}")
    for d in inv:
        if d.get("route") in ("host-spec-tool", "host-ldecod") and d["status"] != 0:
            reasons.append(f"host tool {os.path.basename(d['exe'])} exited {d['status']} before the stop")
        if d.get("route") == "refused":
            reasons.append("an invocation was refused before the stop")
    if image_record and not reasons:
        with open(os.path.join(out, "image-records.txt"), "a") as f:
            f.write(image_record + "\n")
    sl = {"symbol": stop["symbol"], "count": stop["count"], "len": stop["len"], "bytes": stop["bytes"],
          "size": stop["size"], "next_len": stop["next_len"], "base": base, "start": start and {"symbol": start["symbol"], "count": start["count"]},
          "stopped_at": va, "window": window, "arrivals": int(kv["arrivals"]), "image_record": image_record,
          "stopped": True}
    json.dump(sl, open(os.path.join(out, "slice.json"), "w"), indent=1)
    result["runcpu_reasons"] = result.get("reasons", [])
    result["reasons"] = reasons
    result["verdict"] = "SLICE" if not reasons else "FAIL"
    result["slice"] = sl
    json.dump(result, open(result_path, "w"), indent=1)
    print(result["verdict"], reasons)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--nm", required=True); r.add_argument("--kind", choices=("capstone", "plain"), required=True)
    r.add_argument("--exe", required=True); r.add_argument("--symbol", required=True)
    r.add_argument("--count", type=int, required=True)
    r.add_argument("--start-symbol"); r.add_argument("--start-count", type=int)
    f = sub.add_parser("finish"); f.add_argument("out")
    a = ap.parse_args()
    resolve(a) if a.cmd == "resolve" else finish(a.out)
