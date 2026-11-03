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


def symbol(nm, exe, name):
    out = subprocess.run([nm, "-n", "--defined-only", exe], capture_output=True, text=True, check=True).stdout
    addrs = sorted({int(p[0], 16) for p in (l.split() for l in out.splitlines())
                    if len(p) == 3 and p[1] in "TtWw" and p[2] == name})
    if len(addrs) != 1:
        die(f"{exe}: {len(addrs)} text symbols named {name} ({[hex(a) for a in addrs]}); need exactly one")
    return addrs[0]


def first_insn(loads, va):
    for v, m, flags, b in loads:
        if flags & 1 and v <= va < v + len(b):
            o = va - v
            length = 2 if b[o] & 3 != 3 else 4
            return length, int.from_bytes(b[o:o + length], "little")
    die(f"{va:#x} is in no executable segment")


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
    args = []
    for which, name, count in sites:
        if count is None or count < 1:
            die(f"{which}: a count K >= 1 is needed")
        link = symbol(a.nm, a.exe, name)
        length, word = first_insn(loads, link)
        site = {"symbol": name, "count": count, "link": link, "len": length, "bytes": word}
        where = link - rec["low"] if a.kind == "capstone" else link
        args += [f"{which}_vaddr={where:x}", f"{which}_count={count}", f"{which}_bytes={word:x}"]
        rec["sites"][which] = site
    if a.kind == "capstone":
        args.append(f"slide_align={rec['region']:x}")
    if "start" in rec["sites"] and rec["sites"]["start"]["count"] >= rec["sites"]["stop"]["count"] \
            and rec["sites"]["start"]["symbol"] == a.symbol:
        die("start_count must be below stop_count for one symbol")
    rec["plugin_args"] = args
    print(json.dumps(rec, indent=1))


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
        # The site's K-th entry never came: the run is a full run that did not stop. Its verdict
        # stands, and the record says how far the site got.
        result["slice"] = {"requested": stop, "stopped": False}
        json.dump(result, open(result_path, "w"), indent=1)
        print(result["verdict"], "(not stopped: the site's K-th entry never came)", result.get("reasons"))
        return
    line = open(stopped_path).read().split()
    kv = dict(w.split("=", 1) for w in line)
    va_s, _, k_s = kv["stopped_at"].partition(":")
    va, k, window = int(va_s, 16), int(k_s), int(kv["window"])
    if k != stop["count"]:
        reasons.append(f"stopped at entry {k}, asked for {stop['count']}")
    if int(kv["bytes"], 16) != stop["bytes"]:
        reasons.append(f"stopped on bytes {kv['bytes']}, the symbol's are {stop['bytes']:08x}")
    wfile = os.path.join(out, "capprof", f"window-{window}.tsv")
    ft = footer(wfile) if os.path.exists(wfile) else {}
    if "stopped_at" not in ft:
        reasons.append(f"window {window} has no stopped_at footer")
    cands = [c for c in ft.get("stop_candidates", []) if int(c[1])]
    if len(cands) != 1:
        reasons.append(f"{len(cands)} addresses entered as the stop site ({cands}): not one function")
    start = want["sites"].get("start")
    if start:
        st = ft.get("started_at")
        if not st or int(st["value"].split(":")[1]) != start["count"]:
            reasons.append(f"no started_at for entry {start['count']} in window {window}")
        scands = [c for c in ft.get("start_candidates", []) if int(c[1])]
        if len(scands) != 1:
            reasons.append(f"{len(scands)} addresses entered as the start site ({scands})")
    image_record = None
    if want["kind"] == "plain":
        if va != stop["link"]:
            reasons.append(f"stopped at {va:#x}, the symbol is at {stop['link']:#x}")
    else:
        base = va - (stop["link"] - want["low"])
        if base % want["region"]:
            reasons.append(f"base {base:#x} is not a multiple of the grant size {want['region']:#x}")
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
    sl = {"symbol": stop["symbol"], "count": stop["count"], "start": start and {"symbol": start["symbol"], "count": start["count"]},
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
