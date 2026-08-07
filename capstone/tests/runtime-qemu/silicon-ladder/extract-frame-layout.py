"""Re-derive each measured build's frame layout from its ELF, so the rule search runs on the
artifacts rather than on hand-transcribed notes (four rules died today from eyeballed patterns)."""
import subprocess, re, sys, json, os
OBJ = "/home/alexey/dev/llvm-capstone/llvm/cmake-build-debug/bin/llvm-objdump"

def layout(path, fn=None):
    d = subprocess.run([OBJ, "-d", "--triple=capstone64-unknown-elf", path],
                       capture_output=True, text=True).stdout
    for cand in ([fn] if fn else ["fdreg_depth_body", "fdreg_compute"]):
        tag = f"<{cand}>:"
        if tag in d:
            b = d.split(tag)[1].split("\n\n")[0]; break
    else:
        return None
    m = re.search(r"sp, sp, -0x([0-9a-f]+)", b)
    if not m: return None
    fr = int(m.group(1), 16)
    L = b.splitlines()
    rmw, bounds, store = [], {}, None
    for i, l in enumerate(L):
        mm = re.search(r"cincoffsetimm\s+a\d, s0, -0x([0-9a-f]+)", l)
        if not mm: continue
        off = fr - int(mm.group(1), 16)
        if i+1 < len(L) and re.search(r"\bstc\b", L[i+1]):
            store = off
        if i+2 < len(L) and re.search(r"\blw\b", L[i+1]):
            b2 = re.search(r"li\s+a0, 0x([0-9a-f]+)", L[i+2])
            if b2: bounds[off] = b2.group(1)
        if i+3 < len(L) and re.search(r"\blw\b", L[i+1]) and re.search(r"addi", L[i+2]) and re.search(r"\bsw\b", L[i+3]):
            rmw.append(off)
    return {"frame": fr, "store": store, "rmw": sorted(set(rmw)), "bounds": bounds}

if __name__ == "__main__":
    # Takes real paths so this is usable outside the machine it was written on.
    # Optional CAPSTONE_DOM_DIR lets bare names be resolved against a directory of .dom files.
    if len(sys.argv) < 2:
        print("usage: extract-frame-layout.py <domain.dom> [...]   "
              "(or bare names with CAPSTONE_DOM_DIR set)", file=sys.stderr)
        raise SystemExit(2)
    D = os.environ.get("CAPSTONE_DOM_DIR", ".")
    out = {}
    for n in sys.argv[1:]:
        for cand in (n, f"{n}.dom", os.path.join(D, n), os.path.join(D, f"{n}.dom")):
            if os.path.exists(cand):
                r = layout(cand)
                if r:
                    out[os.path.basename(cand)] = r
                break
        else:
            print(f"not found: {n}", file=sys.stderr)
    print(json.dumps(out, indent=1))
