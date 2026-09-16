import sys, re, collections
path = sys.argv[1]; want = sys.argv[2:]
ids = {}          # id -> list of (fullname, width)
scope = []
names_by_id = collections.defaultdict(list)
with open(path, errors="replace") as f:
    header = True
    changes = collections.defaultdict(list)
    t = 0
    for line in f:
        line = line.strip()
        if header:
            if line.startswith("$scope"):
                scope.append(line.split()[2])
            elif line.startswith("$upscope"):
                scope.pop()
            elif line.startswith("$var"):
                p = line.split()
                w, vid, name = int(p[2]), p[3], p[4]
                full = ".".join(scope + [name])
                names_by_id[vid].append((full, w))
            elif line.startswith("$enddefinitions"):
                header = False
                sel = {}
                for vid, lst in names_by_id.items():
                    for full, w in lst:
                        if any(re.search(pat, full) for pat in want):
                            sel[vid] = (full, w)
                print("selected", len(sel))
                for vid, (full, w) in sorted(sel.items(), key=lambda x: x[1][0]): print("  ", full, w)
            continue
        if line.startswith("#"):
            t = int(line[1:]); continue
        if not line: continue
        if line[0] in "01xz":
            vid = line[1:]; val = line[0]
        elif line[0] == "b":
            val, vid = line[1:].split()
        else:
            continue
        if vid in sel:
            changes[vid].append((t, val))
for vid, (full, w) in sorted(sel.items(), key=lambda x: x[1][0]):
    ch = changes[vid]
    def fmt(v):
        try: return hex(int(v, 2)) if len(v) > 1 else v
        except ValueError: return v
    print(f"== {full} [{w}] {len(ch)} changes")
    for t, v in ch[:400]: print(f"   t={t} {fmt(v)}")
