"""Reuse-distance runs, Perl and FFmpeg rows: serial logs -> reuse.json per program.

python3 -I parse.py M TAG OUT   (M: the run directory holding guest-TAG-*/serial.log,
the builds and the kit links; OUT gets perl-reuse.json and ffmpeg-reuse.json)

bucket k = observer bins[k-1] (distance d with 2^(k-1) <= d < 2^k); empty buckets omitted.
A run whose observer error is nonzero, whose output check fails, or whose REUSE-GAP line
is missing makes this script exit 1 (no file written)."""
import hashlib, json, os, re, sys

M, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = os.path.expanduser("~/pg-virtual-kit")
PRE = "16d2171a3edc"
SMOKE_SHA = "a4863fc7bf3ca18217711d9df5fea63f1851bb96d6f79e1a487680ae9268450e"
RUNS = {"perl": {"stock": [1, 2, 3], "protected": [1, 2, 3]},
        "ffmpeg": {"stock": [1], "protected": [1]}}
BOUNDARIES = {"perl": ["sv-heads"], "ffmpeg": ["buffer-pool", "refstruct-pool"]}
frames = lambda t: re.findall(r"^0,.*[0-9a-f]{32}$", t, re.M)
NATIVE_FRAMES = frames(open(f"{K}/ffapp-ref/native.out").read())
failures = []


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_line(line):
    m = re.match(r"^REUSE-GAP (\S+) (.*) bins=([\d,]+)$", line)
    name, fields, bins = m.group(1), m.group(2), [int(x) for x in m.group(3).split(",")]
    f = {k: int(v) for k, v in (kv.split("=") for kv in fields.split())}
    assert len(bins) == 32
    rec = {"buckets": {str(b + 1): n for b, n in enumerate(bins) if n},
           "issues": f["issues"], "releases": f["releases"], "reuses": f["reuses"],
           "distinct_starts": f["distinct_starts"], "attempts": f["attempts"], "error": f["error"]}
    for extra in ("drops", "fresh_reuses", "capacity"):
        if extra in f:
            rec[extra] = f[extra]
    if sum(bins) != f["reuses"]:
        failures.append(f"{name}: bins sum {sum(bins)} != reuses {f['reuses']}")
    return name, rec


def median_bucket(b):
    n = sum(b.values())
    acc = 0
    for k in sorted(b, key=int):
        acc += b[k]
        if 2 * acc >= n:
            return int(k)
    return None


def reusediff(a, b):
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys) / 2


def run(prog, arm, rep):
    rid = f"{prog}-{arm}-{rep}"
    log = open(f"{M}/guest-{TAG}-{rid}/serial.log", errors="replace").read().replace("\r", "")
    sec = lambda n: re.search(rf"^RG_{n}_BEGIN\n(.*?)^RG_{n}_END$", log, re.M | re.S).group(1)
    r = dict(kv.split("=", 1) for kv in re.search(r"^RG_RUN:(.*)$", log, re.M).group(1).split(":"))
    err, out = sec("STDERR"), sec("STDOUT")
    lines = [l for l in err.splitlines() if l.startswith("REUSE-GAP ")]
    obs = dict(parse_line(l) for l in lines)
    res = {"exit": int(r["rc"]), "guest_wall_s": round(float(r["end"]) - float(r["start"]), 2),
           "image_sha256_guest": re.search(r"^RG_IMAGE:(\w+)$", log, re.M).group(1),
           "stdout_sha256_guest": r["stdout_sha256"],
           "cleanup_rmmod_rc": int(re.search(r"^RG_CLEANUP:(\d+)$", log, re.M).group(1)),
           "stderr_other_lines": [l for l in err.splitlines() if l and not l.startswith("REUSE-GAP ")],
           "observers": obs}
    if prog == "perl":
        host = hashlib.sha256(out.encode()).hexdigest()
        res["stdout_sha256_host"] = host
        res["output_check"] = r["stdout_sha256"] == SMOKE_SHA and host == SMOKE_SHA
    else:
        got = frames(out)
        res["frames"] = len(got)
        res["output_check"] = len(got) == 30 and got == NATIVE_FRAMES
        res["stdout_non_frame_lines"] = [l for l in out.splitlines() if l and not l.startswith("0,")]
    if res["exit"] != 0 or not res["output_check"]:
        failures.append(f"{rid}: exit {res['exit']} output_check {res['output_check']}")
    for b in BOUNDARIES[prog]:
        if b not in obs:
            failures.append(f"{rid}: no REUSE-GAP line for {b}")
        elif obs[b]["error"] != 0:
            failures.append(f"{rid}: {b} observer error {obs[b]['error']}")
    return res


platform_files = {
    "qemu-system-riscv64": f"{K}/kit-sublet/qemu-system-riscv64",
    "fw_jump.elf": f"{K}/kit-sublet/images/fw_jump.elf",
    "Image": f"{K}/kit-sublet/images/Image",
    "rootfs.ext2": f"{K}/kit-sublet/images/rootfs.ext2",
    "capstone-vexec": f"{K}/kit-sublet/adapter/capstone-vexec",
    "capstone_vm.ko": f"{K}/kit-sublet/adapter/module/capstone_vm.ko",
    "run-staged.py": f"{M}/repo/capstone/runtime/virtual/run-staged.py",
    "clang": f"{K}/toolchain/bin/clang"}
platform = {"sha256": {k: sha(v) for k, v in platform_files.items()},
            "description": "Sublet-lifetime QEMU af37cc32, launcher 3975314a, module a1c6cb6b, "
                           "exact bounds, one hart, 2 GiB (as registered)"}

meta = {
    "perl": dict(program="Perl 5.36.3",
                 workload="scripts/smoke.pl, CAPSTONE_PERL_SMOKE_OBJECTS=2000, capstone-vexec perl.dom smoke.pl DIR",
                 images={"stock": f"{M}/perl-stock/perl.dom", "protected": f"{M}/perl-protected/perl.dom"},
                 uninstrumented={"stock": f"{M}/perl-stock/uninstrumented.dom",
                                 "protected": f"{M}/perl-protected/uninstrumented.dom"}),
    "ffmpeg": dict(program="FFmpeg 9.0.1",
                   workload="decoder application, capstone-vexec ffmpeg.dom input.mkv 1 (30 frames)",
                   images={"stock": f"{M}/ffmpeg-stock/image/ffmpeg.dom",
                           "protected": f"{M}/ffmpeg-protected/image/ffmpeg.dom"},
                   uninstrumented=None)}

docs = {}
for prog, arms in RUNS.items():
    runs = {arm: [run(prog, arm, rep) for rep in reps] for arm, reps in arms.items()}
    doc = {"program": meta[prog]["program"], "pre_registration": PRE, "workload": meta[prog]["workload"],
           "boundaries": {}, "images": {}, "platform": platform, "runs": {}}
    if prog == "perl":
        doc["buckets_note"] = ("3 repetitions per arm; 'buckets' and the counters beside it are "
                               "repetition 1, 'repetitions' lists all three in order")
    for b in BOUNDARIES[prog]:
        doc["boundaries"][b] = {}
        for arm, rs in runs.items():
            rec = dict(rs[0]["observers"][b])
            if len(rs) > 1:
                rec["repetitions"] = [x["observers"][b] for x in rs]
            doc["boundaries"][b][arm] = rec
        # summary over all repetition pairs
        S = [x["observers"][b]["buckets"] for x in runs["stock"]]
        P = [x["observers"][b]["buckets"] for x in runs["protected"]]
        keys = sorted(set().union(*S, *P), key=int)
        summ = {
            "median_bucket": {"stock": [median_bucket(x) for x in S], "protected": [median_bucket(x) for x in P]},
            "reusediff_rep1": round(reusediff(S[0], P[0]), 6),
            "reusediff_all_cross_pairs": sorted(round(reusediff(s, p), 6) for s in S for p in P),
            "reusediff_within_stock": sorted(round(reusediff(S[i], S[j]), 6) for i in range(len(S)) for j in range(i + 1, len(S))),
            "reusediff_within_protected": sorted(round(reusediff(P[i], P[j]), 6) for i in range(len(P)) for j in range(i + 1, len(P))),
            "identical_rep1": S[0] == P[0],
            "per_bucket_ranges": {k: {"stock": [min(x.get(k, 0) for x in S), max(x.get(k, 0) for x in S)],
                                      "protected": [min(x.get(k, 0) for x in P), max(x.get(k, 0) for x in P)]}
                                  for k in keys},
        }
        summ["ranges_overlap_every_bucket"] = all(
            max(v["stock"][0], v["protected"][0]) <= min(v["stock"][1], v["protected"][1])
            for v in summ["per_bucket_ranges"].values())
        summ["buckets_identical_every_pair"] = all(s == p for s in S for p in P)
        doc["boundaries"][b]["summary"] = summ
    for arm in RUNS[prog]:
        img = meta[prog]["images"][arm]
        h = sha(img)
        doc["images"][arm] = {"sha256": h}
        if meta[prog]["uninstrumented"]:
            doc["images"][arm]["uninstrumented_sha256"] = sha(meta[prog]["uninstrumented"][arm])
        for x in runs[arm]:
            if x["image_sha256_guest"] != h:
                failures.append(f"{prog}-{arm}: guest image hash differs from host")
        doc["runs"][arm] = [{k: v for k, v in x.items() if k != "observers"} for x in runs[arm]]
    docs[prog] = doc

if failures:
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
os.makedirs(OUT, exist_ok=True)
for prog, doc in docs.items():
    with open(f"{OUT}/{prog}-reuse.json", "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    for b, d in doc["boundaries"].items():
        s = d["summary"]
        print(prog, b, "issues", d["stock"]["issues"], d["protected"]["issues"],
              "reuses", d["stock"]["reuses"], d["protected"]["reuses"],
              "median", s["median_bucket"], "reusediff_rep1", s["reusediff_rep1"],
              "identical_every_pair", s["buckets_identical_every_pair"],
              "overlap", s["ranges_overlap_every_bucket"])
