"""Collect the glob-assign validation into one results.json (run on the build host)."""
import csv, glob, hashlib, json, os, re, subprocess
M = os.path.expanduser("~/sa-glob-pf")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
out = {"snapshot": "git archive 16d2171a3edc capstone + ports/perl/musl/patches/5.36.3/0011 (uncommitted, sha256 "
       + sha(f"{M}/repo/capstone/ports/perl/musl/patches/5.36.3/0011-glob-assign-reads-source-before-leave.patch")
       + ") + compiler-rt/lib/builtins of 9bc6f9adf1c9"}
out["images"] = {a: sha(f"{M}/perl-{a}/src/perl-5.36.3/perl") for a in ("stock", "protected")}
out["native"] = {v: sha(f"{M}/native-{v}/perl-5.36.3/perl") for v in ("without-0011", "with-0011")}
out["native_gate"] = json.loads(subprocess.check_output(["python3", "-I", f"{M}/native-compare.py", f"{M}/native-gate"]))
# same-answers boot
log = open(glob.glob(f"{M}/guest-sa-*/serial.log")[-1], errors="replace").read().replace("\r", "")
nat = f"{M}/native-gate/with-0011/raw"
sa = {}
for m in re.finditer(r"^SA_BEGIN:(\w+):(\S+)\n(.*?)^SA_STDERR:\1:\2\n(.*?)^SA_END:\1:\2:(\d+):([\d.]+):([\d.]+):(\w+)$", log, re.M | re.S):
    arm, name, so, se, rc, s, e, h = m.groups()
    nfile = {"control": "control.stdout", "smoke": "smoke.stdout"}.get(name, name.replace("/", "_") + ".stdout")
    nh = sha(f"{nat}/{nfile}")
    fault = re.search(r"domain fault cause=(\d+)", se)
    sa.setdefault(name, {})[arm] = dict(exit=int(rc), guest_wall_s=round(float(e) - float(s), 2), stdout_sha256_guest=h,
        stdout_sha256_host=hashlib.sha256(so.encode()).hexdigest(), fault_cause=int(fault.group(1)) if fault else None,
        stdout_equals_native_with_0011=h == nh)
    sa[name]["native_with_0011_sha256"] = nh
for name, d in sa.items():
    d["stock_equals_protected"] = d["stock"]["stdout_sha256_guest"] == d["protected"]["stdout_sha256_guest"] and d["stock"]["exit"] == d["protected"]["exit"]
out["same_answers"] = sa
out["same_answers_images_guest"] = dict(re.findall(r"^SA_IMAGE:(\w+):(\w+)$", log, re.M))
# corpus and qualify
for arm in ("virtual-malloc", "virtual-nested-pools"):
    out.setdefault("corpus", {})[arm] = dict(inputs=json.load(open(f"{M}/corpus/{arm}/inputs.json")),
        rows=list(csv.DictReader(open(f"{M}/corpus/{arm}/matrix.tsv"), delimiter="\t")))
for a in ("stock", "protected"):
    out.setdefault("qualify", {})[a] = list(csv.DictReader(open(f"{M}/corpus/qualify-{a}/qualify.tsv"), delimiter="\t"))
json.dump(out, open(f"{M}/results.json", "w"), indent=1)
print("ok", len(sa), "same-answers rows")
