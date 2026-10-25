#!/usr/bin/env python3
"""Build a manifest's images with the virtual SDK, plus the run's support programs (ok, the corpus controls, and for
seeded builds the guest rand() check).

    build.py --manifest M --juliet <dir>/C --sdk <virtual SDK dir> --out <fresh dir> [--jobs 48]
             [--seeded | --twins --verdicts <set>-verdicts.tsv]

  default   every case twice: a bad image (-DOMITGOOD) and a good image (-DOMITBAD), both -DINCLUDEMAIN with io.c
  --seeded  the bad image of every rand_flow case once per seed (common.SEEDS), from an overlay that changes only
            main's srand(time(NULL))
  --twins   for every MISSED case of a family with a twin rule (common.TWIN_RULES), its bad image with one print added
            at the statement that decides whether the defect happens (seeded too when the case is rand_flow)

A build failure is recorded with its first error line, never dropped; build.json lists every image with its sha256.
Exit 0 when every image built, 1 otherwise.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as C

OK_C = '#include <stdio.h>\nint main(void){puts("OK-PROGRAM done");return 0;}\n'
RANDCHECK_C = ('#include <stdio.h>\n#include <stdlib.h>\nint main(void){for(unsigned s=1;s<=4;s++){srand(s);'
               'int a=rand()%2,b=rand()%2;printf("RAND seed=%u first2=%d%d\\n",s,a,b);}return 0;}\n')


def write_variant(out, case, srcs, seeded, fam):
    """Copy a case's files into `out`, applying the seed overlay and/or the twin rule; refuse when either does not
    apply exactly once."""
    out.mkdir(parents=True)
    seeds = twins = 0
    for s in srcs:
        text = s.read_text(errors="replace")
        if seeded:
            text, n = C.overlay(text); seeds += n
        if fam:
            text, n = C.twin_source(text, fam); twins += n
        (out / s.name).write_text(text)
    if seeded and seeds != 1:
        sys.exit(f"{case}: expected one srand(time(NULL)), found {seeds}")
    if fam and twins < 1:
        sys.exit(f"{case}: the {fam} twin rule matched nothing")
    return [out / s.name for s in srcs], twins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--juliet", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True, help="the virtual SDK directory holding capstone-cc")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=48)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--seeded", action="store_true")
    mode.add_argument("--twins", action="store_true")
    p.add_argument("--verdicts", type=Path, help="with --twins: judge.py's <set>-verdicts.tsv")
    a = p.parse_args()
    if a.twins and not a.verdicts:
        sys.exit("--twins needs --verdicts")
    cc = a.sdk / "capstone-cc"
    if not cc.is_file():
        sys.exit(f"{cc} missing")
    if a.out.exists():
        sys.exit(f"{a.out} exists; a build goes into a fresh directory")
    img = a.out / "img"; img.mkdir(parents=True)
    man = C.load(a.manifest)
    tc, sup = a.juliet / "testcases", a.juliet / "testcasesupport"
    for f, h in man["support"].items():
        if C.sha256(sup / f) != h:
            sys.exit(f"{sup / f} differs from the manifest")
    missed = set()
    if a.twins:
        with open(a.verdicts) as f:
            head = f.readline().rstrip("\n").split("\t")
            missed = {r["case"] for r in (dict(zip(head, l.rstrip("\n").split("\t"))) for l in f) if r["verdict"] == "MISSED"}

    jobs = []   # (image name, case, arm, seed pattern or None, twin family or None, [sources], [defines])
    for c in man["cases"]:
        for f, h in c["sha256"].items():
            if C.sha256(tc / f) != h:
                sys.exit(f"{tc / f} differs from the manifest")
        srcs = [tc / f for f in c["files"]]
        if not (a.seeded or a.twins):
            jobs.append((f"{c['case']}.good", c, "good", None, None, srcs, ["-DOMITBAD"]))
            jobs.append((f"{c['case']}.bad", c, "bad", None, None, srcs, ["-DOMITGOOD"]))
            continue
        fam = C.family(c["case"]) if a.twins else None
        if a.twins and (c["case"] not in missed or fam not in C.TWIN_RULES):
            continue
        if a.seeded and not c["rand_flow"]:
            continue
        tag = "twin" if a.twins else "bad"
        srcs, _ = write_variant(a.out / "variants" / c["case"], c["case"], srcs, c["rand_flow"], fam)
        if c["rand_flow"]:
            for pat, seed in C.SEEDS.items():
                jobs.append((f"{c['case']}.{tag}.s{pat}", c, "bad", pat, fam, srcs, ["-DOMITGOOD", f"-DJULIET_SEED={seed}"]))
        else:
            jobs.append((f"{c['case']}.{tag}", c, "bad", None, fam, srcs, ["-DOMITGOOD"]))

    def build(job):
        name, c, arm, pat, fam, srcs, defs = job
        out = img / f"{name}.dom"
        cmd = [str(cc), "-O0", "-DINCLUDEMAIN", *defs, "-I", str(sup), *map(str, srcs), str(sup / "io.c"), "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=C.clean_env())
        log = (r.stdout + r.stderr).strip()
        ok = r.returncode == 0 and out.is_file()
        return {"image": name, "case": c["case"], "cwe": c["cwe"], "arm": arm, "seed_pattern": pat, "twin": fam, "ok": ok,
                "warnings": log.count("warning:"),
                "first_error": next((l for l in log.splitlines() if "error" in l), "")[:300] if not ok else "",
                "sha256": C.sha256(out) if ok else None}

    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        images = list(ex.map(build, jobs))
    extra = {"ok": OK_C}
    if a.seeded or a.twins:
        extra["randcheck"] = RANDCHECK_C
    for name, src in extra.items():
        (a.out / f"{name}.c").write_text(src)
    for name, src in [*((n, a.out / f"{n}.c") for n in extra), ("controls", C.CONTROLS_C)]:
        subprocess.run([str(cc), "-O0", str(src), "-o", str(img / f"{name}.dom")], check=True, env=C.clean_env())
    sdk = json.loads((a.sdk / "sdk.json").read_text())
    record = {"set": man["set"], "manifest_sha256": C.sha256(a.manifest), "sdk_fingerprint": C.sdk_fingerprint(a.sdk),
              "mode": "twins" if a.twins else "seeded" if a.seeded else "census",
              "wall_s": round(time.time() - t0, 1), "cc_sha256": C.sha256(cc),
              "sdk": {k: v for k, v in sdk.items() if not (isinstance(v, str) and "/" in v)},
              "flags": ["-O0", "-DINCLUDEMAIN", "-DOMITGOOD|-DOMITBAD"],
              "support_images": {n: C.sha256(img / f"{n}.dom") for n in (*extra, "controls")}, "images": images}
    (a.out / "build.json").write_text(json.dumps(record, indent=0))
    bad = [i for i in images if not i["ok"]]
    print(f"{len(images)} images in {record['wall_s']} s: {len(images) - len(bad)} built, {len(bad)} failed, "
          f"{sum(1 for i in images if i['warnings'])} with warnings")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
