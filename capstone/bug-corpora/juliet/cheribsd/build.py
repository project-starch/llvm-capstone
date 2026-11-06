#!/usr/bin/env python3
"""Build a manifest's images for CheriBSD purecap with the CheriBSD SDK's own purecap configuration, plus the run's
support programs (supervise, the controls, the probes, the seed finder).

    build.py --manifest M --juliet <dir>/C --sdk <CheriBSD SDK dir> --out <fresh dir> [--jobs 48] [--limit N]
             [--only REGEX] [--seeded --seeds <census run>/observations.json | --twins --verdicts <set>-verdicts.tsv]

  default   every case twice: a bad image (-DOMITGOOD) and a good image (-DOMITBAD), both -DINCLUDEMAIN with io.c,
            exactly the parent harness's build (../build.py) with the CheriBSD compiler in place of the virtual SDK
  --seeded  the bad image of every rand_flow case once per seed, from an overlay that changes only main's
            srand(time(NULL)). The seeds are the guest's own: the census run's `randseed` step found, under this
            libc's rand(), the first seed giving each pattern of the first two rand()%2 results.
  --twins   for every MISSED case of a family with a twin rule (../common.TWIN_RULES), its bad image with one print
            added at the statement that decides whether the defect happens (seeded too when the case is rand_flow)
  --limit   the first N cases of the manifest only; --only the cases whose name matches a regex (smoke builds:
            judge.py reports the rest as not built)

The compiler is `<sdk>/bin/clang --config cheribsd-riscv64-purecap.cfg -std=gnu11 -O0 -fuse-ld=lld`: the SDK's own
purecap configuration, which carries -cheri-tgot-tls (the dynamic linker of this CheriBSD refuses an image with
traditional TLS relocations, and io.c's <ctype.h> use references libc's thread-local rune locale).
A build failure is recorded with its first error line, never dropped; build.json lists every image with its sha256.
Exit 0 when every image built, 1 otherwise.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import shared as S

OK_C = '#include <stdio.h>\nint main(void){puts("OK-PROGRAM done");return 0;}\n'
# The use-after-free control: one free, no forced sweep, then the read. At the platform default the quarantined block
# is still tagged and the read completes; with every-free revocation the sweep ran inside free() and the read faults
# PROT_CHERI_TAG. The mark is flushed before the read so a fault cannot lose it in a buffer.
UAF_CONTROL_C = ('#include <stdio.h>\n#include <stdlib.h>\nint main(void){volatile char *p = malloc(100);'
                 'if(!p)return 2;p[0]=7;free((void*)p);puts("CONTROL uaf-free mark");fflush(stdout);'
                 'int v=p[0];printf("CONTROL uaf-free done %d\\n",v);return 0;}\n')
# The seed finder: the first seed, under the guest's rand(), for each pattern of the first two rand()%2 results.
RANDSEED_C = ('#include <stdio.h>\n#include <stdlib.h>\nint main(void){int seen[4]={0},found=0;'
              'for(unsigned s=1;s<100000&&found<4;s++){srand(s);int a=rand()%2,b=rand()%2;int k=a*2+b;'
              'if(!seen[k]){seen[k]=1;found++;printf("RANDSEED seed=%u first2=%d%d\\n",s,a,b);}}'
              'return found==4?0:1;}\n')
PATTERNS = ("11", "10", "01", "00")


def write_variant(out, case, srcs, seeded, fam):
    """Copy a case's files into `out`, applying the seed overlay and/or the twin rule; refuse when either does not
    apply exactly once."""
    out.mkdir(parents=True)
    seeds = twins = 0
    for s in srcs:
        text = s.read_text(errors="replace")
        if seeded:
            text, n = S.C.overlay(text); seeds += n
        if fam:
            text, n = S.C.twin_source(text, fam); twins += n
        (out / s.name).write_text(text)
    if seeded and seeds != 1:
        sys.exit(f"{case}: expected one srand(time(NULL)), found {seeds}")
    if fam and twins < 1:
        sys.exit(f"{case}: the {fam} twin rule matched nothing")
    return [out / s.name for s in srcs], twins


def guest_seeds(observations):
    """The seeds the census run's guest found, pattern -> seed; every batch that found any must have found the same."""
    obs = S.load(observations)
    found = [s["randseed"] for s in obs["steps"] if s["step"] == "randseed" and s.get("randseed")]
    if not found:
        sys.exit(f"{observations}: no randseed reading")
    if any(f != found[0] for f in found) or sorted(found[0]) != sorted(PATTERNS):
        sys.exit(f"{observations}: the batches' randseed readings differ or are incomplete: {found}")
    return found[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--juliet", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True, help="the CheriBSD SDK directory (bin/clang, bin/*.cfg)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=48)
    p.add_argument("--limit", type=int, help="build the first N cases only")
    p.add_argument("--only", help="regex: build only the cases whose name matches (a smoke build)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--seeded", action="store_true")
    mode.add_argument("--twins", action="store_true")
    p.add_argument("--seeds", type=Path, help="with --seeded: the census run's observations.json")
    p.add_argument("--verdicts", type=Path, help="with --twins: judge.py's <set>-verdicts.tsv")
    a = p.parse_args()
    if a.twins and not a.verdicts:
        sys.exit("--twins needs --verdicts")
    if a.seeded and not a.seeds:
        sys.exit("--seeded needs --seeds")
    cc = S.clang(a.sdk)
    if not cc.is_file() or not (a.sdk / "bin" / S.CONFIG).is_file():
        sys.exit(f"{cc} or its {S.CONFIG} missing")
    if a.out.exists():
        sys.exit(f"{a.out} exists; a build goes into a fresh directory")
    img = a.out / "img"; img.mkdir(parents=True)
    man = S.load(a.manifest)
    tc, sup = a.juliet / "testcases", a.juliet / "testcasesupport"
    for f, h in man["support"].items():
        if S.sha256(sup / f) != h:
            sys.exit(f"{sup / f} differs from the manifest")
    seeds = guest_seeds(a.seeds) if a.seeds else None   # --twins carries them too, for its rand_flow cases
    missed = set()
    if a.twins:
        with open(a.verdicts) as f:
            head = f.readline().rstrip("\n").split("\t")
            missed = {r["case"] for r in (dict(zip(head, l.rstrip("\n").split("\t"))) for l in f) if r["verdict"] == "MISSED"}

    jobs = []   # (image name, case, arm, seed pattern or None, twin family or None, [sources], [defines])
    selected = [c for c in man["cases"] if not a.only or re.search(a.only, c["case"])][:a.limit]
    for c in selected:
        for f, h in c["sha256"].items():
            if S.sha256(tc / f) != h:
                sys.exit(f"{tc / f} differs from the manifest")
        srcs = [tc / f for f in c["files"]]
        if not (a.seeded or a.twins):
            jobs.append((f"{c['case']}.good", c, "good", None, None, srcs, ["-DOMITBAD"]))
            jobs.append((f"{c['case']}.bad", c, "bad", None, None, srcs, ["-DOMITGOOD"]))
            continue
        fam = S.C.family(c["case"]) if a.twins else None
        if a.twins and (c["case"] not in missed or fam not in S.C.TWIN_RULES):
            continue
        if a.seeded and not c["rand_flow"]:
            continue
        tag = "twin" if a.twins else "bad"
        srcs, _ = write_variant(a.out / "variants" / c["case"], c["case"], srcs, c["rand_flow"], fam)
        if c["rand_flow"]:
            if seeds is None:
                sys.exit("a rand_flow twin needs --seeds (the census run's observations.json)")
            for pat in PATTERNS:
                jobs.append((f"{c['case']}.{tag}.s{pat}", c, "bad", pat, fam, srcs,
                             ["-DOMITGOOD", f"-DJULIET_SEED={seeds[pat]}"]))
        else:
            jobs.append((f"{c['case']}.{tag}", c, "bad", None, fam, srcs, ["-DOMITGOOD"]))

    def compile_(args, out):
        cmd = [str(cc), *S.FLAGS, *args, "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=S.C.clean_env())
        return r.returncode == 0 and out.is_file(), (r.stdout + r.stderr).strip()

    def build(job):
        name, c, arm, pat, fam, srcs, defs = job
        ok, log = compile_(["-DINCLUDEMAIN", *defs, "-I", str(sup), *map(str, srcs), str(sup / "io.c")], img / name)
        return {"image": name, "case": c["case"], "cwe": c["cwe"], "arm": arm, "seed_pattern": pat, "twin": fam,
                "ok": ok, "warnings": log.count("warning:"),
                "first_error": next((l for l in log.splitlines() if "error" in l), "")[:300] if not ok else "",
                "sha256": S.sha256(img / name) if ok else None}

    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        images = list(ex.map(build, jobs))
    for name, src in (("ok", OK_C), ("uaf-control", UAF_CONTROL_C), ("randseed", RANDSEED_C)):
        (a.out / f"{name}.c").write_text(src)
    support = {"ok": [str(a.out / "ok.c")], "uaf-control": [str(a.out / "uaf-control.c")],
               "randseed": [str(a.out / "randseed.c")], "probe": [str(S.MECHANISM_PROBE_C)],
               "capbounds": [str(S.CAPBOUNDS_C)], "sicode-selftest": ["-DSICODE_SELFTEST", str(S.SICODE_C)],
               "abi-probe": [str(S.ABI_PROBE_C)],
               "supervise": ['-DPROBE_SYMBOL="main"', str(S.SUPERVISE_C), "-lutil"]}
    for name, args in support.items():
        ok, log = compile_(args, img / name)
        if not ok:
            sys.exit(f"support program {name} did not build:\n{log}")
    record = {"set": man["set"], "manifest_sha256": S.sha256(a.manifest),
              "mode": "twins" if a.twins else "seeded" if a.seeded else "census", "limit": a.limit, "only": a.only,
              "seeds": seeds, "seeds_from_sha256": S.sha256(a.seeds) if a.seeds else None,
              "wall_s": round(time.time() - t0, 1), "toolchain": S.toolchain(a.sdk),
              "flags": [*S.FLAGS, "-DINCLUDEMAIN", "-DOMITGOOD|-DOMITBAD"],
              "support_sources": {n: S.sha256(next(x for x in args if x.endswith(".c"))) for n, args in support.items()},
              "support_images": {n: S.sha256(img / n) for n in support}, "images": images}
    (a.out / "build.json").write_text(json.dumps(record, indent=0))
    bad = [i for i in images if not i["ok"]]
    print(f"{len(images)} images in {record['wall_s']} s: {len(images) - len(bad)} built, {len(bad)} failed, "
          f"{sum(1 for i in images if i['warnings'])} with warnings")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
