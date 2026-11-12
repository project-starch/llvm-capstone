#!/usr/bin/env python3
"""judge-probe.py LABEL RUN.json IMAGE [...]: what the judge reads from a case-batch boot's runs of CRASH-139's trigger.

Offline, from the stored record, through run-realphp.py's own observe() and tools/verdicts.py's judge, with the case's
spec as run-realphp.population() loads it (so with the declared reach witness). The stock arm's run of the same
repetition is the stock control, as in the runner. The controls passed are the sanity run of the same boot; the bound
probe did not run in these boots, so a silent run on an arm that requires it for MISSED could not read MISSED here --
none does. Prints one TSV row per run. These are readings of PROBE boots: they are written to no bundle."""
import importlib.util, json, os, sys
from pathlib import Path
R = Path(os.environ["CAPSTONE_REPO_ROOT"]) / "capstone"
spec = importlib.util.spec_from_file_location("rr", R / "ports/php/app/run-realphp.py")
rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
v = rr.verdicts
CASE = "41_9a98904ddd0e_crash-139"
ARM = {"stock": "php-real-stock", "block": "php-real-block", "object": "php-real-object", "exact": "php-real-exact"}
arms = v.load_arms()
case = rr.population(R / "bug-corpora/php/engine-repros", "asan-blind")[CASE]
llvm = Path(os.environ["CAPSTONE_LLVM_BIN"])
print("image\timage_sha256\tarm\trun\texit\tverdict\treason\treached\tfault\treach_evidence\tevidence")
args = sys.argv[1:]
for label, record, image in zip(args[0::3], args[1::3], args[2::3]):
    d = json.load(open(record)); image = Path(image); sha = rr.sha256_file(image)
    texts = {n: rr.app_output(l) for n, l in d["output"].items()}
    reps = sorted({n.rsplit("__", 1)[1] for n in texts if "__case__" in n})
    for rep in reps:
        stock = f"stock__case__{rep}"
        stock_reached = d["exits"].get(stock) == 0 and v.domain_fault(texts.get(stock, ""), None) is None
        for short, arm in ARM.items():
            name = f"{short}__case__{rep}"
            sanity = f"{short}__sanity"
            ok = d["exits"].get(sanity) == 0 and v.domain_fault(texts.get(sanity, ""), None) is None
            controls = [v.Control(name=rr.SANITY, observed="complete" if ok else "did-not-complete")]
            obs = rr.observe(CASE, case, arm, sha, texts[name], d["exits"].get(name), controls, stock_reached,
                             llvm, image, True, False, tuple(arms[arm]["protective_causes"]),
                             "\n".join(d["output"][name]))
            verdict, reason, evidence = v.judge(obs, arms[arm])
            f = obs.fault
            home = os.path.expanduser("~")
            clean = lambda s: (s or "").replace(home + "/crenew-step4/corpus/", "")
            print("\t".join([label, sha[:12], arm, rep, str(d["exits"].get(name)), verdict, reason or "",
                             str(obs.reached), f"cause={f.cause} in {f.symbol}" if f else "",
                             clean(obs.reach_evidence), clean(evidence)[:400]]))
