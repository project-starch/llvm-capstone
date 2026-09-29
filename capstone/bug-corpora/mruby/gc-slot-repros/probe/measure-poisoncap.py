#!/usr/bin/env python3
"""Measure the four target defects on CheriBSD purecap: CHERI baseline, then the
per-slot poison adapter in both modes.

Three arms, and all three are needed to read the result:

  baseline m-     no adapter. Says whether a fault is this level's catch or a path
                  CHERI's bounds already fault on. Run twice, with libc revocation
                  DISABLED and ENABLED per process, because the claim the corpus makes
                  is that malloc-interface revocation cannot see a reuse-not-free.
  poison m0       the adapter publishing exact per-slot bounds and invalidating nothing.
  poison m1       the same, invalidating at slot death.

m0 and m1 differ in ONE thing, the sweep: neither ever stores a NULL alias, so a fault in
m1 cannot be a NULL dereference (which raises the same in-address-space security exception
on purecap and would not be attributable to revocation). See patch 0012's header.

hash-sanity is the positive control and runs in every arm. It must PASS everywhere, with
sweeps=0 in m0 and sweeps>0 in m1; a clean run of the cases means nothing unless it does.

Usage: measure-poisoncap.py --work <poisoncap-work> --baseline <mruby> --poison <mruby>
                            --out <dir> [--port 10440]
"""
import argparse, glob, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "ports/common/host/cheribsd"))
from guest import Guest  # noqa: E402

CASES = ["hash-sanity", "hash-matched-vacated", "hash-scans-vacated", "hash-read-back",
         "hash-delete-in-eql", "string-strip-long",
         "inspect", "except", "rehash-ar", "rehash-ht",
         "del-index", "del-keyp", "del-store", "del-ht", "del-swap", "del-sane"]


def classify(text):
    if "security exception" in text:
        return "FAULT"
    if "EXIT=0" in text:
        return "completed"
    return "UNKNOWN"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--poison", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=HERE / "poison-cases")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=10440)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sources = {}
    for name in CASES:
        for cand in (a.cases / f"{name}.rb", a.cases.parent / f"case-{name}.rb"):
            if cand.exists():
                sources[name] = cand
                break
    missing = [c for c in CASES if c not in sources]
    if missing:                       # a case that is not there is an ERROR, not a zero
        sys.exit(f"cases not found, nothing measured: {', '.join(missing)}")

    guest = Guest(a.work / "sdk", a.work / "output/rootfs-riscv64-purecap",
                  a.work / "published-platform/cheribsd-riscv64-purecap.img",
                  a.out, a.port, disable_default_revocation=True)

    def sh(cmd, timeout=600):
        r = guest.ssh(cmd, timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "")).strip()

    guest.start()
    results, arms = {}, []
    try:
        print("guest:", sh("uname -r"))
        for tag, binary in (("baseline", a.baseline), ("poison", a.poison)):
            guest.copy(str(binary), f"root@127.0.0.1:/tmp/mruby-{tag}")
            sh(f"chmod 0755 /tmp/mruby-{tag}")
        for name, src in sources.items():
            guest.copy(str(src), f"root@127.0.0.1:/tmp/{name}.rb")
        arms = [("baseline", "revoff", "_RUNTIME_REVOCATION_DISABLE=1"),
                ("baseline", "revon", "_RUNTIME_REVOCATION_ENABLE=1"),
                ("poison", "m0", "MRB_POISON_MODE=0 MRB_POISON_REPORT=1"),
                ("poison", "m1", "MRB_POISON_MODE=1 MRB_POISON_REPORT=1")]
        for name in CASES:
            row = {}
            for tag, arm, env in arms:
                out = sh(f"cd /tmp; ulimit -c 0; env {env} /tmp/mruby-{tag} "
                         f"/tmp/{name}.rb 2>&1; echo EXIT=$?")
                row[arm] = {"verdict": classify(out), "output": out}
            results[name] = row
            print(f"  {name:22s} " + "  ".join(
                f"{arm}={row[arm]['verdict']}" for _, arm, _ in arms))
    finally:
        (a.out / "measurements.json").write_text(json.dumps(
            {"arms": [f"{t}/{m}" for t, m, _ in arms], "cases": results}, indent=1))
        guest.close()

    sanity = results.get("hash-sanity", {})
    bad = [arm for arm, r in sanity.items() if r["verdict"] != "completed"]
    if bad:                           # the instrument did not fire: no result at all
        sys.exit(f"hash-sanity did not complete in {bad}; the run measured nothing")
    if "sweeps=0" not in sanity.get("m0", {}).get("output", ""):
        sys.exit("mode 0 swept; it is not a control")
    if "sweeps=0" in sanity.get("m1", {}).get("output", ""):
        sys.exit("mode 1 never swept; a clean case would prove nothing")
    print("positive control OK: m0 swept nothing, m1 swept, both completed")


if __name__ == "__main__":
    main()
