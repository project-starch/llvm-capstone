#!/usr/bin/env python3
"""North-star accounting for two profile.sh runs: A plain, B Capstone.

The input bucket data is profile.json's complete all_functions list, classified
by prof-buckets.py. No cached or rounded text table is used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location(
    "prof_buckets", Path(__file__).with_name("prof-buckets.py"))
buckets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(buckets)

EXCLUDED = {"kernel: idle", "kernel: net/ssh", "kernel: fs/9p/virtio",
            "kernel: proc/printk", "profrun", "other"}


def work(data):
    s = data.get("slice")
    return None if not s else {"symbol": s["symbol"], "count": s["count"],
                               "start": s.get("start")}


def load(path):
    path = Path(path)
    if path.is_dir():
        path /= "profile.json"
    if not path.is_file():
        raise ValueError(f"missing bucket data file: {path}")
    data = json.loads(path.read_text())
    total = data["total_attempts"]
    if type(total) is not int or total <= 0:
        raise ValueError(f"{path}: total_attempts must be a positive integer")
    terms = dict(P=0, K=0, H=0)
    for cls, fn, count in data["all_functions"]:
        if type(count) is not int or count < 0:
            raise ValueError(f"{path}: invalid attempt count")
        name = buckets.bucket(cls, fn)
        key = "P" if name.startswith("program:") else "H" if name in EXCLUDED else "K"
        terms[key] += count
    if sum(terms.values()) != total:
        raise ValueError(f"{path}: buckets do not sum to total_attempts")
    result = path.with_name("result.json")
    if result.exists():
        verdict = json.loads(result.read_text())["verdict"]
        expected = "SLICE" if work(data) else "PASS"
        if verdict != expected:
            raise ValueError(f"{path}: verdict {verdict}, expected {expected}")
    return terms, work(data)


def compare(a, b):
    ta, wa = load(a)
    tb, wb = load(b)
    if wa != wb:
        raise ValueError(f"unequal work: A {wa or 'full run'}, B {wb or 'full run'}")
    denominator = ta["P"] + ta["K"]
    if denominator <= 0:
        raise ValueError("A has no program or boundary attempts")
    return {"a": ta, "b": tb, "work": wa, "denominator": denominator,
            "program_pct": 100 * (tb["P"] - ta["P"]) / denominator,
            "boundary_pct": 100 * (tb["K"] - ta["K"]) / denominator}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    args = parser.parse_args()
    try:
        result = compare(args.a, args.b)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"prof-northstar: {exc}\n")
    print("term                  A                B              B-A")
    for key in ("P", "K", "H"):
        a, b = result["a"][key], result["b"][key]
        print(f"{key:5s} {a:17,d} {b:16,d} {b-a:+16,d}")
    print(f"program {result['program_pct']:+.2f}%  boundary {result['boundary_pct']:+.2f}%"
          f"  H_A/H_B {result['a']['H']:,}/{result['b']['H']:,}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
