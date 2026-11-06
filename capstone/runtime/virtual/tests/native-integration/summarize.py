#!/usr/bin/env python3
"""Raw system-wide attempts, without attributing unknown PCs to the domain."""
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
arms = [line.split() for line in (root / "arms.tsv").read_text().splitlines()]
windows = list((root / "capprof").glob("window-*.tsv"))
assert len(windows) == len(arms), "incomplete or unexpected windows"
assert len(arms) == 27 or "BUDGET_PARTIAL" in os.environ, "unexpected window count"
assert not list((root / "capprof").glob("*.invalid")), "invalid profiling window"
print("arm\tbytes\trep\tattempts\tcalls\tattempts_per_call")
for index, arm, size, rep in arms:
    log = (root / f"{index}.log").read_text()
    calls = 1 if size == "65536" else 100
    assert f"BUDGET bytes={size} calls={calls} failures=0" in log, log
    rows = (root / "capprof" / f"window-{index}.tsv").read_text().splitlines()
    attempts = sum(int(row.split()[3]) for row in rows if not row.startswith("#"))
    print(f"{arm}\t{size}\t{rep}\t{attempts}\t{calls}\t{attempts / calls:.2f}")
