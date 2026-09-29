#!/usr/bin/env python3
"""Compare the result rows of two `postgres --single` sessions over work.sql.

    python3 work-compare.py <native single.log> <domain qemu.log>

The rows are the tab-indented lines a single-user backend prints for each result
("N: col = "value" (typeid ...)" and "----"); prompts, LOG lines, timestamps and
the domain runtime's own lines are not compared. Exit 0 when the rows are
identical, 1 when they differ or either side has none: a log with no rows is an
error, not a match. survey-native.sh writes the native log ($PG_SU_ROOT/single.log);
run-domain.sh work writes the domain's ($PG_SU_ROOT/run/qemu.log).
"""
import difflib
import re
import sys


def rows(path):
    out = []
    for line in open(path, errors="replace"):
        line = re.sub(r"^(backend> )+", "", line.rstrip("\r\n"))
        if re.match(r"^\t( ?\d+: |----)", line):
            out.append(line.strip())
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    native, domain = rows(sys.argv[1]), rows(sys.argv[2])
    # work.sql has one aggregate, three joined rows, and a final count. A
    # backend that dies during CREATE INDEX produces no rows; one that dies
    # after an earlier SELECT can still produce a matching prefix.
    if len(native) != 22 or 'count = "1500"' not in native[-2]:
        print(f"native work.sql oracle incomplete: {len(native)} rows")
        return 1
    if len(domain) != 22 or 'count = "1500"' not in domain[-2]:
        print(f"domain work.sql oracle incomplete: {len(domain)} rows")
        return 1
    if native == domain:
        print(f"IDENTICAL: {len(native)} result rows; last: {native[-2]}")
        return 0
    print(f"DIFFER: native {len(native)} rows, domain {len(domain)}")
    diff = difflib.unified_diff(native, domain, "native", "domain", lineterm="")
    print("\n".join(list(diff)[:40]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
