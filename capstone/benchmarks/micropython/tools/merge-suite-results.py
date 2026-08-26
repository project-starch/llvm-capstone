#!/usr/bin/env python3
"""Merge resumable MicroPython chunk reports and validate their global indices."""

import argparse
import collections
import pathlib
import sys


HEADER = "index\tname\tstatus\tgot\twant\toracle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--failures-out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()

    rows = []
    for name in args.inputs:
        lines = pathlib.Path(name).read_text().splitlines()
        if not lines or lines[0] != HEADER:
            sys.exit(f"invalid result header: {name}")
        rows.extend(line.split("\t") for line in lines[1:])

    rows.sort(key=lambda row: int(row[0]))
    indices = [int(row[0]) for row in rows]
    if len(indices) != len(set(indices)):
        sys.exit("duplicate global indices in result inputs")
    if indices and indices != list(range(indices[0], indices[-1] + 1)):
        sys.exit("gap in global result indices")

    def write(path, selected):
        with pathlib.Path(path).open("w") as out:
            out.write(HEADER + "\n")
            for row in selected:
                out.write("\t".join(row) + "\n")

    write(args.out, rows)
    write(args.failures_out, (row for row in rows if row[2] != "PASS"))

    counts = collections.Counter(row[2] for row in rows)
    print(f"TOTAL={len(rows)} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for directory in sorted({row[1].split("/", 1)[0] for row in rows}):
        directory_counts = collections.Counter(
            row[2] for row in rows if row[1].split("/", 1)[0] == directory
        )
        print(directory + " " + " ".join(
            f"{k}={v}" for k, v in sorted(directory_counts.items())
        ))


if __name__ == "__main__":
    main()
