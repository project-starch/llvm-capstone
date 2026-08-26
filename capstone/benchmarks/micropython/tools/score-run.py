#!/usr/bin/env python3
"""Score a test-runner boot: match the loader's retvals against the expected table.

The loader prints one line per call, `Called dom (N-th time) retval = <decimal>`. The domain packs
index/length/hash into that word (see mpy_domain.c, MPY_TEST_RUNNER). This compares the two.

Four outcomes per test, kept distinct on purpose:
  PASS    the word matches -- the interpreter produced byte-for-byte the expected output
  FAIL    a word came back but differs -- the interpreter RAN and computed the wrong thing
  ABSENT  no word came back at all -- the domain faulted or wedged HERE
  UNSCORED a word came back, but the host could not produce an output oracle

ABSENT is the interesting one and is why no bisection is needed: a fault ends the boot, so the
first absent test is the culprit and everything after it is unknown, not failed.
"""
import argparse
import re
import sys

LINE = re.compile(rb"Called dom \((\d+)-th time\) retval = (\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("expected")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    exp = []
    with open(args.expected) as f:
        for line in f:
            idx, name, length, word, how = line.rstrip("\n").split("\t")
            exp.append((
                int(idx), name,
                None if length == "-" else int(length),
                None if word == "-" else int(word, 16),
                how,
            ))
    if not exp:
        sys.exit(f"{args.expected} lists no tests -- nothing could be scored")

    got = {}
    with open(args.log, "rb") as f:
        for m in LINE.finditer(f.read()):
            # the loader prints an unsigned long; a word with bit 31 set may arrive sign-extended
            word = int(m.group(2)) & 0xFFFFFFFF
            got[int(m.group(1)) - 1] = word
    if not got:
        sys.exit(f"{args.log} contains no 'Called dom' line -- the run never reached the loader")

    npass = nfail = nabsent = nunscored = 0
    first_absent = None
    for idx, name, length, want, how in exp:
        word = got.get(idx)
        if word is None:
            nabsent += 1
            if first_absent is None:
                first_absent = (idx, name)
            continue
        if want is None:
            nunscored += 1
            if args.verbose:
                print(f"UNSCORED {idx:3d} {name}  [{how}]")
            continue
        raised = bool(word & 0x80000000)
        if (word & 0x7FFFFFFF) == want:
            npass += 1
            if args.verbose:
                print(f"PASS   {idx:3d} {name}{'  (raised, as expected)' if raised else ''}")
        else:
            nfail += 1
            got_idx, got_len, got_hash = (word >> 20) & 0x7FF, (word >> 16) & 0xF, word & 0xFFFF
            note = "" if got_idx == idx else f"  INDEX MISMATCH: domain says {got_idx}"
            print(f"FAIL   {idx:3d} {name:<34s} want len%16={length % 16} hash={want & 0xFFFF:#06x}"
                  f"  got len%16={got_len} hash={got_hash:#06x}"
                  f"{'  raised' if raised else ''}{note}  [{how}]")

    if first_absent:
        idx, name = first_absent
        print(f"\nABSENT from test {idx} ({name}) onward: the domain stopped returning there.")
        print("That test is the fault site; the ones after it were never attempted.")
    print(f"\n{npass} pass, {nfail} fail, {nabsent} absent, {nunscored} unscored,"
          f" of {len(exp)} tests")
    return 1 if (nfail or nabsent) else 0


if __name__ == "__main__":
    sys.exit(main())
