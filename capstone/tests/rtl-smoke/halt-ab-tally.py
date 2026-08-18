#!/usr/bin/env python3
"""Tally the pre-declared halt-control A/B, and refuse to guess.

The arm of each boot is read from the TRANSCRIPT ITSELF -- the presence or absence of the
`EARLY HALT CONTROL` line -- and never from a filename or a remembered launch command. Every
mis-assignment this campaign has suffered came from trusting a name over content, so a boot
whose arm cannot be determined is an ERROR, not a quiet skip.

Usage:  halt-ab-tally.py /tmp/capstone/newbit-1{4,5,6,7}.txt ...
        (pass the FULL transcripts, not the -scoped ones; the marker is only in the full log)
"""
import pathlib, re, subprocess, sys
from math import comb

HERE = pathlib.Path(__file__).resolve().parent


def fisher_one_sided(a, b, c, d):
    """P(arm2 >= c events) under the null. Table [[a,b],[c,d]]."""
    n = a + b + c + d
    r, c1 = c + d, a + c
    return sum(comb(c1, x) * comb(n - c1, r - x) / comb(n, r)
               for x in range(0, min(r, c1) + 1) if x >= c)


def arm_of(full_text):
    return "ON" if "EARLY HALT CONTROL" in full_text else "OFF"


def counts_for(scoped, domain="XU"):
    """Re-use the one classifier, rather than re-implementing its rules here."""
    out = subprocess.run([sys.executable, str(HERE / "s07-rate.py"), str(scoped)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"ERROR: s07-rate.py failed on {scoped}:\n{out.stderr}")
    m = re.search(rf"^\s+{domain}\s+k=(\d+) of n=(\d+)", out.stdout, re.M)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def main(argv):
    if not argv:
        raise SystemExit("ERROR: no transcripts given. This tool has no default set of boots; "
                         "naming them is how the arms stay auditable.")
    tally, rows = {"ON": [0, 0], "OFF": [0, 0]}, []
    for a in argv:
        full = pathlib.Path(a)
        if not full.is_file():
            raise SystemExit(f"ERROR: {full} does not exist")
        scoped = full.with_name(full.stem + "-scoped.txt")
        if not scoped.is_file():
            raise SystemExit(f"ERROR: {scoped} missing -- that boot produced no classified "
                             f"output, so it carries no verdict and must not be silently dropped")
        arm = arm_of(full.read_text(errors="replace"))
        k, n = counts_for(scoped)
        tally[arm][0] += k
        tally[arm][1] += n
        rows.append((full.name, arm, k, n))

    print(f"{'transcript':<22} {'arm':<4} {'k':>3} {'n':>3}")
    for name, arm, k, n in rows:
        print(f"{name:<22} {arm:<4} {k:>3} {n:>3}")

    (k_on, n_on), (k_off, n_off) = tally["ON"], tally["OFF"]
    print(f"\n  ON  (early halt control present): k={k_on} of n={n_on}")
    print(f"  OFF (absent)                    : k={k_off} of n={n_off}")
    if n_on == 0 or n_off == 0:
        print("\n  NO TEST: one arm has no reps.")
        return
    p = fisher_one_sided(k_off, n_off - k_off, k_on, n_on - k_on)
    print(f"\n  Fisher exact, one-sided (ON worse): p = {p:.4f}")
    if min(n_on, n_off) < 16:
        print(f"  UNDERPOWERED -- pre-declared target is n=16 per arm, smallest arm is "
              f"{min(n_on, n_off)}. A null here means low power, NOT absence of an effect.")


if __name__ == "__main__":
    main(sys.argv[1:])
