#!/usr/bin/env python3
"""Write one lquery threshold probe: case 03's statement at N OR-variants.

    make-threshold-probe.py <variants> [out.sql]        default: stdout

WHY THIS EXISTS. Case 03 overflows a uint16 while parsing one lquery level, and
a negative control for it is only a control if it stays below that threshold.
The first one did not, and the row was withdrawn on it for two days. What
settled the question was not arithmetic but a bisection: run the same statement
at N and at N-1 variants on each arm and see where the fault appears. These are
the inputs for that bisection, so they are generated here rather than typed
into a shell, and the bisection can be run again on a new target.

THE ARITHMETIC the probe is bisecting, from contrib/ltree/ltree_io.c:539-546:

    cur->totallen  = LQL_HDRSIZE                      -- MAXALIGN(10), so 16
    cur->totallen += MAXALIGN(LVAR_HDRSIZE + len)     -- once per OR-variant

with LVAR_HDRSIZE = MAXALIGN(offsetof(lquery_variant, name)) = MAXALIGN(7). For
the 1000-character variants used here that is 1008 bytes per variant where
MAXIMUM_ALIGNOF is 8 and 1024 where it is 16, so the first count that wraps the
uint16 is 65 and 64 respectively.

DO NOT TRUST THAT NUMBER ON A NEW TARGET. Ask the build what a variant costs on
it, in band -- the difference between these two is the per-variant cost:

    SELECT pg_column_size((repeat('x',1000) || '|' || repeat('x',1000))::lquery),
           pg_column_size((repeat('x',1000) || '|' || repeat('x',1000) || '|' || repeat('x',1000))::lquery);

Measured 2026-10-10: 1008 on the host ASan build (2048 -> 3056) and 1024 in the
purecap guest (2080 -> 3104), whose installed pg_config.h nonetheless says
MAXIMUM_ALIGNOF 8. The measurement is what the arm does.

HOW THE BISECTION WAS RUN. Each probe is one backend against a fresh cluster.
On the Capstone arms, through the corpus runner, which records a probe as
probe-faulted / probe-completed so it can never be read as a verdict:

    make-threshold-probe.py 64 /tmp/N64.sql
    make-threshold-probe.py 63 /tmp/N63.sql
    run-arm.py --arm sublet ... --only 03 --sql /tmp/N64.sql

In the purecap guest, one `postgres --single` per file, read by exit code: 162
is SIGPROT. `CREATE EXTENSION ltree` on its own was run the same way, and
exits 0, which is how the extension script was ruled out as the thing faulting.
"""
import sys

VARIANT_LEN = 1000


def probe(variants, length=VARIANT_LEN):
    if variants < 1:
        raise ValueError("a level needs at least one variant")
    q = "'"
    head = f"repeat({q}x{q}, {length})"
    if variants == 1:
        expr = head
    else:
        expr = f"{head} || repeat({q}|{q} || {head}, {variants - 1})"
    return f"CREATE EXTENSION ltree;\nSELECT ({expr})::lquery;\n"


def main(argv):
    if not 2 <= len(argv) <= 3 or not argv[1].isdigit():
        sys.exit(__doc__.splitlines()[2].strip())
    text = probe(int(argv[1]))
    if len(argv) == 3:
        with open(argv[2], "w") as f:
            f.write(text)
        print(f"{argv[2]}: {argv[1]} variants of {VARIANT_LEN} characters")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main(sys.argv)
