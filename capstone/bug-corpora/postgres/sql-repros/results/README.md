# Results

9 defects in PostgreSQL 17.5 and its contrib extensions, each a `trigger.sql`
run against a real stand-alone backend, measured on three arms on 2026-10-06;
case 03 re-measured on all three on 2026-10-10, and every detection now has a
negative control beside it.

| arm | detected | scored | out of denominator |
|---|---:|---:|---:|
| `spatial` (base Capstone) | **2** | 9 | 0 |
| `sublet` (Capstone + Sublet) | **5** | 9 | 0 |
| `cheribsd-revocation` (purecap) | **2** | 9 | 0 |

Every denominator is the cases *that arm can run*, and nothing is out of one.

## Every detection has a control, and one of the controls was wrong

Each of the five cases that any arm detects has a `control.sql`: the same
statement with one value moved to the safe side of the boundary the defect
crosses, which must COMPLETE on the same image and the same fixture. All of
them now do, on all three arms.

**Case 03's first control was wrong, and it cost the row a withdrawal.** On
2026-10-08 the sublet row was withdrawn because the control faulted at the
same instruction as the trigger. The control was 64 OR-variants against the
trigger's 66, which the file's own arithmetic put at 64512 bytes under a
65535 ceiling. The arithmetic was wrong. The uint16 that wraps is the
per-level `totallen` in `ltree_io.c:539-546`, and it accumulates
`MAXALIGN(LVAR_HDRSIZE + len)` per variant, not the `MAXALIGN(len) +
LVAR_HDRSIZE` that `LVAR_NEXT` walks. A 1000-character variant therefore
costs 1024 bytes wherever `MAXIMUM_ALIGNOF` is 16 -- which is all three arms
-- so 64 variants came to 65552 and wrapped too, by 17 bytes. The control was
sanity-checked against a host whose `MAXIMUM_ALIGNOF` is 8, where 64 variants
really are safe, which is how the error survived.

**The threshold is now measured on each arm rather than computed.** Each build
is asked what a variant costs on it, in band: the difference between
`pg_column_size` of a 2-variant and a 3-variant lquery is the per-variant
cost, 1008 on the host and 1024 in the purecap guest. Then the count is
bisected, one backend per input:

| build | per-variant cost | first count that faults | one below |
|---|---:|---:|---:|
| host, `--enable-cassert` + ASan | 1008 | 65 (heap-buffer-overflow) | 64 clean |
| `cheribsd-revocation` | 1024 | 64 (SIGPROT) | 63 clean |
| `spatial` | 1024 | 64, at the trigger's own instruction | 63 clean |
| `sublet` | 1024 | 64, at the trigger's own instruction | 63 clean |

Every boundary sits exactly where `16 + N x cost > 65535` puts it. This is
also what answers the objection the withdrawal was made on: cases 02 and 03
fault at the same instruction on sublet, so the address distinguishes nothing
-- but that instruction fires at 64 variants and not at 63, which is the wrap
and nothing else. `CREATE EXTENSION ltree` alone exits 0 in the purecap guest,
so the extension script is not what faults there.

`control.sql` is now 48 variants, 49168 bytes, a margin of more than fifteen
variants, and it completes on all three arms while the trigger faults on all
three. **Case 03 is a detection on all three arms and the withdrawal is
lifted.** Case 03's three runs -- trigger, control, and both threshold probes
-- were run on one image per arm, so each control qualifies the detection it
is paired with.

One fault on sublet remains open and unexplained: `CREATE EXTENSION ltree`,
which is why the case runs against a fixture that already carries the
extension. The purecap guest creates it without faulting, so it is not the
extension script. It belongs to whoever owns the Sublet runtime rather than to
this corpus. The second open fault -- "a plain large lquery parse" -- was this
broken control, and is closed.

## The result is the sublet column

All nine defects are palloc clients, so all nine are nested by the allocator
that serves them: the damage stays inside a chunk that AllocSet carved out of
a block it took from malloc, and an arm whose bounds are the malloc block has
nothing to check. Sublet bounds each sub-allocation and reports 5 of 9 where
base Capstone reports 2 of 9 and the purecap guest 2 of 9.

Case 09 was outside both Capstone arms until 2026-10-06, because it reached
the defect through pgcrypto and no OpenSSL is cross-compiled for capstone64.
It now runs the same three statements as a C caller against pgcrypto's real
`PGP_Context`, so all three arms measure it.

## What produced these numbers

| arm | runner | platform |
|---|---|---|
| `spatial`, `sublet` | `shared/run-arm.py` | Capstone application VM |
| `cheribsd-revocation` | `shared/run-cheribsd.sh` | CheriBSD 15.0-CURRENT riscv64-purecap, QEMU |

`matrix.tsv` here is the three arms combined; each `<arm>-<stamp>/` holds that
run's own matrix, its `inputs.json` and a log per case. A directory name says
what was run: `-control-` ran each case's `control.sql`, `-probe-N64-` ran
that threshold probe, and a plain stamp ran `trigger.sql`; `inputs.json`
records it as `ran` as well. `inputs.json` beside this file carries all three
arms, including each image's sha256, the extensions each arm could create, and
the guest's revocation sysctls read back from it.

## Five things that went wrong first, and are worth knowing

Each produced a row that read as a result and was not one.

1. **A case whose extension is absent is not a silent arm.** On 2026-10-05
   case 03 was scored silent on both Capstone arms from an image with no ltree
   in it: a domain cannot dlopen, the build linked neither the module nor its
   control file, `CREATE EXTENSION` failed, the lquery cast never ran, and the
   backend prompt still appeared. The runner now asks each image which
   extensions it can create, by creating them, and a case needing one it
   cannot is `not-applicable`.

2. **A fault during setup is not a detection.** On the sublet arm case 03
   faulted after a single prompt, so the fault was in `CREATE EXTENSION ltree`
   and not in the defect. The runner now compares the prompts seen before the
   fault against the number of setup statements. This check does not work on
   the purecap arm, where a crash loses the whole buffered stdout and every
   faulting run shows zero prompts; there, attribution comes from running the
   setup on its own and reading the exit code.

3. **A trigger that does not parse has not run.** `postgres --single` takes a
   line at a time with no continuation, so the multi-line triggers of cases 07
   and 09 reached it in pieces and produced `syntax error` on all three arms
   while being recorded silent. Both are now one statement per line -- the
   bytes are unchanged -- and a syntax error is now a control failure.

4. **Absence of an error is not evidence of success.** The first preflight ran
   every `CREATE EXTENSION` in one session and read "no error" as "created";
   when that session faulted on its second statement, the three extensions
   after it were recorded available although they never ran. Each extension
   now gets its own session and has to be read back out of `pg_extension`.

5. **A control whose arithmetic is unchecked is not a control, and a run
   directory that does not say what it ran will hide it.** Case 03's control
   is above. The CheriBSD runner had no control mode at all until 2026-10-10,
   so a leg launched to collect controls re-ran the defects and wrote them
   into a directory named like a defect run, which is exactly what it was;
   nothing in the output said otherwise. Both runners now name the directory
   after the file they ran and record it in `inputs.json`.

### One row comes from a different build

The eight rows other than case 03 were measured on 2026-10-06, and case 03's
on 2026-10-10, after the build tree in `/tmp` had been rebuilt. The image was
rebuilt from the same pinned tarball (sha256 `fcb7ab38...`, re-fetched and
verified) with the same patch set, but a rebuild is not byte-identical and the
image hashes differ. Each `inputs.json` records which image its rows came
from.
