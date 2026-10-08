# Results

9 defects in PostgreSQL 17.5 and its contrib extensions, each a `trigger.sql`
run against a real stand-alone backend, measured on three arms on 2026-10-06.

| arm | detected | scored | not applicable |
|---|---:|---:|---:|
| `spatial` (base Capstone) | **2** | 9 | 0 |
| `sublet` (Capstone + Sublet) | **5** | 9 | 0 |
| `cheribsd-revocation` (purecap) | **2** | 9 | 0 |

Every denominator is the cases *that arm can run*, and all three are now
whole.

**Case 03 on `sublet` was measured against a fixture cluster that already
carried ltree, and that is not the same as the arm being able to create it.**
`CREATE EXTENSION ltree` still takes a capability fault there. The defect is
in the lquery parser rather than in extension creation, so the extension was
created once on the `spatial` arm (`shared/make-fixture.py`) and the cluster
handed to `sublet`, which then ran the statement the case is about. The two
runs of 2026-10-08 are a controlled pair: the same image against the plain
fixture reports FAULTS ON CREATE and scores the case `not-applicable`, and
against the ltree fixture scores it `detected`. Only the fixture differs.

That fault is deterministic, unfixed, and worth reporting on its own -- a
capability fault while loading an extension is a finding about the port, not
about any defect in this corpus. The case's `underlying_fault_still_open`
field carries the detail.

Case 09 was outside both Capstone arms until 2026-10-06, because it reached
the defect through pgcrypto and no OpenSSL is cross-compiled for capstone64.
It now runs the same three statements as a C caller against pgcrypto's real
`PGP_Context`, so all three arms measure it.

**The result is the sublet column.** All nine defects are palloc clients, so
all nine are nested by the allocator that serves them: the damage stays inside
a chunk that AllocSet carved out of a block it took from malloc, and an arm
whose bounds are the malloc block has nothing to check. Sublet bounds each
sub-allocation and reports 5 of 9 where base Capstone reports 2 of 9 and the
purecap guest 2 of 9.

## What produced these numbers

| arm | runner | platform |
|---|---|---|
| `spatial`, `sublet` | `shared/run-arm.py` | Capstone application VM |
| `cheribsd-revocation` | `shared/run-cheribsd.sh` | CheriBSD 15.0-CURRENT riscv64-purecap, QEMU |

`matrix.tsv` here is the three arms combined; each `<arm>-<stamp>/` holds that
run's own matrix, its `inputs.json` and a log per case. `inputs.json` beside
this file carries all three, including each image's sha256, the extensions
each arm could create, and the guest's revocation sysctls read back from it.

## Four things that went wrong first, and are worth knowing

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
   fault against the number of setup statements.

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

### One row comes from a different build

Case 03's `sublet` figure was measured on 2026-10-08, after the build tree
that produced the other eight had been cleaned from `/tmp`. The image was
rebuilt from the same pinned tarball (sha256 `fcb7ab38...`, re-fetched and
verified) with the same patch set, but a rebuild is not byte-identical and the
image hashes differ. Both `inputs.json` files record which image each row came
from.
