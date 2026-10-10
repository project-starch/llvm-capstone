# PostgreSQL 17.5 engine defects, reached through SQL

9 defects in PostgreSQL 17.5 and its contrib extensions, each a `trigger.sql`
run against a real server. Siblings: [`../mmgr-repros`](../mmgr-repros), five
memory-manager defects reduced to C programs against the managers themselves,
and [`../c-repros`](../c-repros), five frontend defects reduced to C against
the real upstream functions; these are not reducible that way, so they use the
`script-trigger` schema described in [`../../SCHEMA.md`](../../SCHEMA.md).

## Classes

| class | cases |
|---|---|
| invariant | 9, 10, 11, 12, 14 |
| integer | 4, 5, 6, 7 |
| spatial | 1, 2, 3 |
| spatial/oob-read | 15, 16 |
| type-confusion | 17, 18 |
| null-deref | 8 |
| wrong-answer | 13 |
| authz | 19 |

**Six of the nineteen are what a sanitizer can see.** The rest are integer
wraparound, broken invariants, type confusion, an authorization failure and a
wrong answer -- defects with no memory error to detect. A corpus filtered to
sanitizer-visible defects would keep six of these and call the other thirteen
clean, which is the reason this corpus records the class per case.

## Arms

| arm | configuration (`tools/arms.json`) | what it is |
|---|---|---|
| `virtual-malloc` | `virtual-mallocng` | the server built for the virtual Capstone profile (`ports/common/application/build-virtual.sh postgres`): a Linux process under `capstone-vexec`, malloc is musl mallocng run locally with exact bounds and lifetime retirement; palloc chunks unbounded inside their block |
| `virtual-pg-pools` | `virtual-mallocng-pg-pools` | the same with `PGSU_NESTED=sublet`: the memory-context port's patch 0003 makes every palloc chunk a child lifetime of its block (`CDERIVE`), bounded to the request and revoked by `pfree`/`repalloc` (`CREVOKE`) |
| `cheribsd-revocation` | -- | stock CheriBSD purecap, revocation at the platform default (`shared/run-cheribsd.sh`) |

On the virtual profile the runner takes `--virtual-kit <platform>` and `--image
<build-virtual.sh OUT>/image/postgres.dom`, stages the image, the share and the fixture cluster, and
runs every session -- each on a fresh copy of the cluster, as `nobody` -- in one boot.

`shared/run-arm.py` reads what an image IS from the manifest `build-virtual.sh` wrote beside it --
its `nested` and its heap -- and refuses an arm whose configuration needs another build. Before any case
it runs the configuration's controls inside the server through pgcorpus_reach's
`corpus_control()` (a write past and a read after free, through malloc and, on the nested arm,
through palloc), then each `trigger.sql` on a fresh copy of the fixture. It reports one Observation
per run to the shared judge (`tools/verdicts.py`, SCHEMA.md "Verdicts"):

* a SQL trigger has no marker before its access, so **reached** is the trigger statement running
  after the case's CREATE EXTENSION lines, counted by backend prompts, and the evidence says so;
* a fault counts as the defect's only when the case's `control.sql` -- the same statement below the
  defect's threshold -- completes on the same image in the same invocation, or when it lies in a
  function `case.json` `fault_sites` justifies. Five cases have a control (01, 02, 03, 04, 06). A
  fault with neither is NO-READING, `unattributed`. Case 03's sublet fault was recorded that way on
  2026-10-10 because its control faulted at the same instruction -- and the control was what was
  wrong. 64 OR-variants wrap the same uint16 the defect wraps wherever `MAXIMUM_ALIGNOF` is 16,
  which is every arm here, so it crossed the boundary it was written to stay under. The control is
  now 48 variants, the threshold is measured per arm instead of computed -- it faults at 64 and
  completes at 63 on all three arms, which is where the wrap is -- and the fault is attributed;
* a silence is MISSED only when the controls behaved, and its evidence carries what the trigger's
  directives showed:

      -- EXPECT-ERRORS: N     a correct build rejects N statements
      -- EXPECT-ABSENT: <re>  a correct build can never print this

CheriBSD runs each case **twice**: the plain purecap build gives the verdict, and a separate
`--enable-cassert` purecap build supplies the reachability witness through PostgreSQL's own
`Assert()` and its `MEMORY_CONTEXT_CHECKING` chunk sentinel. Its runner is not yet on the shared
judge.

## Measured 2026-10-11

On the Sublet platform (QEMU `af37cc32`, module `a1c6cb6b`, launcher `3975314a`), the
servers built from c775cdf0, bundles in `results/2026-10-11-virtual`, every control as
declared:

| arm | CAUGHT | MISSED | NO-READING |
|---|---|---|---|
| `virtual-malloc` | 02, 03 | 01, 04-09 | -- |
| `virtual-pg-pools` (patch 0003) | 01, 02, 03, 04, 06 | 05, 08 | 07, 09 (unattributed) |

07 faults in `pg_popcount_optimized` during the CREATE INDEX whose picksplit the defect
reads past a signature in, and 09 in `memcpy` in the pgcrypto session-key path; neither
case has a control or a declared fault site, so the judge does not count them, and
naming a site after the run would fit the instrument to the reading. The exact-request
bounds of 93971f29 (`results/2026-10-11-exact`) also faulted 05, and faulted 07 in
`nsphash_lookup`: PostgreSQL's word-at-a-time string hash reading the aligned word that
holds a terminator, a false positive that `MAXALIGN` bounds removed.

## What these results do not say

`results/matrix.tsv` and the `results/*-20261006-*`, `-20261008-*` and `-control-20261010-*`
directories are the runs scored by the runner of their day, which counted any fault after the setup
as detected and used corpus case 02 as its mechanism gate -- a spatial defect the spatial arm
catches too, so it could not show that the sublet arm's pools were active.

Our own constructed memory-context demonstrators are deliberately not here.
They are this project's fixtures, and `../../README.md` places those under
`ports/*/security-tests` rather than in bug material.

## History

Until 2026-10-11 the corpus also ran a `spatial`/`sublet` pair as physical Capstone application
domains (`app-level0`, `app-level0-pg-nested-sublet`; the `spatial-*`, `sublet-*` and
`sublet-control-*` results), and `virtual-pg-pools` used the memory-context port's lifetime
adapter over a separately granted 64 MiB arena (`results/2026-10-10-virtual`). `CDERIVE`/`CREVOKE`
made the adapter unnecessary; the pair's configurations are removed and their results stay as
recorded.
