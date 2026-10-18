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

## Measured 2026-10-18, case by case

Bundles in `results/2026-10-18-review`, on the emulator this repo pins
(`capstone/capstone-qemu d35af47b163c`) with a compiler carrying the C-80 fix, against a
cluster from `ports/postgres/app/build-native16-fixture.py`. Every control as declared,
including the four the pools arm runs through both `malloc` and `palloc`:

| arm | CAUGHT | MISSED | NO-READING |
|---|---|---|---|
| `virtual-malloc` | 02, 03, **08** | 01, 04, 05, 06, 07, 09 | -- |
| `virtual-pg-pools` (patch 0003) | 01, 02, 03, 04, 06, **07**, **08** | 05 | 09 (unattributed) |
| `cheribsd-revocation` | 02, 03, **08** | 01, 04, 05, 06, 07, 09 | -- |

**Two of those cells are false positives and the four-cell tables withdraw them**: 08 on
`virtual-malloc` and on `cheribsd-revocation`. 08 is this corpus's only out-of-bounds
WRITE, and a write leaves damage behind. The pools arm faults on the store itself
(`cause=28` in `varstr_levenshtein_less_equal`); the other two let it succeed and fault
later, in `AllocSetReset`, on allocator bookkeeping the store had cleared the tag of --
`cause=24` on the virtual arm and `code=PROT_CHERI_TAG` under `ktrace -t s` on CheriBSD.
Counting them the arms read 3 / 9, 7 / 9 and 3 / 9; withdrawn, 2 / 9, 7 / 9 and 2 / 9.
`tools/verdicts.py` would score the first automatically if `virtual-mallocng` declared
`catch_causes` as SQLite's configurations do, but that configuration is shared by eleven
corpora across four lanes, so it is left to them.

### What the 2026-10-18 review changed

**08 was not reaching its defect.** Its two cost triples overflow and return a negative
distance, which is upstream's "nonsensical results", but `levenshtein.c:159` takes
`if (max_d >= max_theo_d) max_d = -1`, which leaves `stop_column` at its default `m + 1`
(`:138`), and the store at `:240` is guarded by `stop_column < m` with `m` already
incremented. The three MISSED verdicts it carried measured nothing. The trigger now
carries two rungs solved from that arithmetic, `stop_column = -1` and `-16`, and a
`control.sql` that reaches the same store at a valid index.

A ladder across the chunk boundary settles what the arm sees. `prev` is
`palloc(2 * (m+1) * sizeof(int))` and the chunk capability covers
`[data - 32, data + MAXALIGN(request))`, so the store is inside the bound until
`stop_column <= -9`. Probed on both virtual arms, eight rungs: the pools arm completes at
-1, -4 and -8 and faults at -9, -12, -16, -32 and -64, every one at the same instruction;
`virtual-malloc` completes at every rung but -16. -32 completes and -36 faults, so the
boundary is measured rather than assumed, and base missing the store at every rung is
what makes the pools arm's catches chunk-level rather than block-level.

**07 was faulting at its defect all along and could not be attributed.** The chain has no
branch: `sizebitvec(GETSIGN(cache[j].sign), siglen)` at `:900` and `:913` inside
`gtrgm_picksplit`, `:651` returns `pg_popcount`, `pg_bitutils.h:360` returns
`pg_popcount_optimized`, which performs the loads; `sizebitvec` and `hemdistsign` are
static and inlined, so `case.json` declares the two symbols that can carry the pc. Its new
`control.sql` builds the same index with sparse rows only, so no signature goes all-true
and the branch holding the defect is never entered.

**05 and 09 are the two the pools arm does not catch, for different reasons.** 05's
over-read leaves the request and stays inside `MAXALIGN(5) = 16`; the exact-request bounds
of 93971f29 (`results/2026-10-11-exact`) faulted it in `ascii`, and also faulted 07 in
`nsphash_lookup` -- PostgreSQL's word-at-a-time string hash reading the aligned word that
holds a terminator, a false positive `MAXALIGN` bounds removed. That trade-off is a
configuration choice with a measured cost on both sides. 09's `memcpy` overruns
`sess_key[32]`, a member of the 184-byte `PGP_Context`, into the next member: the chunk
bound is the struct and is correct, `palloc` never saw `sess_key`, and the case faults in
the same place on both bound settings, so bound width is irrelevant to it.

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
