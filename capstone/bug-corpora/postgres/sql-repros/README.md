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

## Measured 2026-10-18 and 2026-10-21, case by case

Bundles in `results/2026-10-18-review/virtual-malloc` and
`results/2026-10-21-review/virtual-pg-pools` -- `verdict_bundles` names one per arm, and the
pools arm was re-measured on 2026-10-21 to settle case 07's attribution (below). Every
verdict is identical across the two pools runs; only 07's attribution route changed. On the
emulator this repo pins
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
`cause=24` on the virtual arm and, on CheriBSD, `code=PROT_CHERI_TAG`.
**That CheriBSD signal code is not in this corpus.** What the bundle holds for the cell is
the exit status `shared/run-cheribsd.sh` scores -- 162, which its rule at `:203` reads as
`SIGPROT(34)` -- plus the trigger log and the paired control; the log carries no `si_code`,
and the `ktrace -t s` trace the code was read from is not committed under `results/`. The
withdrawal of the cell therefore rests on a reading a reader cannot recheck from this tree.
What IS checkable here is weaker and points the same way: the revocation log ends on the
trigger's last answer with none of the shutdown-checkpoint lines the control's log reaches,
so the backend died after the statements completed rather than at the defective store.
Counting them the arms read 3 / 9, 7 / 9 and 3 / 9; withdrawn, 2 / 9, 7 / 9 and 2 / 9.
Those pools figures put case 09 in the denominator, where its cell is a NO-READING;
`catch-tables.py`, which counts a NO-READING as neither, prints `7 / 8 (+1 ?)` for the
same arm. Both are right about different denominators, and SCHEMA's rule is the tool's.
A key that would score the first automatically, `catch_causes`, is proposed in PR #259 and
exists in no tree here: `tools/verdicts.py` on this branch has no such key, and no
configuration in `tools/arms.json` declares one. Were it to land, declaring it on
`virtual-mallocng` would reach eleven corpora across four lanes, so the declaration is
theirs to make and not this corpus's.

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

**None of those rungs is committed, so the paragraph above is a report and not evidence.**
A rung is not a case: it is the case's statement with one parameter changed, and the way the
corpus documents running one -- `shared/make-threshold-probe.py` writing a `.sql` that
`run-arm.py --only <case> --sql <file>` then runs, as that script's header describes for case
03 -- cannot be followed, because `run-arm.py` has no `--sql` option; its arguments are at
`shared/run-arm.py:383-406`. The generator is also case 03's lquery arithmetic specifically,
not a ladder for this one. Committing this ladder therefore needs a probe path that exists,
and until there is one the two claims that depend only on it -- that the chunk boundary sits
between -8 and -9, and that `virtual-malloc` faults at -16 alone -- stay unverifiable here.
The cells in the bundle do not depend on them: 08's two rungs are in `trigger.sql`, solved
from the source, and each arm's verdict for them is in its own `verdicts.jsonl`.

**07 was faulting at its defect all along and could not be attributed.** The chain has no
branch: `sizebitvec(GETSIGN(cache[j].sign), siglen)` at `:900` and `:913` inside
`gtrgm_picksplit`, `:651` returns `pg_popcount`, `pg_bitutils.h:360` returns
`pg_popcount_optimized`, which performs the loads; `sizebitvec` and `hemdistsign` are
static and inlined, so `case.json` declares the two symbols that can carry the pc. Its new
`control.sql` builds the same index with sparse rows only, so no signature goes all-true
and the branch holding the defect is never entered.

**And on 2026-10-21 that `control.sql` did the attributing, which is what closes this
cell.** `shared/run-arm.py:365` runs the control only
`if o.fault and o.reached and not o.attribution`, so while the site list stood the declared
symbol matched the pc first and the control never ran: the `CAUGHT` rested on a list read
off the source -- a list whose own `fault_sites_why` opened "Derived from the source, not
from a run" -- rather than on the paired measurement. The arm was therefore re-run with
`fault_sites` ABSENT. The fault is then unattributed by site, the runner falls through to
`control.sql`, and the control **completed** on the same image, so the cell is now
attributed by control: `cause=28` in `pg_popcount_optimized`, by `control.sql`.

The declaration turned out to be right, and the run says so rather than assuming it: the
fault lands in exactly the symbol the branch-free chain above predicts, and its pc differs
from the 2026-10-18 run only in load base (`0x3f92b39538` against `0x3fbcb39538`). The site
list was not wrong; it was doing a job a control should do. It is not restored, because the
control is the stronger evidence and a declared site would silence it again on the next run.
Nothing else moved: all nine verdicts are identical to `results/2026-10-18-review`, the arm
tally is unchanged at CAUGHT 7 / MISSED 1 / NO-READING 1, and 07 is the only case whose
attribution route changed, from `function` to `control`.

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

**Seven vp2 bundles beside this corpus ran case 08's OLD trigger, and their 08 cell is not a
measurement.** `results/2026-10-19-vp2c-treeA/vp2-malloc`,
`results/2026-10-20-vp2c-treeA/vp2-malloc-nf2`, `.../vp2-malloc-nomovc`,
`results/2026-10-20-vp2c-treeB/vp2-malloc-nf2`, `.../vp2-malloc-nomovc`, and the two dev
landed on 2026-10-23, `results/2026-10-22-batch-c721c1a2a-b1/vp2-pg-pools` and
`results/2026-10-22-batch-c721c1a2a-b2/vp2-malloc-nomovc`, each record 08 `MISSED`,
"reached ... and completed with no fault" (line 8 of each `verdicts.jsonl`), with the old
trigger's reach shape "(4 prompts, 1 setup line(s))".

The `b2-silicon-tally` rows `sql0` and `pg-pools` come from dev-tree boots, whose
`08/trigger.sql` has been sha256 `9e298de7...` since 2026-10-06 against this tree's
`28586f0c...`, so they ran the old trigger too. Those bundles record no per-case trigger
hash, so for the tally rows that is inferred from the tree they booted and not read from a
record. The tier3 table's output is unaffected either way. They
ran the parameters this review replaced, and `08/trigger.sql`'s own header sets out why those
never reached the defective store: `levenshtein.c:159` takes `max_d = -1`, `stop_column`
stays at `m + 1`, and the guard at `:240` is never true. A silence from a trigger that cannot
arrive is not a reading about any mechanism, on the board or anywhere else. The fixed trigger
faults the same case on the same configuration -- `results/2026-10-18-review/virtual-malloc`
line 8, `cause=24 in AllocSetReset`, attributed by `control.sql` -- and is the one the
CheriBSD bundle also ran, by sha256: `inputs.json`'s `ran_sha256` for 08 is
`28586f0c...`, this tree's `trigger.sql`. Those five bundles are left exactly as recorded;
a bundle is only ever written by a runner, so correcting them means re-running them, and
whether the board arm is re-run is not this corpus's call. Read their 08 row as void.

**Exact bounds: what the record says, and what it does not.** The 2026-10-18 bundles record
`platform.exact_bounds: true` for both arms. The 2026-10-21 pools bundle, which
`verdict_bundles` now declares, **omits the field**: that run went through appvm's session
path, and the session branch of `platform()` never copies it. Whether exact bounds were on
for the 10-21 run is therefore not established by this tree. Its nine verdicts are identical
to the 10-18 pools bundle's, which is consistent with exact bounds but does not prove them.

The same comparison is not one-variable in its runner either. Between the two pools bundles
the runner hash moves `2de3338a2036` -> `292e7440305d` and the judge hash `07c940f428b4` ->
`2ed6701045dd`; the kit, the image and the fixture are the same. A chunk-level catch on the
pools arm is a statement about a bound the hardware applied as requested only to the extent
that the 10-18 record carries forward; nothing here measures the compact-bounds behaviour the
vp2 bundles exercise.

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
