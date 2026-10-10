# The corpus contract

What a case *is*, field by field, so the shape can be copied rather than
re-derived. `tools/check-corpus.py` enforces every rule below; if the two ever
disagree, the checker is the authority and this file is stale.

This file used to live inside the pymalloc corpus and describe only it, while
five of the eight corpora had no checker at all. It now covers every corpus in
this tree and the cross-language corpora in `xlang/`, each of which declares
itself in a `corpus.json` the checker reads. `tools/build-index.py` reads the
same declarations and writes [INDEX.md](INDEX.md), which is the one place that
lists all bug material; no count in it is typed by hand.

## One directory per case

    NN_<upstream-fix-id>_<slug>/
        case.c           the sequence, inside <MACRO>_CASE(NN)
        case.json        machine-readable claims
        PROVENANCE.md    the upstream hunk, quoted; what is real and what is reduced

`<MACRO>` is the corpus's own case macro, declared as `case_macro` in
`corpus.json` (`PYC`, `PG`, `WM`, `MC`, `APR`, `APRB`, `FF2`), and the checker
holds `case.c` and `case.json` to the same number.

**The directory name is metadata.** `NN` is the case number a run selects, so
sorting the tree puts the corpus in run order and `--cases 5` is findable by
eye; the rest names the upstream fix and the case. Case numbers are dense: N
cases are numbered 0..N-1, with no gaps and no duplicates. `shared/defects.c`
carries a CASE INDEX mapping each number to its directory and shape, and the
checker keeps the two in step. Run artifacts are named the same way, so an
archived result tree stays readable away from the corpus:
`05-odict-copy-stale-link-mode1`, not `defect-5-mode1`.

**One program per case, and no `run.sh` beside it.** A capability fault ends
the domain, so a case that provokes one cannot also report results next to it:
one case per run. The sequence lives in the case directory as `case.c`, which
includes `shared/corpus.h` and writes its body inside `PYC_CASE(NN)`; the macro
supplies `pym_replay`, so the file is a complete translation unit that the
port's one-source build seam compiles on its own. Building and running stay
with the target directories, because both need a toolchain and a guest that a
per-case script would have to reinvent twenty times.

`case.c` must declare the number its directory carries. A fixture naming
another case is refused rather than silently run.

## `script-trigger`: a case that is an interpreter script

Some defects are not reducible to a C program against an allocator. A
PostgreSQL engine defect is reached by running SQL against a real server; an
mruby or Perl defect is reached by running a script through the interpreter.
Their oracles are a fault, one of the program's own assertions, or a
differential against what a correct build prints. For those the case directory
carries a trigger script in place of `case.c`, and declares:

    case, upstream_fix, title, consumer, class, trigger, fidelity, arms, status

`class` is what kind of defect it is (`spatial`, `integer`, `invariant`,
`type-confusion`, `wrong-answer`, …) rather than a reduction shape, because
there is no reduction. `trigger` names the file the case runs, and the checker
holds the case to it: the name is per case, not per corpus, so one corpus may
mix `.t` and `.pl`, or `.sql` and anything else, without splitting in two or
renaming upstream files to fit. `allocator_layer` is optional here -- a Perl
case has one, a PostgreSQL integer-overflow case does not. `object`,
`lifetime_ender` and `shape` do not apply either; a case may still carry
`shape` where one genuinely fits. Everything else -- the directory name, dense numbering, the
arms, `live_in_pin` with a `live_proof` -- is as for `case-json`.

## Required fields

| field | meaning |
|---|---|
| `case` | the number a run selects; dense over the corpus |
| `upstream_fix` | the upstream identifier, matching the directory prefix |
| `title` | one line, what the upstream defect is |
| `consumer` | the upstream file the defect lives in |
| `live_in_pin` | whether the defect is live at the pinned version |
| `live_proof` | **how that was established** — a backport, or an inspection with file and line. Never bare assertion |
| `object` | what is freed, in the upstream's own terms |
| `lifetime_ender` | what ends the object's life |
| `shape` | the reduction class; must be one of the shapes the README tabulates |
| `allocator_layer` | which allocator the memory came from |
| `fidelity` | how faithful the reproduction is, stated as a limitation |
| `arms` | one entry per executable arm, see below |
| `status` | what has actually been run, dated. Not a plan |

## Optional fields

`distinguishing` (why this case is not a duplicate of its siblings),
`sibling_issue`, `size_class`, `size_note`, `layer_note`, `note`,
`allocator_consumed`, `channel`, `nested`, `nested_why`, `citation_constraint`, `fault_sites`,
`fault_sites_why`, `duplicate_of`. Optional means
optional:
the checker does not invent them, and absence is not a defect.

`nested` is a **boolean**, and `nested_why` its one-sentence reason. The axis is
**who allocated the object**: an inner allocator's sub-allocation is nested, a
direct `malloc` is not. It is **not** which bound the access crosses — collapsing
those two produced a retraction on 2026-10-05.

It exists because the inventory's headline nesting share was being computed from
`allocator_layer` **prose**. On 2026-10-06 a script doing that put 11 of 25
spatial cases into an "unclassified" bucket and reported **44%**; the correct
figure is **60%**, so publishing the script's number would have been wrong by 16
points — the unanswered probe was reading as "not nested". A tally that a
headline depends on should come from a field, not from a substring match over
sentences that are free to be reworded.

`duplicate_of` names a sibling case directory recording the SAME upstream defect -- a backport of a fix another case
already carries, equal by `git patch-id`. The case is kept, so numbering and citations stay valid; the checker
requires the sibling to exist and not to be a duplicate itself, and `tools/catch-tables.py` leaves the case out of
every table. First use: ffmpeg/plain-heap-repros 13 (2026-10-10 audit).

`citation_constraint` records that a case's upstream commit cannot be quoted freely — in practice
that its **subject line names a person**, so the fix may be cited **by hash and path only**. The
naming rule in this tree is absolute for committed files, so the constraint belongs in the case
rather than in a reader's memory. The first instance, memcached `d5d9ff0`, sat undispositioned in a
triage doc for exactly that reason: what made it awkward was written in prose somewhere else.

`allocator_consumed` is the companion to `allocator_layer`, and a corpus that
records one without the other has half of axis 2. The layer says which
allocator the memory came from; `allocator_consumed` says whether the damage
stayed inside one block of that allocator, which is the difference between a
defect the system allocator cannot see and one it can. The two do not share an
empty count -- a case can have a measured layer and no consumed verdict -- so a
corpus reporting coverage must count them separately rather than quoting one
number for both.

A field that is meaningful in exactly one schema belongs in that schema's
required list, not here. `trigger` is the worked example: it is required for
`script-trigger`, and keeping it out of the optional set is what lets the
unknown-field check still catch a stray `trigger` on a `case-json` case.

`fault_sites` names the functions a fault may land in and still be this defect's, for a case whose
stale access has no labelled probe -- a double free faults inside the manager, a copy routine faults
inside `memcpy`. It needs `fault_sites_why`, and the reason must come from the source or from an
earlier independent run, never from the fault it is about to admit: a site chosen after seeing where
a run faulted would turn every crash into a catch.

## Arms

Each arm declares an **oracle**, never an outcome. Outcomes live in `status`
and in `results/`. An arm that is declared but not written says so with
`"status": "not written"` instead of an oracle, so the gap is visible rather
than silently absent.

An unwritten arm may carry **`not_run_reason`**: one or two sentences saying
why it was not run and whether it can be. A bare "not run" is the one cell a
reader cannot interpret -- it covers a case that is impossible here, a case
waiting on a build change, and a case nobody got to, and those three carry
different weight in a denominator. Without the field the reason survives only
in whoever ran it. The checker does not yet enforce this: an unwritten arm
short-circuits before its keys are examined, so a misspelt `not_run_reason`
passes silently.

| arm | target | oracle |
|---|---|---|
| `spatial` | Capstone domain | the sequence completes |
| `sublet` | Capstone domain | fault at the labelled read probe, with `cause` |
| `native-detect` | host | ASan, built `-O0`, run fixed then buggy by `tools/run-native-asan.py`. A silence is a reading only when the buggy arm printed `VERDICT DEFECT-REPRODUCED` and the same run's positive controls -- a read past, and a read after free of, a heap block the size of the corpus's arena or block -- were reported. Until 2026-10-09 this arm was 'declared, not written' as tautological; it is measured now, because an argument is not a reading |
| `sysalloc-none` | Capstone domain | the whole-program run completes; the first-fit heap is built with `-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`, so only tag integrity and the arena's bounds are left |
| `sysalloc-bounds` | Capstone domain | the same, with the heap as applications get it since PR #170: each allocation bounded, no revocation. **The baseline a catch is measured against** |
| `sysalloc-sublet` | Capstone domain | the Sublet heap: per-object bounds and a revoke on every free |
| `sublet-gc` | Capstone domain | `sysalloc-sublet` plus the program's own Sublet port, so its nested allocator's objects are issued and revoked too |
| `sublet-chunks` | Capstone domain | tshark's chunk allocator linked over the Sublet heap (`HEAP_LOG 24`, 40 MiB arena): the configuration the tshark application runs in |
| `capstone-subobject` | Capstone domain | the `spatial` arm with `-Xclang -fcapstone-subobject-bounds`: an ARRAY-typed field that is not the last member is narrowed to the field. The tool's `subobj` control must be CAUGHT in the same boot |
| `cheribsd-subobject` | CheriBSD purecap | stock CheriBSD with `-Xclang -cheri-bounds=subobject-safe`: a pointer taken to a member is narrowed to it (size-0/1 trailing arrays excepted). The shared field-crossing control must die by `SIGPROT` in the same boot |
| `capstone-carve-bounds` | Capstone domain | the carved corpus only: `-DFFC_CARVE_BOUNDS` makes `ffc_carve()` narrow each carved region to its own extent -- the remedy at the carving code. Fixed arms run under the same narrowing |
| `cheribsd-carve-bounds` | CheriBSD purecap | the same with `cheri_bounds_set`; the run prints each carve's granted length, and a carve control must fault in the same boot |
| `sublet-malloc` | Capstone domain | the nested corpora: Sublet ONLY as the system allocator, the program's nested allocator stock -- memcached's ledger in mode 2 (a chunk carries its page's bound, nothing revoked until a page or object is given back), FFmpeg's pools unported on the Sublet heap (`poolstock`). Column 2 of the per-bug table |
| `sublet-full` | Capstone domain | a plain case built in the program's whole Sublet configuration: the Sublet heap plus the program's nested-allocator port linked and initialised (tshark's is `sublet-chunks`). The bug's path never enters the nested allocator, so this equals `sublet` by construction; the run confirms no interaction. Column 3 for plain cases |
| `sublet-carve` | Capstone domain | the carved corpus: `-DFFC_SUBLET_CARVE` takes the block LINEAR from the Sublet heap and splits it into one region per carve, each issued as a bounded alias (a re-carve revokes the old aliases); a temporal carve control must fault in the same boot. Column 3 for carved cases |
| `sublet-svheads` | Capstone domain | the same for Perl: `sysalloc-sublet` plus Perl's SV head arena |
| `sublet-pymalloc` | Capstone domain | the same for CPython: `sysalloc-sublet` plus pymalloc's pools and arenas, via patch 0014. The difference between this arm and `sysalloc-sublet` is what sublets the nested allocator rather than only the allocation under it, which is the whole comparison a nested/non-nested corpus exists to make |
| `virtual-malloc` | virtual Capstone (`capstone/runtime/virtual`) | the case on the virtual profile, its nested allocator stock: the application runs under `capstone-vexec` with musl mallocng compiled for Capstone and run locally, which bounds each object exactly and retires its lifetime on free. Configuration `virtual-mallocng` in `tools/arms.json`; not the physical Sublet heap |
| `virtual-pg-pools` | virtual Capstone | `virtual-malloc` with PostgreSQL's memory contexts protected by the memory-context port's patch 0003: every chunk a child lifetime of its block (`CDERIVE`), revoked by `pfree`/`repalloc` (`CREVOKE`). The server is built with `build-virtual.sh postgres`, `PGSU_NESTED=sublet` (configuration `virtual-mallocng-pg-pools`); the replay with `PG_SUBLET=ON` (`virtual-mallocng-replay-pools`) |
| `virtual-nested-pools` | virtual Capstone | `virtual-malloc` with the program's nested allocator on its own pool port (SQLite: memsys5 on `sublet-3220000-memsys5.patch`, its pool lent linear by the virtual heap; wmem: the wmem port's patch 0001, every `block` and `block_fast` object a child lifetime of its block). The configuration in `tools/arms.json` names which |
| `cheribsd-revocation` | stock CheriBSD purecap | the case under CheriBSD's malloc with heap revocation on (the platform default, or forced per process where the corpus says so), a revocation control faulting at its labelled load in the same boot. `NOT-REISSUED` is the plain-temporal reading for a freed chunk HELD in quarantine (the stale pointer followed, the chunk never reissued); column 1 of `tools/catch-tables.py --board` counts it as caught and shows it apart |
| `native-fix-differential` | host | the buggy and the fixed build of the same reduction, run plainly: `TWO-SIDED` means the buggy run reproduces the defect and the fixed run does not. It shows the reduction is about the fix; it is not a protection mechanism and no table counts it as one |
| `backing` | Capstone domain | FFmpeg pool corpora: revocation only at the pool's BACKING allocation (mode 1), so a buffer the pool never frees is never revoked. `n/a` where there is no block distinct from the object |
| `sublet-port` | Capstone domain | FFmpeg pool corpora: `case.c` unchanged against the Sublet port of FFmpeg's own pools (app port, `FFAPP_POOL=sublet`); a fault counts only at an instruction the case's line table puts on a line that dereferences the stale pointer. Column 3 for pooled cases |
| `host-asan` | host | the upstream function built with ASan at the pin: the oracle is the report (kind, access, region) it must produce. Liveness evidence and the row corpora's only arm, not a protection column |

The four above are the glossary's system-allocator arms
(`docs/ref/runtime-terms-glossary.md` section 6). They are one image each of the
same source and differ only in the heap the image links, so a difference between
them is the heap's protection and nothing else -- which is the whole reason the
unprotected one exists. It is not a C baseline: tag integrity is in the hardware
and cannot be switched off, so a defect that reads a pointer out of overwritten
memory faults there too. The nearest C baseline is a host run, recorded per case
in `live_proof`.

The two protected arms name an instruction, not merely a fault. That is the
point: the process status alone cannot tell this corpus reproducing from an
arbitrary crash, a bounds fault, a permission fault, a fault elsewhere, or the
allocator refusing the request.

## Rules a reader can rely on

1. **Paired arms over a single binary.** Every case runs twice against the same
   program, which picks its arm at runtime, so the two arms differ in exactly
   one thing.
2. **The oracle names the instruction.** A fault is accepted at the labelled
   probe and nowhere else, and the expected address is published by the run
   rather than hardcoded, so a relink cannot turn the check into a tautology.
3. **Setup is proven before the marker.** The `CHECK`s that establish the block
   really came back at the same address run *before* the ready marker, so the
   marker's presence is itself evidence that reuse happened.
4. **Infrastructure failures are not measurements.** A boot that produced no
   result exits 75 with no verdict, rather than recording a failure that reads
   like the defect not reproducing.
5. **Every oracle has a negative control.** `--negative-control` corrupts the
   fixture so the program refuses it before any case runs, executes every
   selected arm anyway, and exits 0 only when every oracle reports a failure.
   A suite whose oracles cannot say FAIL proves nothing by saying PASS.
6. **Results are summaries, never captures.** `results/<stamp>/` holds a
   `matrix.tsv`, an `inputs.json` and a README. Raw serial or console logs are
   not committed.

## Verdicts: one judge, one original

A corpus that declares `arm_configurations` and `verdict_bundles` (PostgreSQL's three, CPython's pymalloc-repros and
SQLite's engine-repros as of 2026-10-10; none of FFmpeg's, tshark's or memcached's yet) keeps
its results under one contract, implemented in `tools/verdicts.py` and tested in
`tools/test_verdicts.py`:

* **A runner reports, it does not decide.** Each run of a case on an arm becomes an Observation:
  did the case reach the marker before its defective access, did it complete, where did it fault,
  and what ties the fault to the defect (`probe`, a declared `fault_sites` function, or a paired
  `control` run). The verdict comes from the shared judge.
* **Three verdicts.** `CAUGHT` needs reached, a fault, and an attribution. `MISSED` needs reached,
  completed, and the configuration's controls behaving in the same invocation. Anything else is
  `NO-READING` with one closed reason (`infra`, `build-failed`, `not-reached`, `setup-fault`,
  `control-failed`, `unattributed`, `out-of-denominator`, `inconclusive`) and counts as neither.
* **An arm name is a label; the configuration is the thing measured.** `tools/arms.json` describes
  each configuration and the controls a run of it must show -- a use-after-free control that
  COMPLETES is what shows an arm does not revoke. `arm_configurations` maps the corpus's arm names
  onto it, so two corpora whose `spatial` differ cannot be summed as one.
* **The bundle is the only original.** `results/<stamp>/<arm>/` holds `verdicts.jsonl` (each
  Observation with its verdict) and `inputs.json` (configuration, and the compiler, QEMU, firmware,
  kernel and runner hashes, each said or said to be `unrecorded`). `verdict_bundles` names the
  current bundle per arm; `tools/derive-verdicts.py` writes each `case.json` arm's `verdict`,
  `verdict_reason`, `verdict_note`, `verdict_from` and `results/verdicts.tsv` from them, and
  `check-corpus.py` fails when either differs from what it would write -- re-judging every stored
  Observation, so a change to the judge or to `arms.json` cannot leave an old verdict standing.

## The corpus declaration: `corpus.json`

One per corpus, at its root. It is what makes a corpus findable, countable and
checkable from outside, and it is the only input to the generated index.

| field | meaning |
|---|---|
| `program`, `boundary`, `title` | what the corpus is about: the upstream program, the allocator or API boundary its cases cross, and one line of scope |
| `upstream` | `{version, port}` -- the release the cases are built against, and the port component that pins it. Absent where each row pins its own commit |
| `cases` | how many cases the corpus has. The checker counts the tree and refuses a mismatch |
| `case_schema` | `case-json` (a reduction with `case.c`), `script-trigger` (a defect whose trigger is an interpreter script rather than a C reduction; each case names its own file in `trigger`), `sqlite-row` (a binding row carrying the provenance ledger's columns), or `xlang-row` (a shim row, declaration-level only) |
| `case_macro`, `case_glob`, `case_exclude`, `case_number_base`, `case_doc` | how cases are named and found, where the defaults do not fit. Case numbers are dense from `case_number_base` (0, or 1 for the SQLite rows) |
| `case_table`, `case_dir_column` | for `xlang-row`: the row table that is the corpus's authority, and the column naming each case directory |
| `required_arms`, `arm_keys` | the arms every case must declare, and any arm whose oracle carries more than `oracle` (the pymalloc corpus records a `cause` on `sublet`). **Order is not significant and is not checked**: reordering a corpus to match another would churn every case file for nothing |
| `status` | `planned`, `built`, `measured` or `triaged`. A corpus with no cases may only be `planned` or `triaged` |
| `live_in_pin_recorded` | whether the cases record liveness at all. If false, a `live_in_pin_note` must say why, and no case may carry the field |
| `expect_live_in_pin` | the expected `{true, false, not_asserted}` split. The checker recomputes it from the cases and refuses a mismatch, so this number cannot drift |
| `expect_provenance` | only where not every case has a `PROVENANCE.md`: how many do. The checker holds the corpus to exactly that count |
| `runners`, `checker`, `inventory`, `evidence`, `related` | repository-root-relative paths. Every one must exist |
| `advisories` | one entry per advisory, with its aliases in the same string, so counting entries counts advisories rather than identifiers |
| `shape_table`, `shape_prose` | whether the README carries a shape table that must partition the cases, and any prose claim about its size |
| `arm_configurations` | arm name -> configuration in `tools/arms.json`; see "Verdicts" above |
| `verdict_bundles` | arm name -> the current `tools/verdicts.py` bundle, relative to the corpus |
| `note` | anything a reader needs that no field above holds |

**`live_in_pin` is three-valued.** `true` and `false` both need a `live_proof`
that says how it was established -- a backport, or an inspection with file and
line, never a bare assertion. `null` is the third value and means *recorded and
deliberately not asserted*: the APR and bucket corpora pin allocators rather
than a shipped httpd, and the SQLite rows are defects in bindings rather than in
the engine. `null` needs a `live_proof` or a `live_note` giving that reason.
Absence of the field is not a value; the corpus declares that it does not record
liveness instead.

## The port declaration: `port.json`

One per port component, checked by `tools/check-ports.py`. A component's
upstream version used to be readable only by reading its build script, which is
how `experiments/study/catalog.json` could carry PostgreSQL 17.0 while the
ported backend was 17.5.

| field | meaning |
|---|---|
| `program`, `title` | the upstream program, and one line of scope |
| `role` | `full-application` (the whole program runs), `allocator-component` (one allocator, replayed or driven), `platform-build`, `domain-libc`, or `census` |
| `upstream` | `{version, commit?, pin_source}`. `version` may be a list where a component pins more than one release |
| `upstream.pin_source` | `{file, grep, version?}`, one or a list: the text that actually decides the pin. The checker requires `grep` to appear verbatim in `file`, and the declared version or commit to appear inside `grep` -- so the declaration cannot become a second source of truth, and a bump to the recipe fails the check until the declaration follows |
| `targets` | where it runs: `capstone-domain`, `cheribsd-purecap`, `linux-guest`, `silicon`, `native` |
| `workload`, `evidence`, `corpora`, `related`, `status`, `note` | the qualifying workload, result bundles, the corpora whose cases are its own, neighbouring components, and anything else a reader needs. Paths must exist |

**`role` says what a component is, so a path never has to.** Three components were
renamed to the rule in [`../ports/README.md`](../ports/README.md#naming) on 2026-09-28 —
`app/` for the complete application — and four are deferred there with their reasons.
Archived result bundles keep quoting the old paths, because a `build-manifest.json`
records which directory a measured binary was built from and is evidence rather than a
link; `../ports/renames.json` is the resolution table, and `check-ports.py` fails if a
listed bundle stops quoting the old path or a renamed component loses its declaration.

## Naming, on this side

A corpus is `<program>/<boundary>-repros/`, where the boundary is the allocator or
API its cases cross: `pymalloc-repros`, `mmgr-repros`, `wmem-repros`, `pool-repros`,
`apr-pool-repros`, `bucket-repros`, `capi-repros`, `allocator-repros`,
`gc-slot-repros`. The name says what a case crosses, never where the defect was
reported: `sqlite/cve-repros` was renamed to `capi-repros` on 2026-09-28 because it
named a provenance class the corpus does not have — 19 rows, 2 advisories — and
because SQLite's own engine CVEs are explicitly out of its scope. `allocator-repros`
is broad rather than wrong (memcached's slabs *and* its object cache) and stays.
