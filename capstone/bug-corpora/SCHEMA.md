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
`allocator_consumed`, `channel`, `nested`, `nested_why`, `citation_constraint`. Optional means
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
| `poisoncap-spatial` | CheriBSD purecap, mode 0 | the sequence completes |
| `poisoncap-protected` | CheriBSD purecap, mode 1 | `SIGPROT` at the labelled read probe, with `signal` and `si_code` |
| `native-detect` | host | declared, not written: it needs Valgrind, and ASan's silence here is tautological because the memory never reached `malloc` |
| `sysalloc-none` | Capstone domain | the whole-program run completes; the first-fit heap is built with `-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`, so only tag integrity and the arena's bounds are left |
| `sysalloc-bounds` | Capstone domain | the same, with the heap as applications get it since PR #170: each allocation bounded, no revocation. **The baseline a catch is measured against** |
| `sysalloc-sublet` | Capstone domain | the Sublet heap: per-object bounds and a revoke on every free |
| `sublet-gc` | Capstone domain | `sysalloc-sublet` plus the program's own Sublet port, so its nested allocator's objects are issued and revoked too |

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

## The three protection arms, and `protection` in `corpus.json`

The arms above are **mechanism** arms: each one names an oracle for one thing a
vehicle can do, and every corpus grew the ones its own measurements needed.
Sixteen of them now exist, in seventeen result-table shapes, and that is the
right record of how a number was produced — but it is not an answer to the
question the corpora are for: *for this bug, which defence reports?*

[`arms.json`](arms.json) defines the three arms that question is asked of:

| arm | what is protected | vehicle |
|---|---|---|
| `cheribsd` | the libc malloc/free boundary, by stock CheriBSD with revocation at its shipped default: asynchronous, quarantine 1/4 | a purecap process |
| `capstone-sysalloc` | the same boundary, by Capstone: bounded on issue, revoked at `free` with no quarantine | a Capstone process, or a freestanding domain where a corpus has no process run |
| `capstone-sublet` | the above, plus the program's **nested** allocator issuing and revoking its own objects | the same |

PoisonCap is not among them and will not be: it is our adapter for a
competitor's platform, so a reader is entitled to discount it. The
`poisoncap-*` arms stay in the cases as a record of what was measured and are
not read into this table.

Every corpus declares, in `corpus.json`, a **`protection`** block with one entry
per arm, and each entry is one of three things:

    "protection": {
      "cheribsd":          {"bundle": "results/20261006-cheribsd",
                            "reader": "supervise-blocks", "key": "number",
                            "vehicle": "cheribsd-purecap"},
      "capstone-sysalloc": {"bundle": "results/20261007-virtual", "reader": "virtual",
                            "arm": "virtual", "key": "dir",
                            "vehicle": "capstone-virtual"},
      "capstone-sublet":   {"coincides_with": "capstone-sysalloc", "why": "..."}
    }

* a **`bundle`** (or a list of them) with the `reader` that understands its
  shape, the `key` naming which of the case's identifiers that bundle uses, and
  the `vehicle` it ran on. The verdict is then read out of the bundle, never
  typed into the declaration.
* **`coincides_with`** plus a `why`, for an arm that is the same binary and the
  same measurement as another — a corpus whose objects come straight from
  `malloc` has no nested allocator for `capstone-sublet` to protect.
* **`not_run`** with a reason of at least a few words. "Cannot be measured
  here", "waiting on a port" and "nobody got to it" weigh differently in a
  denominator, and a bare blank hides which one it is. `tools/check-corpus.py`
  refuses a one-word reason.

### A detection is not a finished reading either: `detection_unproven`

The mirror of `silence_unproven`, and it was needed on 2026-10-08 for the same
reason: a fault is only a detection of THIS defect if the same image is quiet
without it. Where a control run shows the fault firing on the upstream-FIXED
sequence too, the entry says so and the cell becomes `not-run`, with the control
named:

    "capstone-sysalloc": [{"bundle": "...", "reader": "virtual", "key": "dir",
                           "vehicle": "capstone-virtual", "group": "bucket-repros",
                           "detection_unproven": {
                             "bundle": "results/20261008-fixed-arm-control",
                             "cases": [0, 1, 4, 6],
                             "why": "the fault lands at apr_buckets_alloc.c:113 ..."}}]

`bundle` must be a filed bundle, `why` twelve words at least, `cases` optional.

How eight cells came to need it is the lesson, not the mechanism. The runs had
reported `attributed=false` on every one of them -- the runner resolves the case's
labelled probe in the image that ran and says whether the faulting pc lies inside
it -- and nobody read that column. Rule 2 of the contract is exactly this: a fault
is accepted at the labelled probe and nowhere else. **Read the attribution before
the verdict**, the same way `controls N / M` has to be read before a verdict
column. Where a corpus has no fixed arm on a vehicle, there is nothing to control
a detection against, which is why the fixture's spare event word now selects the
arm (`shared/corpus.h` in both affected corpora).

### A miss is not a finished reading: `disposition`

A `caught` cell is a statement; a `missed` cell is two different things wearing
one symbol. Either the arm's mechanism got a chance and said nothing, or it never
got a chance at all -- and for `cheribsd` that difference is the whole claim,
because its revocation is asynchronous: an object may be freed, quarantined, and
read through before any sweep clears it. Left undistinguished, a column of dots
reads as "the mechanism failed" when some of those dots mean "the object never
reached the mechanism".

So a bundle that has measured the reason declares it beside the verdict:

    "cheribsd": [{"bundle": "results/20261006-cheribsd", "reader": "case-lines",
                  "key": "number", "vehicle": "cheribsd-purecap",
                  "group": "allocator-repros",
                  "disposition": {"bundle": "results/20261008-cheribsd-quarantine",
                                  "finding": "never-freed",
                                  "cases": [0, 1, 2, 3, 4, 5, 6, 7],
                                  "why": "34 of 34 frees quarantined, none reused "
                                         "while quarantined, and the reuse is on "
                                         "memcached's own freelists"}}]

* **`bundle`** is the run that measured it, and it must exist. The disposition
  is evidence from a run, not an opinion typed into the declaration.
* **`finding`** is one of `never-freed` (the program's own allocator recycles the
  object and the system allocator never sees it), `quarantined-unswept` (freed,
  quarantined, and handed back out before a sweep) or `not-temporal` (nothing was
  freed; the defect crosses a bound inside one live allocation).
* **`why`** carries the numbers that settle it, twelve words at least.
* **`cases`** is optional and narrows the disposition to those case numbers;
  without it, it covers every missed case of the group.

One group's silences can have more than one cause, so the field also takes a
**list**. mruby is the example: four of its cases are `quarantined-unswept` and
one is `not-temporal`, from two different runs. With a list, every entry must name
its `cases` -- one that named none would claim every missed cell -- and the
checker refuses two entries that claim the same case, or a list of one.

A disposition only attaches to a cell whose verdict is `missed`: there is nothing
to dispose of otherwise. What it then does depends on the finding, and the split
is the point of the field:

* **`quarantined-unswept`** turns the cell into a **CATCH**, printed
  <code>C<sup>q</sup></code> and counted both in `caught` and in a column of its
  own. IN THE QUARANTINE COUNTS AS CAUGHT is the user's rule, 2026-10-08, and it is
  deliberately generous to the arm: the mechanism received the object and held it,
  and only the batching of its own sweep let the stale access through. The separate
  column exists because the evidence is membership in a shadow bitmap rather than a
  reported fault, and two kinds of proof should not disappear into one number.
* **`never-freed`** and **`not-temporal`** leave the cell a **miss**. There the
  mechanism never had the object -- the program's own allocator recycled it, or
  nothing was freed at all -- and no sweep policy would change that.

`--focus disposed@cheribsd` lists the explained silences of either kind and
`--focus undisposed@cheribsd` the ones still open.

### The axis every comparison is split on: `allocator_boundary`

Who allocated the object a case crosses decides which arm can possibly report, so
every group declares it and `arms.json` defines the values: `system` (the system
allocator handed the object out directly -- all three arms protect this),
`nested` (the program's own allocator carved it from a block it holds -- only
`capstone-sublet` protects it), `interior` (the crossing is inside ONE allocation
and no allocator vested the crossed region -- no arm in this study claims it), or
`mixed`. A `mixed` group obliges **every** case in it to carry its own
`allocator_boundary`, and the checker refuses a case that carries one where its
group already answered. The `xlang-row` corpora are exempt, as they are from the
`protection` block.

PROTECTION.md also carries a ledger built from the dispositions: how many cells
were `quarantined-unswept` (credited as catches), how many `never-freed` (no sweep
policy reaches them; only protecting the nested allocator does), how many
`not-temporal`, and how many are still open. The arm itself is unchanged --
`cheribsd` stays revocation on, asynchronous and batched, the way the platform
ships it. What the rule changes is how its own quarantine is scored, not how it
runs.

PROTECTION.md's next section uses the boundary axis: the arms compared only on the cases where
all three were measured, split by boundary class. Without the split, a tree whose
corpora are mostly nested-allocator bugs makes the malloc-boundary arms look weak
at the boundary they do protect, and an unrun cell would count as a failure.

The instrument for `cheribsd` is
`capstone/ports/common/host/cheribsd/quarantine-probe.c`: `LD_PRELOAD`ed through
a case's `env`, it wraps `malloc` and `free` and reads the kernel's revocation
shadow bitmap, one bit per 16-byte granule, set while that granule is
quarantined. Reading a bit forces no sweep and changes nothing else about the
run. Forcing a sweep with `_RUNTIME_REVOCATION_EVERY_FREE_ENABLE=1` is **not** a
substitute: it makes cases fault that no mechanism may legitimately catch, so it
cannot tell a catch from an artefact, and verdicts taken from it were withdrawn.

`tools/protection-matrix.py` reads the blocks and writes
[PROTECTION.md](PROTECTION.md) and `protection.json`: one row per bug, three
cells, each carrying the bundle it came from. Both files are generated; the
checker holds every bundle path, reader name and arm name to what exists, and
`--check` fails if the generated files are stale.

The `xlang-row` corpora are exempt for now. Their cases are not case
directories and their arms live in their own `rows.tsv`, so folding their 30
rows in needs a reader per table, not a declaration.

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

## The corpus declaration: `corpus.json`

**One per program**, at `bug-corpora/<program>/`, with the cases in **groups**.

It was one per boundary -- nineteen declarations over nine programs -- and that
is one unit too many to hold in the head: nineteen protection blocks, nineteen
index rows, nineteen READMEs, for nine programs. A group is now what a corpus
was, and the program is the corpus.

What stays with the group is what is true of it alone: the **boundary** its
cases cross, the **upstream pin** (FFmpeg's four groups share one, PostgreSQL's
three do not), the build seam's **case macro** and **case schema**, its own
runners, inventory and result bundles. What belongs to the program moves up: the
title, the total, and the protection sources. Nothing moved on disk, so no
archived bundle's case names became wrong.

    {
      "program": "ffmpeg", "title": "...", "cases": 19,
      "checker": "capstone/bug-corpora/tools/check-corpus.py",
      "groups": {
        "plain-heap-repros": {"boundary": "one av_malloc_array, no inner layer", ...},
        "pool-repros":       {"boundary": "AVBufferPool / AVRefStructPool", ...},
        ...
      },
      "protection": {"cheribsd": [{"group": "plain-heap-repros", "bundle": ...}, ...], ...}
    }

Each arm holds **one entry per group**, because the groups of one program were
measured by different runs -- FFmpeg's pool bundle is not its plain-heap bundle
-- and a cell must name the run that scored its own case. A group that runs on
no arm here carries `ignore` instead and owes no entries.

The fields below are a group's, except `program`, `title`, `cases`, `checker`,
`groups` and `protection`, which are the program's. A declaration with **no**
`groups` is read as one group spelled inline, which is the shape `xlang/` uses.

| field | meaning |
|---|---|
| `program`, `boundary`, `title` | the upstream program, the allocator or API boundary the group's cases cross, and one line of scope |
| `groups` | the program's groups, each a map of the fields below |
| `protection` | per arm, one entry per group, saying where that verdict comes from -- see above |
| `ignore` | on a group: it runs on no arm here, with the reason. See **Ignored cases** |
| `upstream` | `{version, port}` -- the release the cases are built against, and the port component that pins it. Absent where each row pins its own commit |
| `cases` | how many cases the corpus has. The checker counts the tree and refuses a mismatch |
| `case_schema` | `case-json` (a reduction with `case.c`), `script-trigger` (a defect whose trigger is an interpreter script rather than a C reduction; each case names its own file in `trigger`), `sqlite-row` (a binding row carrying the provenance ledger's columns), or `xlang-row` (a shim row, declaration-level only) |
| `case_macro`, `case_glob`, `case_exclude`, `case_number_base`, `case_doc` | how cases are named and found, where the defaults do not fit. Case numbers are dense from `case_number_base` (0, or 1 for the SQLite rows) |
| `case_table`, `case_dir_column` | for `xlang-row`: the row table that is the corpus's authority, and the column naming each case directory |
| `required_arms`, `arm_keys` | the arms every case must declare, and any arm whose oracle carries more than `oracle` (the pymalloc corpus records a `cause` on `sublet`). **Order is not significant and is not checked** — `wmem-repros` lists `cheribsd-revocation` third, to mirror the paper's column order, while `allocator-repros` and `pool-repros` list it after the two PoisonCap arms. Both are valid; noted here because the difference otherwise reads as a defect, and reordering a corpus to match another would churn every case file for nothing |
| `status` | `planned`, `built`, `measured` or `triaged`. A corpus with no cases may only be `planned` or `triaged` |
| `live_in_pin_recorded` | whether the cases record liveness at all. If false, a `live_in_pin_note` must say why, and no case may carry the field |
| `expect_live_in_pin` | the expected `{true, false, not_asserted}` split. The checker recomputes it from the cases and refuses a mismatch, so this number cannot drift |
| `expect_provenance` | only where not every case has a `PROVENANCE.md`: how many do. The checker holds the corpus to exactly that count |
| `runners`, `checker`, `inventory`, `evidence`, `related` | repository-root-relative paths. Every one must exist |
| `advisories` | one entry per advisory, with its aliases in the same string, so counting entries counts advisories rather than identifiers |
| `shape_table`, `shape_prose` | whether the README carries a shape table that must partition the cases, and any prose claim about its size |
| `note` | anything a reader needs that no field above holds |

## Ignored cases

A case that stays in the corpus and leaves every denominator says so in an
`ignore` object, and says which of three kinds it is. The kinds are separated
because a bare "ignored" is the one cell a reader cannot interpret, and each of
these carries different weight:

| `kind` | what it means |
|---|---|
| `no-runtime` | no arm here can run it at all. Declared on a **group** where it holds for all of it. **Unused:** the one group that had it, SQLite's 19 binding rows, was deleted on 2026-10-08 rather than kept as unrunnable material |
| `arm` | one or more **named** arms cannot run it; the others measure it normally. `arms` lists them. **Unused:** the one case that had it, SQLite's heap-configuring OOM case, was deleted on 2026-10-08 |
| `out-of-scope` | neither spatial nor temporal, so outside what the study measures. **Unused:** mruby's four null dereferences, one type confusion and one C-stack exhaustion were deleted on 2026-10-08 rather than kept |
| `parked` | measured or measurable, and **deliberately not reported**. The first three kinds are properties of the case; this one is a decision about what the study says, so its `why` must name on whose decision and what the omission costs |

Every kind needs a `why` of more than a few words -- twelve for `parked` -- and
the checker refuses a label.

**To LOOK at a subset, do not park anything.** `tools/protection-matrix.py
--focus <spec>` prints a selection to stdout and writes no file, so
PROTECTION.md stays the whole record:

    tools/protection-matrix.py --focus ffmpeg --focus caught
    tools/protection-matrix.py --focus caught@capstone-sublet --focus missed@cheribsd
    tools/protection-matrix.py --focus undisposed@cheribsd

A verdict spec widens within its kind and narrows across kinds, so the first
line is "FFmpeg's caught ones" and the second is "what only we catch". `parked`
is itself a spec, so a parked case can always be listed again. `disposed` and
`undisposed` select misses by whether a run has said why the arm was silent, so
the third line is "the CheriBSD misses nobody has explained yet".

The difference is deliberate. Parking changes the denominator every reader sees;
focusing changes what one reader is looking at. Collapsing the two would make a
table that reads as the whole truth while being somebody's current interest.
An ignored case is `∅` in PROTECTION.md and is subtracted from that arm's
denominator -- the totals there are stated with the ignored count beside them,
so a reader can see what was left out rather than infer it from a gap.

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
| `targets` | where it runs: `capstone-domain` (a freestanding domain), `capstone-virtual` (a Capstone application in a Linux process, user virtual addresses), `cheribsd-purecap`, `linux-guest`, `silicon`, `native` |
| `workload`, `evidence`, `corpora`, `related`, `status`, `note` | the qualifying workload, result bundles, the corpora whose cases are its own, neighbouring components, and anything else a reader needs. Paths must exist |
| `arms` | optional: one note per arm this component builds, where the arm's construction is a property of the component rather than of a corpus |

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
`apr-pool-repros`, `bucket-repros`, `allocator-repros`. The name says what a case
crosses, never where the defect was reported: `sqlite/cve-repros` was renamed to
`capi-repros` on 2026-09-28 because it named a provenance class the corpus does not
have — 19 rows, 2 advisories — and because SQLite's own engine CVEs are explicitly
out of its scope; that group and `mruby/gc-slot-repros` were both deleted later, the
first on the user's order and the second because it declared a boundary with no
cases behind it. `allocator-repros`
is broad rather than wrong (memcached's slabs *and* its object cache) and stays.
