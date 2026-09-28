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
`sibling_issue`, `size_class`, `size_note`, `layer_note`, `note`. Optional
means optional: the checker does not invent them, and absence is not a defect.

## Arms

Each arm declares an **oracle**, never an outcome. Outcomes live in `status`
and in `results/`. An arm that is declared but not written says so with
`"status": "not written"` instead of an oracle, so the gap is visible rather
than silently absent.

| arm | target | oracle |
|---|---|---|
| `spatial` | Capstone domain | the sequence completes |
| `sublet` | Capstone domain | fault at the labelled read probe, with `cause` |
| `poisoncap-spatial` | CheriBSD purecap, mode 0 | the sequence completes |
| `poisoncap-protected` | CheriBSD purecap, mode 1 | `SIGPROT` at the labelled read probe, with `signal` and `si_code` |
| `native-detect` | host | declared, not written: it needs Valgrind, and ASan's silence here is tautological because the memory never reached `malloc` |

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

## The corpus declaration: `corpus.json`

One per corpus, at its root. It is what makes a corpus findable, countable and
checkable from outside, and it is the only input to the generated index.

| field | meaning |
|---|---|
| `program`, `boundary`, `title` | what the corpus is about: the upstream program, the allocator or API boundary its cases cross, and one line of scope |
| `upstream` | `{version, port}` -- the release the cases are built against, and the port component that pins it. Absent where each row pins its own commit |
| `cases` | how many cases the corpus has. The checker counts the tree and refuses a mismatch |
| `case_schema` | `case-json` (a reduction with `case.c`), `sqlite-row` (a binding row carrying the provenance ledger's columns), or `xlang-row` (a shim row, declaration-level only) |
| `case_macro`, `case_glob`, `case_exclude`, `case_number_base`, `case_doc` | how cases are named and found, where the defaults do not fit. Case numbers are dense from `case_number_base` (0, or 1 for the SQLite rows) |
| `case_table`, `case_dir_column` | for `xlang-row`: the row table that is the corpus's authority, and the column naming each case directory |
| `required_arms`, `arm_keys` | the arms every case must declare, and any arm whose oracle carries more than `oracle` (the pymalloc corpus records a `cause` on `sublet`) |
| `status` | `planned`, `built`, `measured` or `triaged`. A corpus with no cases may only be `planned` or `triaged` |
| `live_in_pin_recorded` | whether the cases record liveness at all. If false, a `live_in_pin_note` must say why, and no case may carry the field |
| `expect_live_in_pin` | the expected `{true, false, not_asserted}` split. The checker recomputes it from the cases and refuses a mismatch, so this number cannot drift |
| `expect_provenance` | only where not every case has a `PROVENANCE.md`: how many do. The checker holds the corpus to exactly that count |
| `runners`, `checker`, `inventory`, `evidence`, `related` | repository-root-relative paths. Every one must exist |
| `advisories` | one entry per advisory, with its aliases in the same string, so counting entries counts advisories rather than identifiers |
| `shape_table`, `shape_prose` | whether the README carries a shape table that must partition the cases, and any prose claim about its size |
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

**`role` is why nothing was renamed.** The complete application of a program is
variously `app`, `interpreter`, `musl`, `single-user` or the port root, and those
names are load-bearing in build recipes and result bundles. The role is declared
instead, so both a reader and the index can ask what a component is without the
directory name having to answer.
