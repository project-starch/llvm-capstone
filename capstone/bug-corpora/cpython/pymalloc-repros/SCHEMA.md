# The corpus contract

What a case in this corpus *is*, field by field, so the shape can be copied
rather than re-derived. `tests/check-corpus.py` enforces every rule below; if
the two ever disagree, the checker is the authority and this file is stale.

## One directory per case

    <upstream-fix-id>_<slug>/
        case.json        machine-readable claims
        PROVENANCE.md    the upstream hunk, quoted; what is real and what is reduced

The directory name begins with the same id as `upstream_fix`. Case numbers are
dense: N cases are numbered 0..N-1, with no gaps and no duplicates, because the
number is what a run selects.

**There is no `run.sh` per case here**, and that is deliberate. A capability
fault ends the domain, so a case that provokes one cannot also report results
beside it: one case per run, selected by number, from one shared program. The
executable material is therefore `shared/` and the per-target directories, not
the case directory. A corpus whose cases *can* run side by side may prefer a
per-case script; this one cannot.

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
