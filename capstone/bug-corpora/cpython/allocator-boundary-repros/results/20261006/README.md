# Three arms over 32 cases, 2026-10-06

| | | spatial | sublet | cheribsd-revocation |
|---|---|---:|---:|---:|
| spatial | nested | 6/12 | **9/13** | 5/13 |
| spatial | non-nested | 2/3 | **2/3** | 1/4 |
| temporal | nested | 2/10 | **9/10** | 1/10 |
| temporal | non-nested | 0/2 | **2/2** | 0/3 |
| **total** | | **10/27** | **22/28** | **7/30** |

Each cell is `detected / measured`. A case that timed out or exhausted the
application heap is **not** a measurement (SCHEMA rule 4) and is out of the
denominator rather than counted as silence.

## Why no arm's denominator is 32

| arm | excluded | case | why |
|---|---:|---|---|
| `spatial` | 5 | | |
| | | `07` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |
| | | `08` | CAPACITY: MemoryError. The arm's full run predates run-arm.sh's CAPACITY gate and recorded this as SILENT; re-reading its log corrected it |
| | | `13` | CAPACITY: MemoryError at `ast.Name(**{object(): 'y'})`, inside the one process |
| | | `21` | TIMEOUT at 600 s. The cancel path then timed out too, which is why the guest stopped answering |
| | | `32` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |
| `sublet` | 4 | | |
| | | `07` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |
| | | `08` | CAPACITY: this arm revokes on every free and reaches the heap's limit on a case the base arm completes |
| | | `21` | TIMEOUT at 600 s, as above |
| | | `32` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |
| `cheribsd-revocation` | 2 | | |
| | | `07` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |
| | | `32` | SUPERSEDED: the trigger reached a child process, and the application heap size is compiled in, so the child wanted a second heap of the same size. The trigger now runs in one process and the row is pending a re-run |

An earlier version of this file read "that sublet loses twice as many as spatial
is itself a result: revocation's cost is what pushes those cases over the limit".
That is withdrawn. Two of the four it lost, `07` and `32`, were not over any
limit: they reached a child process the domain could not give a second
application heap. Of what remains, `08` exhausts the heap on BOTH
arms, and `13` exhausts it on the base arm while sublet measures it and reports
`cause=5`. So these exclusions show revocation costing more on no case at all,
and on one case they point the other way.

## What the measurements say about the oracles

The oracles were derived from the mechanism, per cell, and written before these
runs. 33 measurements contradicted them. **The spatial arm's oracles have since
been rewritten**, and because rewriting an oracle after seeing the results is
the easiest way to launder a disagreement into an agreement, the change is set
out here in full. No verdict was changed; only the `oracle_met` column was
recomputed.

### What was wrong, and it was the oracle

Both spatial oracle texts described a bounds-only baseline: "complete: the stale
access stays inside one arena the system allocator still holds" for the nested
cases, "fault: the heap bounds each allocation" for the non-nested ones. The arm
is not bounds-only. `CAPSTONE_REVOCATION_ENFORCE` defaults to 1
(`capstone-qemu/target/riscv/op_helper.c:1420`) and is independent of the sublet
heap discipline, so the base arm enforces revocation as well.

The measurements say how much this matters. Of the arm's 11 detections, **10
carry `cause=24`**, a dereference of a revoked capability, and exactly **one**
(`12_gh-143377`) carries `cause=7`, a store bounds fault. The arm detects almost
entirely by revocation. No cell in this corpus isolates spatial safety.

### The rewrite, and why it is not an improvement

| | before | after |
|---|---|---|
| nested | completion | completion **or** `cause=24`; a `cause=5` or `7` falsifies |
| non-nested, spatial class | a fault | `cause=5`/`7`, **or** `cause=24` where the access reaches freed and revoked memory |
| non-nested, temporal class | a fault | `cause=24`; a bounds fault or completion falsifies |
| `18_gh-157335` | a fault | `cause=7` at the write past the mapping's end |

`oracle_met` on this arm went from 16/30 to 25/30. **Nine rows flipped, every one
of them from False to True, and none the other way.** That direction is exactly
what a results-fitted rewrite produces, so read the number for what it is: the
new nested oracle is *weaker* than the old one. It admits two outcomes where the
old admitted one. A weaker oracle is met more often and says less.

The honest statement is not "25 of 30 oracles now hold". It is that **the base
arm has no defect-determined outcome for the 24 nested cases at all**: whether it
faults depends on whether pymalloc has returned the block's emptied arena to the
system allocator (`insert_to_freepool`, when `nf == ao->ntotalpools && ao->nextarena
!= NULL`), which is a property of arena occupancy and not of the defect. The arm
is uninformative for them, and the corrected oracle now says so instead of
predicting a completion it cannot guarantee.

### What the arm genuinely misses

Five rows remain unmet, and these are the informative ones -- all non-nested,
all system-allocator blocks the arm had an event to act on:

- `07_gh-140594`, `08_gh-140607` -- silent. 07's measurement is void anyway: its
  trigger reached a child that never got a heap, and it has since been rebuilt to
  run in one process.
- `18_gh-157335` -- silent on all three arms. Its buffer is an mmap mapping, not
  an allocator block, so no allocator-level mechanism has an event.
- `19_gh-142664`, `20_gh-143308` -- temporal, non-nested, silent. A
  use-after-free on a system-allocator block is what revocation exists to catch,
  and this arm did not catch it.

## The finding that was worth following, resolved

`../pymalloc-repros` holds 11 of these defects as C model consumers, and on its
`spatial` arm all 20 of its cases complete -- 0 faults, which is that corpus's
declared oracle, not a failure. The same defects reached through the
**interpreter** fault on the same arm in 3 of 11 cases.

Those 3 all carry `cause=24`. That resolves it. A C model frees its object back
to its own pool and so never produces an allocator-level free, which is the only
thing a revoked-capability fault can come from. The interpreter runs a real
object lifetime and does reach the path where pymalloc returns an emptied arena
to the system allocator. The two corpora's `spatial` arms are therefore not
measuring the same thing, and their numbers still must not be pooled -- but the
reason is now known rather than open.

## How much each arm discriminates, which is not what `oracle_met` measures

Both the spatial and the cheribsd oracles have now been corrected, and both
corrections moved `oracle_met` up and only up: spatial 16/30 -> 24/27,
cheribsd 18/30 -> 29/30, with 9 and 11 rows flipping from False to True and
none the other way. Read on its own that looks like the corpus improving. It is
not, and the number that says what actually happened is this one:

| arm | measured | `oracle_met` | cases the classification PREDICTS | of those, met |
|---|---:|---:|---:|---:|
| `spatial` | 27 | 24/27 | **5/27** | 2 |
| `sublet` | 28 | 22/28 | **28/28** | 22 |
| `cheribsd-revocation` | 30 | 29/30 | **1/30** | 0 |

The corrected oracles stopped predicting outcomes that the mechanism does not
determine, so they are met more often and say less. On the base arm, whether a
nested defect faults depends on whether pymalloc has returned the block's
emptied arena to the system allocator (`insert_to_freepool`, when `nf ==
ao->ntotalpools && ao->nextarena != NULL`). On CheriBSD it depends on that and
on sweep timing as well, because the run records
`security.cheri.runtime_revocation_every_free_default` as 0 and
`runtime_revocation_async` as 1 -- revocation there is sweep-based and
asynchronous, so a stale access reached before the next sweep still holds a
tagged capability.

Those outcomes are reproducible: the triggers are deterministic. They are simply
not derivable from the case's class and side, which is the whole claim a
four-cell table makes. **`sublet` is the only arm whose outcome follows from the
defect's classification for every case it measured**, and it is the only arm
whose misses are therefore informative: six of them, `06`, `11`, `15`, `17`,
`18` and `23`, where the oracle names a fault and the arm was silent.

The one case the cheribsd oracle does predict is `18_gh-157335`, and it is not
met. Its buffer is an mmap mapping, so a bounds fault was the prediction and the
arm was silent -- the same outcome it has on the other two arms.

## Provenance

`inputs.json` carries every run directory and its `run.meta`: the arm, the image
sha256 for the Capstone arms, `CAPSTONE_REV_NODES`, and for CheriBSD the whole
`security.cheri` subtree read inside the guest before any case ran and again
afterwards. Both Capstone arms refuse to boot an image whose hash is not the one
recorded for the arm they were asked for; the CheriBSD arm refuses to start if
`runtime_revocation_default` is not 1.

Two controls ran before every batch: `objects.py` (the interpreter is qualified
on this image at all) and, on CheriBSD, `mech-control S_OOB` (the si_code
handler reports — without it a fault is only "it crashed").
