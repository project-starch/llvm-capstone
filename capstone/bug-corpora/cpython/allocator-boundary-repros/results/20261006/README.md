# Three arms over 32 cases, 2026-10-06

| | | spatial | sublet | cheribsd-revocation |
|---|---|---:|---:|---:|
| spatial | nested | 6/12 | **9/13** | 5/13 |
| spatial | non-nested | 2/5 | 2/3 | 2/5 |
| temporal | nested | 3/11 | **9/10** | 2/11 |
| temporal | non-nested | 0/2 | **2/2** | 0/3 |
| **total** | | **11/30** | **22/28** | **9/32** |

Each cell is `detected / measured`. A case that timed out or exhausted the
application heap is **not** a measurement (SCHEMA rule 4) and is out of the
denominator rather than counted as silence.

## Why the two Capstone denominators are not 32

| arm | excluded | case | why |
|---|---:|---|---|
| spatial | 2 | `13_gh-144169` | CAPACITY: the domain's application heap could not satisfy the allocation, so the defect site was never reached |
| | | `21_gh-146169` | TIMEOUT at 600 s: expat re-entrant parsing does not finish |
| sublet | 4 | `08_gh-140607` | CAPACITY: this arm revokes on every free, so it reaches the heap's limit on a case the base arm completes. The capacity constants are compile-time and this image does not carry the raised ones |
| | | `07_gh-140594`, `32_gh-149449` | NOT capacity, though they reported it: both reached a **child process**, and the application heap size is compiled in, so the child wanted a second heap of the same size. Both triggers now run in one process and are pending a re-run |
| | | `21_gh-146169` | TIMEOUT at 600 s, as above |
| cheribsd-revocation | 0 | — | runs on a real OS: no domain heap ceiling, nothing timed out |

That sublet loses twice as many as spatial is itself a result: revocation's cost
is what pushes those cases over the limit.

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
