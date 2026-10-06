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
| sublet | 4 | `07_gh-140594`, `08_gh-140607`, `32_gh-149449` | CAPACITY: this arm revokes on every free, so it reaches the heap's limit on cases the base arm completes. The capacity constants are compile-time and this image does not carry the raised ones |
| | | `21_gh-146169` | TIMEOUT at 600 s, as above |
| cheribsd-revocation | 0 | — | runs on a real OS: no domain heap ceiling, nothing timed out |

That sublet loses twice as many as spatial is itself a result: revocation's cost
is what pushes those cases over the limit.

## What the measurements say about the oracles

The oracles are derived from the mechanism, per cell, and were written before
these runs. **33 measurements contradict them** and none of the oracles was
rewritten to match. The contradictions fall into two groups.

**Nested cases that WERE caught by an arm that should not see them (14).** On
both the base Capstone arm and CheriBSD, several nested defects fault. For
Capstone the reason is identified: `CAPSTONE_REVOCATION_ENFORCE` defaults to 1
and is independent of the sublet discipline, so the base arm revokes too — every
one of those detections carries `cause=24`, a dereference of a revoked
capability, not a bounds fault. The arm is therefore **not** a bounds-only
baseline, and any statement of the form "sublet catches what the baseline cannot"
has to be read against that.

**Non-nested cases that were NOT caught (19).** `18_gh-157335` is missed by all
three; its buffer is an mmap mapping rather than an allocator block, so no
allocator-level mechanism has an event. The three temporal non-nested cases are
missed by spatial and CheriBSD, which is consistent: neither revokes anything
pymalloc does, and a use-after-free needs a revocation to catch.

## The finding worth following

`../pymalloc-repros` holds 11 of these defects as C model consumers, and on its
`spatial` arm all 20 of its cases complete — 0 faults, which is that corpus's
expected result. The same defects reached through the **interpreter** fault on
the same arm in 3 of 11 cases. Same defects, same arm, two ways of reaching
them, different outcomes. The likely reason is that the C models perform a
malloc/free/read sequence directly against pymalloc while the interpreter runs
a real object lifetime, so the two do not exercise the same allocation path.
This is not resolved here and the two corpora's numbers should not be pooled
until it is.

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
