# Results, 2026-10-05 -- every case run on every arm

Supersedes `../20261004`, which this re-states with the gaps closed. Raw serial
and console logs are not committed (SCHEMA rule 6); `matrix.tsv` has one row per
case per arm.

| arm | detected | not detected | hang | other | run | of |
|---|---:|---:|---:|---:|---:|---:|
| `spatial` | 10 | 30 | 3 | 0 | 43 | 43 |
| `sublet` | **36** | 5 | 1 | 1 | 43 | 43 |
| `cheribsd-revocation` | 4 | **31** | 0 | 8 | 43 | 43 |

**`not run` is 0 on all three arms**, down from 15 on 2026-10-04.

## What closed the gaps

**CheriBSD, 12 cases.** Six had been removed from the build manifest on the
argument that "no input reaches a memory error, so they are not bugs of this
corpus". That argument overstated the evidence: each has a real upstream fix
commit and a `live_proof` that is a line-numbered inspection of the 3.22.0
tree, and each already ran on both Capstone arms from the same sources. They
were restored. Five R2 fuzz-diff cases were in the manifest but had never been
built here -- the earlier note that their binaries were cross-compiled was
about the Capstone arms, not this one. `blobclose` was built all along and
simply never run.

**Sublet, 3 cases.** `fz08` and `fz09` had been left `NORUN` when the group
exhausted its boots: 25 of this arm's faults halt the VM, each boot retires one
case, and the run stopped at boot 7 with `INFRA: boot kept flaking`. Re-run on
their own they finished in two boots. `blobclose` needed its source ported from
the legacy standalone domain onto `REPRO322_MAIN` first -- the old entry point
had its own `domain_main` and never captured the Sublet grant, so `memsys5Init`
faulted inside `capstone_cap_base` before the case could run, and the build
skipped the tag on that arm. The defect sequence is unchanged.

## Three verdicts the raw logs corrected

`fz06` on both arms and `fz11` on `spatial` were recorded as having returned
`rc=0` **without** a completion marker. All three printed one -- `fz06 NOTRAP
done`, `==RC fz06_r2=0==`. `corpus322.sh` had filed them INFRA because the
*boot* returned 1, the next case in the queue having hung, and the bookkeeping
charged that to the head of the remaining list. **A verdict derived from the
batch's exit status was contradicted by the output of the case it described.**

Their logs also carry the reason nothing was detected, which no verdict word
did: `fz06 s0_rc=11` is `SQLITE_CORRUPT` -- the trigger's database image is
rejected before the defect -- and `fz11 s0_rc=7` is `SQLITE_NOMEM`, the 256 KB
memsys5 arena exhausted first. Neither is a statement about a mechanism.

## What the `other` column holds

`sublet` 1: `fz08` left the domain at the defect without faulting, its log
ending at `about to step (sqlite3VdbeCursorMoveto runs here)` then `unexpected
domain return`, rc=1. Neither a detection nor a clean run.

`cheribsd-revocation` 8: three the probe proves never reached the defect
(`hits=0`), and five the harness could not score at all (`no verdict: ...`),
where the case's own script stopped partway.

## A limit that applies to every `not detected` cell on the Capstone arms

Neither Capstone arm has a defect-site probe, so "ran to completion and
reported nothing" cannot be separated there from "the input never reached the
defect". A marker cannot survive on that target: the domain's `out_text()`
writes a shared region the host reads only after the domain returns, so a
faulting domain prints nothing. On `cheribsd-revocation` the distinction does
exist -- 20 of its 31 carry probe evidence that the defective code executed --
which is why its zeros are measurements and the Capstone columns' are weaker.

Nine of CheriBSD's 31 are the cases restored on 2026-10-05, which have no probe
site derived for them. They are counted as not detected because they ran and
nothing was reported; they are **not** evidence that the defect site was
reached.

## One diagnostic, deliberately not folded in

`26_2c7a73eaea` reports `hits=0` here: at the configured 256 KB arena it
exhausts the heap before reaching `fts3SegWriterAdd`. Rebuilt with an 8 MiB
arena it reaches the defect 1429 times and CheriBSD faults with
`PROT_CHERI_BOUNDS`. That is a different configuration and is **not** in the
table above: the same rebuild on either Capstone arm fails with `create_dom
failed`, because the memsys5 heap is a static array compiled into the domain
image and 8 MiB exceeds the domain's budget. The three arms cannot be held at
that arena size, so the row stays at the configured one.
