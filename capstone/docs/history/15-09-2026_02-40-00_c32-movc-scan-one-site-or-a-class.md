# C-32 is a CLASS, not one site — and the obvious scan shape flags its own control

**Date:** 2026-09-15. Asked by the board lane after C-32 was observed live on silicon in the Sublet
port's `setupLookaside` at `-O2` (image `c506694f9f6f6889`): is this one site or a class? Instrument:
`capstone/tests/c32-movc-scan.py`, committed with this note.

## The shape, and a correction to the one proposed

The proposed scan was "`movc rd, rs` with `rs` read again before redefinition". That over-reports,
and it can be shown to: run it on the committed reproducer
`llvm/test/CodeGen/Capstone/c32-movc-untagged-live.ll` and it flags BOTH functions — including
`real_cap_copy`, which that file carries precisely as the CONTROL that a genuine capability copy must
remain a `movc`. Both functions have a `movc` whose source is read again; they differ in how the
source was **defined**:

    bridged_copied_live:  mv   s0, a0   -> integer-defined, UNTAGGED -> MOVC nulls it   (the defect)
    real_cap_copy:        movc s0, a0   -> capability-defined, NONLIN -> survives       (the control)

So the discriminator is the DEF side. The scanner classifies the last definer of `rs` as integer-
producing, capability-producing, or unknown, and counts only the integer-defined ones.

**Control, run every time:** on that reproducer the scanner reports 2 hits for the loose shape and
exactly **1** C-32 candidate — `bridged_copied_live` at `0x1c`, with `real_cap_copy` correctly
classified `cap-def`. The instrument is therefore shown both to FIRE and to SEPARATE the two
hypotheses, which the loose shape does not.

## Result: a class, and opt-level gated

Plain memsys5 SQLite silicon images (the B6 set, `~/capstone-artifacts/b6-2026-09-14/`) — a DIFFERENT
workload from the Sublet port where the defect was seen:

| image | movc | loose shape | INT-DEF | of which `li rX,0` (benign) | **non-zero bridged** |
|---|---:|---:|---:|---:|---:|
| -O0 `6cf8edf637f72063` | 6,755 | 2,898 | **0** | 0 | **0** |
| -O1 `50ca86aa70b3e425` | 17,442 | 10,367 | 199 | 25 | **174** |
| -O2 `ec061577fb008e18` | 17,721 | 10,637 | 227 | 34 | **193** |

The `-O0` column is an internal control: same source, same toolchain, and the shape vanishes —
independently reproducing the board lane's silicon observation that `-O0` does not diverge because its
cast goes through memory. A `-O0` speedtest1 image from `speedtest1-size100` also gives 0 with 7,321
`movc`, so the zero is not an artefact of one build.

Verified against the unfiltered disassembly rather than trusted from the filter, e.g. in
`sqlite3_str_vappendf`:

    1458c: li   s5, 0x0
    145cc: movc a0, s5        <- nulls s5 on RTL
    145d0: jalr a1
    145e4: movc a0, s5        <- second read; cnull on silicon

and a non-zero instance in `sqlite3_step` at `0x178f0`: `movc a0, s3` with `s3` defined `li s3, 0x7`
and read again at `0x17930`.

**`li rX, 0` is separated out because nulling a register that already holds zero changes nothing.**
Counting those as defects would have overstated the result by ~15 %; the harmful figure is the
non-zero column.

## What these numbers are NOT

* **Candidates, not defects.** The scan is linear and does not follow control flow, so a branch
  between the `movc` and the read is unmodelled; and it cannot prove a source is untagged at runtime —
  the def classification is a static proxy, good but not a proof.
* **Reachability is unknown.** Nothing here says these sites execute, or that a consumer would notice.
  The Sublet `setupLookaside` instance is the only one demonstrated to change a silicon result.
* A **miss is not a proof of absence**, for the same reasons.

The honest statement is: the shape the silicon instance exhibits is present ~190 times in one
unrelated `-O2` production image and zero times in its `-O0` twin, so **C-32 is a class whose
exposure is opt-level gated**, and the count is a scale, not a defect list.

## Not prototyped, deliberately

The bridge pseudo was NOT prototyped. The registry says both designs "touch the ABI of integer-bridged
pointers, which is why neither is a lane's call", and the lead's standing ruling on C-32 is to
document and discuss rather than build. A peer lane's request does not change that; the scan is
offered instead because it informs the choice without pre-empting it. The relevant new input for the
decision is that the residue argument now has a number against it: a fix that leaves PHI copies behind
is leaving part of a ~190-site class in place, and that is worth knowing before choosing.
