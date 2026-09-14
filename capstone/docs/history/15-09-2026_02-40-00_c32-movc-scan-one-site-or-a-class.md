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

---

## RETRACTED 2026-09-15, same day: "~190 sites, a class" — the harm-shaped count is a HANDFUL

The headline number above is wrong and I withdraw it. The board lane rebuilt the scan over a real
control-flow graph (`capstone/tests/movc-cfg-scan.py`, `f2ae79db7ad5` on dev). Run on the **same three
B6 images** this note reports on:

| image | movc | source re-read on some path | INT-ONLY | mixed |
|---|---:|---:|---:|---:|
| -O0 | 6,755 | 2,978 | 0 | 0 |
| -O1 | 17,442 | 10,350 | **0** | 2 |
| -O2 | 17,721 | 10,371 | **1** | 1 |

So the harm-shaped class is a handful per image, not ~190. **The -O0 = 0 result and the "this is
opt-level gated" conclusion both survive; the SCALE does not.**

**Why my count was high, precisely.** Not the re-read condition — that was applied (measured: 717
integer-defined `movc` in the -O2 image, of which my scan reported 227, excluding 490 dead-source
copies). The error is that my forward and backward searches walk the disassembly in **linear address
order**, which is not the control-flow graph. Linearly, the instruction after a `movc` is frequently on
a different path, so a "read" that never executes after the copy is counted, and a redefinition on an
unrelated path terminates the search early. Both directions are wrong, and the net effect here was a
two-orders-of-magnitude over-count. A CFG is not a refinement of a linear scan; it is the only correct
instrument for a reaching-definitions question, and the note above should have said so rather than
listing "does not follow control flow" as a caveat and then quoting a number as if the caveat were
small.

**The lesson, which is the reusable part:** I did label the linear scan's limitation correctly and
still published the count as the headline. A stated caveat that is not allowed to change the claim is
decoration. Where the caveat names the exact mechanism that could invalidate the number, the number
does not go out until it is measured with the caveat removed.

`capstone/tests/c32-movc-scan.py` is kept, because its DEF-side classification is what separates the
defect from `real_cap_copy` and that part held up — the CFG tool adopts the same discriminator, and
both agree on the reproducer (1 hit at `bridged_copied_live+0x1c`, control excluded). Its header now
has to be read with this retraction: **use `movc-cfg-scan.py` for counts.**

### Two corrections back to the CFG tool

1. **Its summary line over-reports INT-ONLY.** `strong` accumulates INT-ONLY *and* MIXED entries, but
   the printed line labels `len(strong)` as *"have ONLY integer reaching definitions"*. True INT-ONLY
   is `len(strong) - mixed`. My -O1 image prints "2 have ONLY integer" and is really 0 INT-ONLY + 2
   mixed. The per-site tags are right; only the summary conflates, so any count quoted from that line
   (including the board images') needs the subtraction.
2. **It takes `<path> <label>` PAIRS** (`argv[1::2]`/`argv[2::2]`). Invoked with a path alone it prints
   NOTHING and exits 0, which reads exactly like a clean image.

### The PHI question, answered on the sites rather than as a rate

With only two -O2 sites, a "PHI share" statistic would be noise. Characterised individually:

* `renameResolveTrigger+0x111ce0` — `movc s8, s11` at a **block entry** (preceded by an unconditional
  `j`), source live around a back-edge; the read the tool finds is the copy itself on the next
  iteration. **PHI-shaped: remat cannot remove it.**
* `main+0x3ab18` — `movc a1, s5` before a call, `mv a1, s5` before a later call, straight-line.
  **Not PHI-shaped.**

One of two. The residue the bridge pseudo cannot reach is real and present at this scale, which is the
input the design choice needs — but it is one site in this image, not a share of ~190.
