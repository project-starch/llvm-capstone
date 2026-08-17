# S-07 instrument generation 3, and a lint gate that could not fail

Date: 2026-08-18. Lane: RTL. Peer: board lane (silicon results quoted here are theirs).

Two things happened. The S-07 debug instrument was redesigned after silicon showed the
previous one measures nothing, and the synthesis-hazard gate that has been signing off every
RTL hash this week was found to have been running a **truncated analysis** since it was
written.

---

## 1. The instrument was measuring nothing, and silicon said so

Generation 2 (`a2ef8eb` -> `618f4ce`) recorded the **first** LDC response that came back with
tag=0 since reset, one-shot. The board lane read switch 208 immediately after boot, before any
domain ran:

```
PRE-RUN baseline      sw=208 = 0xb8     ldc0_valid=1 src=1 stc_valid=1 stc_ctag=1 match=0
after control domain  sw=208 = 0xb8     identical
after XU rep 1/2/3    sw=208 = 0xb8     identical
```

The one-shot is spent by **boot-time software** — Linux/OpenSBI/the entry glue — long before
the workload starts. The byte never moves again, so it carried no verdict on any of the runs
it was read for. Worse, it fails SILENTLY: `ldc0_valid=1, gran_match=0` reads exactly like a
legitimate unmatched result and would have been reported as one.

`src=1` on that record means the untagged response came from a **miss refill during ordinary
boot software**. So untagged LDC responses are **routine, not anomalous** — which is the
S-06 fix behaving as designed (an untagged 128-bit granule loads without faulting). The
anomaly was never "an untagged LDC happened"; it is "an untagged value was USED as a
capability".

### The fix, and the correction it needed

Rolling instead of one-shot: drop the `!s07_ldc0_valid_q` guard, so the record always holds
the most recent untagged LDC and the **wedge** is what freezes it. Considered and rejected:

* a **clear** aperture like 191's — has to be issued at exactly the right moment, and the
  driver has already shown how easy that is to get wrong;
* **scoping to capmode/domain execution** — the monitor's entry glue runs in capmode too, so
  that boundary does not exist; and it still leaves a first-vs-last choice inside the scope.

**Rolling was then overstated by this lane and needed correcting.** "The wedge freezes the
record on the load before the fault" is only true if nothing untagged intervenes — and
untagged LDCs are routine. `stall_waw_rs1` stops the *consumer* issuing until the producer
commits; it does not stop unrelated capability loads completing in that window. A record
clobbered by ordinary traffic is indistinguishable from a correct one: the same failure shape
as the one-shot, one level up.

So generation 3 adds a **producer/consumer correlation computed in hardware**: `rd` of the
last committed LDC versus `rs1` of the faulting instruction (switch 193). That is the
four-wedge invariant *tested* on silicon rather than inferred from four disassemblies.
`match=0` marks the verdict byte void instead of letting it read as a finding. Honest limit,
recorded in the RTL: it is a 5-bit equality, so unrelated instructions coincide 1 time in 32 —
`match=1` corroborates, it does not prove.

---

## 2. What an adversarial audit found in this lane's own diff

Run before any of it was proposed for synthesis. Four claims audited; **one refuted**, one
downgraded to plausible-but-unproven, and the refutation was correct.

| Finding | Status |
|---|---|
| Evidence apertures placed at **Hamming distance 1** from the destructive one-shot trigger | REAL. Switches are eight independent mechanical toggles with no debounce, so every read is a *walk* through intermediate values. Reading 221 then 222 transits `11100` in one of the two orders and fires the control mid-sequence, overwriting essentially the whole instrument. Fixed with a ~21 ms dwell counter plus a two-flop synchronizer. |
| The `selftest_seen` marker "always distinguishes synthetic from real" | FALSE, and falsified by the same commit that made the record rolling. The bit is sticky-until-reset and means "the control fired at some point". The authoritative synthetic detector is the **paddr sentinel** `0x5A5A0`, because it travels with the record. |
| Two producers of mcause 25 behave oppositely at commit | REAL. The capability-unit path asserts no commit ack, so the faulting instruction cannot overwrite the producer record — that is the case the instrument is built for. The PC-capability check in `commit_stage.sv` raises cause 25 **with** the ack asserted; if that instruction is an LDC, the latch would overwrite the producer with the consumer, pinning the correlation bit at 0 forever. Fixed by gating on `!ex_commit.valid`. |
| The rs1 capture passes interrupts | REAL. The enclosing filter excludes only causes 0 and 2, so every timer interrupt re-latched it from whatever sat at commit port 0. Narrowed to cause 25. |
| The switch-216 sentinel as a **generation discriminator** | REFUTED, and it was already live in the board lane's decoder. On the previous bitstream reg 24 was `tval[47:40]`, which reads `0xFF` for any sv39 upper-half address — decoded on the new map that is `{sentinel=1, valid=1, cnt=63}`, i.e. "untagged LDCs are saturated routine traffic", forged off a bitstream with no census at all. Reachable with kernel VAs latched repeatedly the same session. Generation is now keyed on switch 193/194, whose previous encodings have bit 7 **hard-zero by construction**. |

---

## 3. The gate that could not fail

`verif/sim/rtl-lint-gate.sh` exists because a commit passed the 79-test sweep bit-identically
and then made `synth_design` balloon to 343 GB. Its header names "a combinational self-loop"
as half that defect.

**It could not detect one.** The vendored FPU (`core/cvfpu/src/fpnew_*.sv`) emits 50
`%Error-BLKANDNBLK`, and an error makes verilator stop after its early passes — before the
ordering pass that finds combinational loops. Every `LINT GATE PASS` this week, including the
numbers attached to hashes handed to the board lane, came from a truncated run.

It was found by **trying to make the check fail**, not by reading it:

```
deliberate `assign x = ... && !x;` in core/cva6.sv
  -> lint output MD5 8b596f7d4eb0e6c50f3c890abc29803e, 386348 bytes
clean tree
  -> lint output MD5 8b596f7d4eb0e6c50f3c890abc29803e, 386348 bytes
```

Byte-identical. With `-Wno-BLKANDNBLK` (which the model build already sets) the run completes
and the same loop is named by signal and line. Fixed; `UNOPTFLAT` is now counted, and the
counter is positive-controlled at 39 clean / 40 with the loop injected.

Two precisions worth keeping, both from the board lane:

* The gate was **partially** proven, not unproven. Their negative test against the real
  defect fired — MULTIDRIVEN 3->6, ALWCOMBORDER 0->1, COMBDLY 0->4, naming
  `load_unit.sv:708`. Those classes are emitted *before* the abort. What the gate could not
  have caught is a **pure** combinational loop with no NBA-in-comb. "Proven for three of four
  counters" was the accurate claim.
* The truncated run emitted 3249 lint lines, the complete one 3358. The 109-line difference
  was a visible signature of the abort, present in every run, unnoticed by both lanes.

New baseline: `LATCH 52 / MULTIDRIVEN 3 / ALWCOMBORDER 0 / COMBDLY 0 / UNOPTFLAT 39`.
Measured independently at `618f4ce36` — the RTL of the bitstream on the board — by the board
lane in an isolated worktree: identical. So that bitstream adds no combinational loops, and
that is now measured rather than inferred.

---

## 4. Silicon results this generation produced (board lane)

* **The selftest fired on hardware.** `pre-204 = 0x00`, `220 = 0x01`, `post-204 = 0x41` —
  count moved by exactly one, marked synthetic at 208 bit 0. Every `0x00` on 204 in that boot
  is a **controlled** negative rather than an argued one.
* **A third wedge, same instruction** (`ldc a4,0x20(a4)` at `sqlite3OsRead+0x4c`, mcause 25),
  `204 = 0x00` — on a boot where the detector was proven live in the same session. **Case (a),
  syncer displacement onto a scalar writeback port, is refuted with a controlled negative.**
  Precision retained: the two earlier wedges were on `a2ef8eb`, a different place-and-route of
  the same detector logic, so those are strongly supported; this third one is controlled
  outright.
* Still open: **(b)** the load genuinely returned tag=0, or **(c)** the granule was never
  tagged. A retry probe remains unsafe — it would confirm (b) wrongly under (c).

---

## The lesson, which is already a rule

"A CLEAN result is not evidence until the check is known to fire" covers every item above; no
new rule is needed. What this session adds is the *frequency*: **three checks that could not
fail were found in one day, and every one was found by trying to break it rather than by
reading it.** A gate that has never blocked anything is not a passing gate. Negative-test it
the day you write it — and when a result is surprisingly clean, suspect the instrument.
