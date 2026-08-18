# S-07 instrumentation: the complete reader set, specified once

**Status:** specification, not yet implemented. For the RTL lane.
**Target:** one bitstream carrying every reader S-07 could need, so no further synthesis is
spent on this defect.

## Why this document exists

Synthesis takes hours and fails all-or-nothing; board runs take minutes and fail in isolation.
That asymmetry is why board experiments are batched wide and RTL has been batched narrow — and
narrow batching is what produced the complaint that prompted this document: a reader at a time,
a synthesis at a time.

But "one big batch" is not the answer either, and the evidence is in this repo. Gen 3
(`d65c67589`) *was* the batched request — rolling records, census, 208 positive control, lint
gate, all at once — and it drove `synth_design` past 100 GB and had to be withdrawn
(`8c75d899b`), keeping only the rolling record. A batch large enough to be interesting is large
enough to be unsynthesizable, and when it fails you learn nothing about which part was to blame.

**The resolution is this document.** Every reader is specified here, once, up front. They are
partitioned into tiers that are *independently droppable* and ordered by synthesis risk. If
synthesis blows up, the RTL lane drops the highest surviving tier and retries — no new design
conversation, no round trip, no question to this lane. Ordering is what makes a wide batch safe,
exactly as domain ordering is what makes a wide board batch safe.

## What is already in the flashed bitstream

`caplifive_s07debug_18august.bit` = `capstone-ariane 6882b265f`.

| reader | where | note |
|---|---|---|
| rolling untagged-LDC record (`valid`, `src[1:0]`, `paddr`) | `core/load_unit.sv:774-778` | overwrites on **every** untagged LDC response |
| rolling cap-STC record (`valid`, `ctag`, `paddr`, `clobbered`) | `core/store_unit.sv:545-554` | overwrites on **every** capability store; `clobbered` set by a later plain store to the same granule |
| `s07_gran_match` | `core/cva6.sv:1026-1027` | combinational compare of the two paddrs, granule granularity |
| on-silicon selftest | `core/cva6.sv:1025`, switch 220 | proves the detector fires |
| verdict byte | switch 208 | packs valid/src/ctag/match/clobbered/selftest |
| displacement byte | switch 204 | `{stc_seen, ldc_seen, count[5:0]}` |

**Both records ROLL. Neither is a one-shot.** Neither capture condition carries a
`!..._valid_q` guard, and the `load_unit` comment says so: *"capture EVERY LDC response that
comes back untagged, so the record holds the most recent one and the wedge is what freezes it."*
The Python driver asserted the opposite until 2026-08-18 and printed "PROBE ALREADY SPENT /
carries NO weight", which caused a boot to dismiss its own usable readings. Fixed in
`run_sqlite_stages_fpga.py` (`S07_RECORDS_ROLL`).

### The three things that actually limit us today

1. **Two 56-bit addresses are compressed to one bit before we ever see them.** `s07_ldc0_paddr`
   and `s07_stc_paddr` exist at the `cva6.sv` top level and are *not* on the mux. We read
   `gran_match` and nothing else, so a match cannot be mapped onto the domain's memory map and a
   mismatch cannot be attributed.
2. **No register-level correlation gate.** Gen 1 has no 193/194 `rd_rs1_match`, so a granule
   match between two independently-rolling records is *suggestive, not licensed* — either record
   may have rolled onto that granule on its own.
3. **No scoping.** Nothing clears or arms the records, so they cannot be attributed to a domain.
   `ldc0_valid=1` is set during Linux boot as a matter of routine.

Naming the site — *which instruction stored it, which instruction loaded it back* — is what none
of the current readers can do, and it is the thing that would end this investigation.

### The concrete case Tier 0.1 would settle immediately

On BOTH boots taken on this bitstream (2026-08-18), the PRE-RUN baseline read
`sw=208 = 0x9c`, bit-identical, before any test domain ran:

    ldc0_valid=1  src=0 (L1 array)  stc_valid=1  stc_ctag=1  gran_match=1  clobbered=0

That is the S-07 signature itself — an LDC returning untagged on the same granule as the most
recent tagged STC, with no intervening plain store — occurring **reproducibly during boot**, in
Linux/OpenSBI/the entry glue. If it is genuine it is a far cheaper repro than the SQLite wedge,
which needs several boots to catch once.

It is NOT yet a conclusion, for four reasons, and each one is answered by exposing the paddrs:

1. **No correlation gate.** Both records roll independently, so a granule match can in principle
   be a coincidence of two unrelated events. The addresses would show whether the granule is a
   plausible shared object or an unrelated collision.
2. **Back-to-back LDC skew.** `s07_ldc0_paddr_q` captures `cap_clear_addr_q`, which is written
   for every LDC in `SEND_TAG_LDC` (`load_unit.sv:489`) — so it is the in-flight LDC's address,
   not a stale clear-path address, which is the good news. But `load_unit.sv:542` accepts a new
   LDC while the previous one is still sending its tag, so under overlap the captured address can
   belong to the newer LDC.
3. **`clobbered` does not see every writer.** It is set from `store_buffer_valid`, so a write
   that reaches the granule by another path would not clear the flag, and a legitimately
   invalidated tag would read as loss.
4. **An untagged LDC during boot does not fault.** Post-S-06 it is legal until the result is
   used as a capability, so this pattern may be entirely benign boot behaviour rather than the
   defect.

With the two addresses on the mux, 1 and 2 are testable in one board read and 3 and 4 become a
question of mapping one number onto the monitor's memory map. Without them, this observation
cannot be advanced at all.

## Tier 0 — pure mux decode. No new logic at all.

Adding `case` arms to the existing `debug_led_o` mux in `cva6.sv`. No new registers, no new
fanin into any LSU cone, no term added to anything on the `UNOPTFLAT` list. If Tier 0 cannot be
synthesized, something unrelated is wrong.

The selector is `switches_i[7:0]` (`debug_byte_sel = sw[7:5]`, `debug_reg_sel = sw[4:0]`), 256
apertures, most of them free.

* **0.1 — `s07_ldc0_paddr` byte-wise, and `s07_stc_paddr` byte-wise.** The single highest
  value-per-risk item in this document: the data is already computed and already at the top
  level. Turns "match" into an address that can be mapped onto `DBAS` and the domain's
  disassembly, and turns "mismatch" into two addresses that can be compared.
* **0.2 — `s07_ldc0_src[1:0]` on its own aperture.** It currently rides in 208, where it has
  been read back as `3` (undefined encoding) on every non-zero sample. On its own aperture,
  with the other six bits driven to a fixed known pattern, a contaminated read is
  self-identifying instead of decoding as a plausible source.
* **0.3 — a signature nibble on every S-07 aperture.** Drive the top nibble (or any 4 bits not
  otherwise used) to a per-aperture constant. Any read whose signature nibble is wrong is
  discarded by the driver automatically. This is the cheapest possible defence against the
  readout contamination described below, and it costs literally no logic.

## Tier 1 — top-level `always_ff` only. Does not touch the LSU.

Everything here is fed from signals **already exported** out of `load_unit`/`store_unit`, so
nothing is added to a load/store combinational cone. This is the same structural argument that
made the surviving rolling record safe.

* **1.1 — an ARMED SHADOW of both records.** A debug switch sets `arm_q`; on the rising edge of
  `arm_q` the shadow clears, and thereafter the shadow latches the *first* exported record it
  sees. This gives domain scoping — "the first untagged LDC after I armed" — **without** adding
  an arm term to the capture condition inside `load_unit` (see 2.2, which is the unsafe way to
  get the same thing). Rolling and armed-first are complementary and both are worth having.
* **1.2 — commit PC latched on the rising edge of `s07_ldc0_valid_o` / `s07_stc_valid_o`.**
  `commit_instr_id_commit[0].pc` is already on the mux at 230-237. Latching it when the record
  updates names the site to within pipeline skew, which is enough to identify a function and
  almost always an instruction. This is the *cheap* version of 2.1 and should be built first;
  2.1 is only worth its risk if this proves ambiguous in practice.
* **1.3 — saturating census counters, in their own registers.** Count untagged-LDC responses and
  capability stores separately. A rate computed in hardware over a whole run is worth far more
  than the handful of samples a board session can take, and it is the reader that would settle
  per-boot vs per-run directly. **Separate registers, separate apertures** — the gen-3 mistake
  was sharing structure with `ldc_result_back`.

## Tier 2 — touches the LSU. Highest risk. Drop this tier first.

* **2.1 — true `ldc_pc` / `stc_pc`** plumbed into `load_unit`/`store_unit` and latched beside the
  paddr. More precise than 1.2, and the only version immune to pipeline skew. Requires new
  signals crossing a module boundary into the load/store path.
* **2.2 — an arm qualifier inside the capture condition** (`&& arm_q`). Adds a term to a
  condition in the cone that carries `ldc_result_back`, which is on the `UNOPTFLAT` list.
  **This is the highest-risk edit in this document.** 1.1 obtains the same capability from
  outside; prefer it and build 2.2 only if 1.1 proves insufficient.
* **2.3 — injection / forced tag clearing** for a positive control deeper than the 220 selftest.
  Genuinely useful — it would let the whole detection chain be negative-tested end to end — but
  it modifies the data path, not just observation. Last, always.

## The readout path — NOT conditional any more. This one is required.

**The LED mux cannot deliver a trustworthy non-zero reading on this bitstream, running or
halted.** Both halves were tested on 2026-08-18 and both fail, by different mechanisms:

* **Running:** the pulse stretcher (`corev_apu/fpga/src/ariane_xilinx.sv:956-979`) holds each bit
  high for 2^20 cycles (~21 ms) after it is last driven, so a reading is the bitwise OR of every
  aperture visited on the way — including the transit apertures of the switch walk. Raising the
  settle 4x (0.5 s to 2.0 s) changed nothing, so this is persistent reload, not a decaying tail.
  It produced `src=3`, an undefined encoding, and `count>0 with ldc_seen clear`, which the
  encoding is designed to make unrepresentable.
* **Halted:** a halted core drives nothing, so the stretcher freezes and **no `led_state` event
  is emitted at all**. Boot 5 is the proof: the halted read of 204 saw exactly one event (`0x7c`,
  emitted *during* the switch walk, hence contaminated) and the halted read of 208 saw none and
  returned that same `0x7c`. Two apertures with entirely different field layouts cannot both hold
  `0x7c`; it was one cached value read twice. The driver's two-sample agreement check passed
  while comparing a stale value against itself — the exact failure its own docstring warns about,
  reintroduced by a `latest()` fallback. Fixed: a halted read with no fresh event now returns
  VOID.

So the "halt, settle, sample, resume" protocol does not exist and cannot be built. **The fix is a
JTAG-readable debug register, not more LED apertures** — the debug path already works (it is how
`mcause`/`mepc`/`mtval` are read at a wedge), and it is immune to both mechanisms above.

This raises Tier 0's priority rather than lowering it: every Tier 0 reader is worth strictly more
once it can be read at all, and 0.3's signature nibble stays worth building because it makes
contamination *detectable* rather than merely suspected.

**What survives from the LED path:** zeros, and only while running. The stretcher can only turn
bits ON, so a `0x00` read cannot be manufactured by contamination. Every `0x00` result on record
still stands. Halted zeros do not, because a halted read has no fresh event behind it.

## Rules for the batch

* **Separate registers per reader.** Never share a mux term or a condition with
  `ldc_result_back`. Gen 3 did, and that is the readable difference between it and the
  rolling record that survived.
* **`RETIMING` off for debug bitstreams.**
* **Never add a term to a cone on the `UNOPTFLAT` list.** If a change does, it goes to synthesis
  before it goes anywhere else.
* **Run `synth-guard.sh`** (`c5043e8a8`) — hard memory ceiling, memory trace, timing.
* **`verif/sim/rtl-lint-gate.sh` must pass, and run `claim-auditor` over the diff.** Both are
  necessary and neither is sufficient: both passed on the change that blew past 100 GB. **A hash
  is ready when synthesis has RUN.**

## Acceptance criteria

Each reader ships with a way to make it produce the *opposite* result, or it is not evidence:

| reader | positive control |
|---|---|
| 0.1 paddrs | a domain that stores a capability to a known address must read back that address |
| 0.2 `src` | the 220 selftest must move it to a defined encoding |
| 0.3 signature | deliberately read a contaminated aperture and confirm the driver discards it |
| 1.1 armed shadow | arm, run a domain with no capability traffic, confirm the shadow stays clear |
| 1.2 commit PC | must land inside the domain's address range, not in monitor or Linux |
| 1.3 census | must be zero on a boot with no domain, non-zero after one |

A reader with no positive control is an unproven reader, and this project has published
retractions on exactly that.
