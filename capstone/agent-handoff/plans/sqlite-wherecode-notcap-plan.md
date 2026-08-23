# Plan — the SQLite `sqlite3WhereCodeOneLoopStart` NOT_CAP wedge

**Status: NOT root-caused by this investigation.** A tight localization, a large set of
controlled exclusions, and a strong candidate mechanism that is *already documented elsewhere*
and does not cleanly fit.

## 1. What is ESTABLISHED (each by its own control, on silicon)

| Fact | How |
|---|---|
| Faults at `sqlite3WhereCodeOneLoopStart+0x8c`, `cincoffsetimm a4,a4,0xb0` right after `ldc a4,0x0(a0)` | latched `mepc` mapped through per-arm `DBAS`, disassembled in 3 binaries, 7 boots |
| The operand's `cap_type == NOT_CAP` | `mcause 25` from the FLU; `cincoffsetimm` is the IMMEDIATE form whose guard is a single `NOT_CAP` test, verified by disassembly |
| Producer is the FLU, **not** commit_stage | `tval != mepc` across 5 boots; the three log registers latch from one event (`cva6.sv:1126-1136`) |
| Writeback-port displacement EXCLUDED | switch 204 = `0x00` halted, with the 220 selftest firing in-boot; 7 boots. Ports 0/4 carry cap data, port 3 is FPU-only, ports 1/2 watched |
| Arm POSITION is not the variable | un-probed wedges at arm 2 AND arm 3; probed completes at both |
| The INPUT is a variable | `q_one` completes, `q_two` wedges — same binary, same arm |
| Instrumentation removes the fault | probed completes, un-probed wedges — same input, same arm |
| **Delay alone is NOT the variable** | `loop3` (~10 dynamic instrs) completes where `pad10` (~10) wedges; a 66x delay sweep with alignment pinned shows no change |
| Reproducer is 588 bytes | `SELECT t1.a FROM t1, t1 AS y` vs `SELECT t1.a FROM t1`, empty table |

## 2. What is EXCLUDED

VDBE execution; setup statements; data volume; `ORDER BY`; rev-node exhaustion (head 630 of
65532); codegen live-range overlap; S-06; the AMO residual; miscompiled bounds check (the LDC
guard wraps every relational); writeback-port displacement.

## 3. What is DISFAVOURED but NOT excluded

* **Spill-side breakage** — the total type query reads healthy right after the spill, but only in
  runs the probe perturbed into completing. Testable via the last-wins STC recorder or a memory
  marker (below).
* **Delay/drain latency** — survives as an observation, not as a cause.

## 4. THE LIKELY ANSWER, and why it is not yet the answer

**S-07 is already root-caused** (`fpga-repros/S07-capability-untagged-on-reload/00-README.md:54`):
the write buffer hits at 64-bit WORD granularity so a granule's halves take separate entries,
each entry writes the WHOLE granule's tag on drain, and drain order is `rr_arb_tree`, not program
order. An older plain store to `G+8` drains after a younger `stc` to `G+0` and clears its tag.
Arms: `wb1` (plain `G+8`; `stc G`) 1107/16384 lost; `wb3` (same + 64 draining stores) 0.

**`wb1` vs `wb3` IS my `pad10` vs `pad600`.** Same mechanism shape, rediscovered expensively.

**Why it does not close:** S-07 requires a plain store *into the subject granule*. Mine has none.
Subject granule `[s0-0x70, s0-0x60)`; the nine stores in the window are at `s0-0x74`, `-0x5d0`,
`-0x5b0`, `-0x90`, `-0x98`, `-0x5a0`, `-0x10c`, `-0x110`, `-0x120`. Sites differ too: S-07 faults
in `sqlite3OsRead` at `0x2a83c` from a `sqlite3JournalOpen` memset.

## 5. PLAN, in dependency order

**P0 — settle identity before spending anything.** (in flight)
* Auditor attacking my granule arithmetic and store enumeration, including **stores made by any
  callee invoked between spill and reload**, which I never considered.
* RTL lane: is the S-07 fix in `caplifive_s10fix_80843404c.bit`? Can the reorder clear a granule's
  tag *without* a same-granule plain store (adjacent granules, `wr_idx` aliasing)? Is
  `sqlite3WhereCodeOneLoopStart` a known second site?
* **If it is S-07 or a known residual (S-09/S-10): STOP.** Record the new site in that folder,
  close this, and hand the reproducer over. Do not open a new issue.

**P1 — if distinct, get the spill-side fork in a FAULTING run.** Needs no new RTL.
* The STC recorder is **last-wins** and survives a wedge. Compare its `paddr` against the subject
  slot (now reported as `slotaddr` from a completing arm). On any wedge where they match,
  `stc_ctag` answers spill-vs-reload directly, unperturbed.
* If it never matches, use a **memory marker**: total `lcc` on the register being spilled, `sd`
  the answer into the shared region, read it over JTAG while halted. Memory survives a wedge —
  only reporting dies. **Control required:** a marker-only arm that must still wedge.
* Use a **light** probe: query the register about to be spilled. The current width probe forces an
  extra `ldc`, the operation under suspicion, which is likely why it removes the fault.

**P2 — placement sweep, expected to be a proxy.** Pads of 0/1/2/3/4 nops shift the faulting
instruction 4 bytes each at near-constant delay. A tight boundary is a real constraint on any
mechanism even when it is not the mechanism. Record as proxy, not cause.

**P3 — blocked on the project lead, surfaced not lobbied.** The s07 LDC recorder is one-shot with
no clear and is consumed by boot software before any domain runs, so it is unusable here. A clear
mirroring `dom_switch_log_clear` is purely additive and would make it work. Competes for a
synthesis slot with the latent dom-switcher defect; that priority is not mine.

## 6. Do NOT do

* Do not open an `fpga-repros` folder until P0 says this is distinct. One issue per folder, and a
  duplicate of S-07 would be worse than nothing.
* Do not build more sim arms of the delay/contention shape — the peer lane measured peak
  write-buffer occupancy at **1** against depth 8, so that environment cannot hold the condition.
* Do not read switch 208 without the granule-address attribution check; bit 7 alone passes a false
  positive.
