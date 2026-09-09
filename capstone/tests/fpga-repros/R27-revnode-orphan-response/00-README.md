# R-27 — a pipeline flush during the DYN unit's revocation-node query leaves the core dead

> **Status 2026-09-09: pre-existing on `ef5a8eaf2`, fixed in RTL** (`capstone-ariane` `66c4e7517`, the tip of
> `fpga-testing-dev`), **sim-verified** (three triggers HANG → PASS at the default latency; the load-fault trigger HANG →
> PASS at a verified 40-cycle latency), lint at the baseline counts, adversarially audited. **Bitstream: SYNTHESISED 2026-09-09 from `66c4e7517`
> (sha256 `b03bd967…52da3`, WNS −12.425 ns at 40 ns); not flashed, and silicon UNCONFIRMED** — no board arm exists yet, and whether historic board wedges were this is UNRESOLVED.

**Wrong symptom? Read this paragraph first.** This package is the **silent deadlock**: no trap, no UART output, the
core stops fetching a few instructions after a capability instruction. `../R26-ccsrrw-stale-cpmp/` is the permission
hazard whose fix exposed this. `../S13-o1-dyn-rev-node-hang/` describes a `-O1` domain "hanging in the DYN/rev-node
path with no exception" — the same signature; whether it was this defect is not established (its images are
unreconstructable). Registry R-28 is the write-op relative named by the audit (node state mutated for a killed
instruction), undemonstrated. `../RTL-domain-trap-vector-unset/` (M-1) is a trap that is not delivered; here no trap is
raised at all.

## The defect in one paragraph

Eight DYN-unit operations — REVOKE, SPLIT, TIGHTEN, CALL, RETURN, **LDC, STC**, LCC — ask the revocation node whether
their capability's node is still valid (`capstone_dyn_unit.anvil`, `get_node_query_validity`), a request/response
channel with a window of roughly four cycles after the op's dispatch. A flush that reaches EX in that window
(`flush_csr`, `fence`, `fence.i`, `sfence.vma`, exceptions and interrupts, `eret`, `flush_commit`; not a mispredict)
resets the DYN unit — `ex_stage.sv` wires `flush_i` to its flush endpoint and the generated unit drops every `recv`
sync-state — but not the node, whose only flush hook increments a counter. The node then completes `send ep.query_res`
into a channel nobody acknowledges and stays inside that send forever (`capstone_rev_node.anvil:55-60`; the generated
sync-state has no fallthrough, no timeout, no drain). The next capability instruction that queries waits forever,
`capstone_dyn_ready` stays low, issue, decode and fetch back up, and the core is dead with nothing to read but a
timeout. Waveform (`records/r26/cscratch-fence-hang.vcd`, half-cycle ticks): query request handshake 1139, `fence.i`
flush 1141, the DYN unit's ack drops 1143, the node's response rises 1145 **and never falls**; the re-issued
instruction's request rises 1683 and is never acknowledged. In the matched passing arm the flush lands twelve cycles
after the query completed. `fence`/`fence.i`/`sfence.vma` commit only when the store buffer is empty, so a draining
store moves their flush by tens of cycles — which is why the fence-timed triggers hit the window at one memory latency
and miss it at another, while the load-fault trigger does not depend on memory timing.

## The arms (`sim/`, list `sim/testlist_r26.yaml`)

| trigger | arm | `ef5a8eaf2` delay 0 | `ef5a8eaf2` delay 40 | with the drain |
|---|---|---|---|---|
| `csrw mstatus` + `fence.i` + CALL, older cache miss | `r26-diag-csfence-mstatus` | **HANG** | PASS | PASS |
| `fence` then eight LDCs (setup store draining) | `r27-fence-nost` (its one-line pair `r27-fence-st` passes) | **HANG** | PASS | PASS |
| **a load that faults, then eight LDCs** | **`r27-ldf-n0`** (`n1…n8`, one nop of separation, pass) | **HANG** | **HANG** | **PASS**, one trap |
| CCSRRW then LDCs, 0–52 nops, with/without an older miss | `r27-ccsr-*` | PASS | PASS | PASS |
| ecall / misaligned load then LDCs | `r27-exc-*` | PASS (no overlap: those exceptions are known at decode) | PASS | PASS |

**The load-fault trigger is the shape real code produces:** an ordinary access that faults immediately before a
capability access hangs the core instead of trapping. A timeout prints as the harness's `SUCCESS after time_out+13`.

## The fix (`fix/66c4e7517-ex_stage.sv.diff`)

In `ex_stage.sv`, per revocation-node response channel (init/rev/drop/delin/mrev/query): remember a request the node
accepted whose response is still owed; when `flush_i` lands while one is owed, drain that response — acknowledge it to
the node, hide it from the DYN unit — and hold `capstone_dyn_ready_o` low until it is gone, so a re-issued instruction
can never consume its predecessor's answer. One `always_ff`, two registers per channel, no new combinational path.
Stated assumptions and one known corner (the node's designed non-answer when its pool is exhausted, `head == 16'hFFFF`,
becomes a permanently held ready after the next flush) are in the RTL comment and the commit message. Flushing the node
itself was tried upstream and reverted (`f5f9291c8`, `d15d45b33`, `7bcbdb39c`).

## Run it

`bash run.sh <checkout> r27-ldf-n0` — HANG on `ef5a8eaf2`, PASS with one counted trap at `66c4e7517`.

## Records

`docs/history/09-09-2026_16-00-00_r25-r26-fix-cycle.md` (mechanism, waveforms, the audit, the latency correction).
Registry R-27, R-28. The waveforms themselves (20–30 MB each) are not committed; the tick tables above and in the
history note are the record, and `sim/callhang-fe-1660-1760.txt` is the frontend dump from the first hang.
