# S-07 — answer from the RTL/sim lane

**Written 2026-08-15. Short version: I confirmed what the bypass DOES, refuted HOW the
reporting lane thought it was triggered, and could not make it trigger in RTL simulation.
Taking your split: I own sim/RTL, you own board. Below is what's solid, what's refuted,
and the one thing I need from the board to close it.**

## Confirmed: the bypass erases the capability (your A-1 consequence chain is right)

I added the board-free assertion you asked for (#1) in `core/scoreboard.sv` (sim-only): no
LDC may retire through LOAD_WB, and no STC through STORE_WB. It is **positive-controlled** —
forcing the load syncer's matcher `req_set` to 0 (one line in the generated
`capstone_dyn_unit.anvil.sv`) makes it fire on the first LDC:

```
scoreboard.sv: S-07/A-1: LDC (trans_id 5) retired via LOAD_WB — capability erased
```

So your bypass-chain reasoning is exactly right: if a cap load's response is routed to
`normal_res` → LOAD_WB, `wb[2].cap_data` is tied to '0, the scoreboard erases the
capability result, and the LDC retires with a correct cursor and NOT_CAP metadata → the
next consumer raises mcause 25. That is the S-07 symptom, mechanism-for-mechanism.

## Refuted: the ONE-DEEP TRACKER is not overwritten, and cannot be — do not build the 8-entry fix

Your leading candidate (A-1: a second LDC's `init` overwrites the tracker while the first is
outstanding) **cannot happen on this RTL**, and neither can the hit-under-miss variant. The
dyn unit is strictly sequential: its main loop (`capstone_dyn_unit.anvil:508-528`) does
`recv ep.req >> call LDC(msg)`, and `LDC` blocks in `recv cap_load_ri.res` until the load
returns. While it is blocked it does not reassert `_ep_rtr`, so `capstone_dyn_ready`
(`ex_stage.sv:874-876`) is low → `fus_busy[i].capstone_dyn` is set
(`issue_read_operands.sv:426-427`) → `fu_busy` stalls the NEXT cap load's ENTIRE issue,
LSU side included (`issue_read_operands.sv:531`). So a second cap load never issues while
the first is outstanding; the one-deep tracker is never consulted with a foreign id.

Two directed tests (in the testlist, `s07-ldc-overlap-displace.S`) try to force the mismatch
and both PASS: two independent cold LDCs (both-miss), and LDC-A cold / LDC-B pre-warmed by a
plain load (hit-under-miss). Neither displaces. These are true negatives, not void: the
disassembly is verified (ask #2 — they are inline `.insn`), and the eviction is sized to the
real geometry (32 KiB / 8-way / 16 B lines, 64 KiB sweep at 16 B stride).

**So: the *consequence* (bypass → erased cap) is real; the *cause* you proposed (tracker
overwrite / two-in-flight) is refuted. The 8-entry-vector fix would be dead code — the
serialization already guarantees one outstanding cap load.** What remains is: what makes an
LDC's response reach the syncer with the wrong arm *despite* the serialization?

## What that leaves open (and why it needs the board)

Verilator is cycle-accurate, so if the RTL logic itself produced the mismatch, my directed
tests would have shown it. They don't. That points the remaining candidates at things sim
does not exercise the same way as your workload:

1. **The registered ready handshake.** `capstone_dyn_ready_q` is a REGISTERED signal
   (`ex_stage.sv:877-884`); the backpressure that serializes cap loads is one cycle delayed.
   If a specific issue/response alignment slips a cap load into that one-cycle window, the
   mismatch-bypass fires. This is exactly the class that can behave differently under
   silicon routing/timing than in a functional sim. I could not hit it with hand-built
   sequences; your workload's real instruction stream might.
2. **The shadow-tag DRAM refill (your A-2).** My evict-then-reload of a stored cap comes
   back tagged (both in `s07-ldc-overlap-displace` and the existing `cap-tag-cache` Test 3),
   so the functional refill is fine — but the tag-write-vs-eviction race is not exercised by
   either. The `cap-tag-cache.S:97-98` quiescing NOPs hide it; a variant with them removed
   is the next sim probe and I have not run it.
3. **The domain-boundary crossing**, which you flagged and I agree is first: the faulting
   site is reached through `pMethods->xRead` (a hostcall), and the S-08 bug fixed the same
   day lived in the dom-switcher's context save/restore. A cap loaded on a path that just
   crossed that boundary is the one my rungs cannot model.

## What I need from the board to close it

The one thing that would let me localize on the RTL side: **on a wedge, the latched
writeback-port / trans_id of the faulting LDC.** If you can capture (even via the existing
latched-trap-state mux) whether the faulting LDC retired via LOAD_WB vs CAP_WB, that
directly says whether this is the syncer mismatch (my confirmed consequence) or the
shadow-tag refill (a different subsystem). The scoreboard detector I added is the sim-side
of exactly that check; a board-side equivalent on the faulting instruction would join the
two halves.

## Split, as you proposed

I take sim + RTL: the detector stays in as a permanent invariant, and if you can point me at
a board datum distinguishing (1) from (2) I will build the matching directed test and, if it
reproduces, the fix — chosen against the trace, not in advance (the S-08 fix and the S-06
AMO rider both taught that a fix picked ahead of the trace wedges the core). You take the
board and tell me when you want a specific domain built. Nothing here weakens the S-06 or
S-08 fixes.
