# Proposal: break the S-12 alias in codegen, so the SQLite corpus can run on the current bitstream

**Status: PROPOSAL, NOT STARTED, and DEPRIORITISED 2026-09-03 — kept only as a fallback.**

The project lead's call, and it is the right one: fix the hardware rather than work around it in
the compiler. Three reasons this should not be built while the RTL fix is live:

* it **contaminates every measurement** taken with it — performance numbers would come from a
  compiler dodging a hardware defect, which the paper would have to caveat permanently;
* it is **not even a complete unblock** — it covers only binaries this compiler builds, leaving the
  monitor, hand-written assembly and everything else exposed. 68 sites is SQLite's share of a
  machine-wide defect;
* it is **throwaway if the RTL fix lands** — a reflash is needed either way, and afterwards this is
  dead code plus a flag to maintain.

Build it ONLY if synthesis shows the RTL fix is unsynthesizable AND corpus numbers are needed
before that is resolved. See `s12-fix-synthesis-request.md`.

## Why this exists

S-12 is now root-caused as an RTL defect (see `capstone-ariane` commits `2c0f1917a` and
`e5e8e614a` on `s12-ldc-rolling-filter`, and the mechanism section below). That means there are
two possible fixes and they have very different costs:

* **Fix the RTL.** Correct, permanent, and gated behind synthesis (~90 min) plus a reflash, which
  is the project lead's call. It also does nothing for any result measured on the current
  bitstream.
* **Avoid the trigger in codegen.** Works on the silicon we have today, needs no bitstream, and
  is measurably cheap. It does not make the hardware correct and must never be described as
  fixing S-12.

These are not alternatives. The RTL fix is the real one; this proposal is what unblocks the
SQLite logic-test corpus in the meantime, which is the standing goal S-12 has been blocking.

## The mechanism, in one paragraph

A capability store decodes its scoreboard destination to its own store-data register
(`decoder.sv:1313`, `rd := rtype.rs2`; a plain integer store has `rd = x0`). When such a store
stalls on a full store buffer, `commit_stage.sv` asserts `we_gpr_o[0]` and clears **only**
`commit_ack_o[0]`, so `we_gpr = 1 / waddr = rX / ack = 0` persists for the whole stall while the
entry stays live. That `we_gpr` reaches the issue stage (`cva6.sv:1993 -> 1728`), where the WAW
guard is cleared by `we_gpr_i[c] && waddr_i[c] == rs1` — so the guard releases a consumer against
a producer that has **not** retired. Forwarding candidacy requires `still_issued & sbe.valid`, so
the written-back store entry is the only candidate and the consumer is handed `create_cnull()` =
`{cursor 0, cap_type 0}`, which the FLU rejects as UNEXPECTED_OPERAND with `tval = 0`.

Established in simulation with a matched single-variable control (`stc-ldc-sbpressure` vs
`-norel`: 254 traps vs 0, `escape = 258` in both). The architectural register holds a **correct**
capability at every trap, so the load lands and only the consumer is misfed.

## The rule to implement

Do not emit, for any register `rX`:

    stc  rX, <any>          # a capability store whose SOURCE is rX
    ldc  rX, <any>          # a capability load whose DESTINATION is rX
    <cap op reading rX>     # a consumer that ingests rX as an operand

Breaking any one of the three legs is sufficient; the cheapest is to make the store's source
register differ from the load's destination.

**This is the shape the board already tested one byte at a time.** The shipped 2x2 in
`capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/minimal-repro/` differs by a
single byte per pair and shows the relation deciding the outcome: relation present 12 wedges / 15
valid draws, relation absent 0 / 20. `02-clean-stc-t0.dom` is exactly this mitigation applied by
hand at one site, and it is 0/4.

## Measured cost, from the real binary

Counted over the 331,808 instructions of the built SQLite silicon domain:

| pattern | count | share |
|---|---|---|
| `stc rX` followed immediately by `ldc rX` (the bare alias) | 2832 | 12.8% of stores |
| the **full triple**, with a capability consumer reading `rX` | **68** | 0.02% of instructions |
| triples on `a4` | **1** | the S-12 site |

The bare alias is common and harmless: with no immediate capability consumer nothing ingests the
stale operand, so the escape is unobservable. Only the full triple is exploitable. **68 sites in
a 1.6 MB image** is the size of the problem — small enough that the mitigation should cost
essentially nothing in code size or performance.

This also answers the obvious objection to the mechanism ("why doesn't this fire constantly?"):
the alias is frequent, the triple is rare, and the triple additionally needs the store buffer to
be full at that instant — which is why the board rate is ~54% per draw rather than deterministic,
and why a `fence rw,rw` in the window cures it (0 wedges / 7, against a same-layout positive
control wedging 4/4).

## Implementation options, in preference order

1. **Post-RA peephole.** Scan for the triple; when found, rewrite the `ldc`'s destination to a
   free register and rename the consumer's use. Smallest blast radius, easy to gate behind a
   flag, easy to count. Needs a free register at the point of use; if none is free, fall back
   to (3).
2. **Register-allocation hint.** Add an anti-affinity so a capability store's source and the next
   capability load's destination do not coalesce onto the same physical register. Cleaner in
   principle, harder to make deterministic, and harder to prove it fired.
3. **Insert one independent instruction** between the store and the load. Weakest option and
   listed only for completeness: it changes timing rather than removing the alias, so it is a
   perturbation of the kind this bug has repeatedly punished. Do not ship this as the primary
   mitigation.

Recommend (1), behind `-capstone-avoid-stc-ldc-alias`, defaulting **off** until validated.

## Acceptance criteria

A criterion that cannot fail is not a criterion, so each of these names what would refuse it:

* **A test that FAILS without the flag.** A lit test containing the triple, asserting the alias is
  ABSENT from the output. With the flag off it must fail; confirm that before trusting a pass.
* **Site count drops to zero.** Re-run the triple counter over the rebuilt image: 68 -> 0. If it
  reports 0 for a build with the flag off, the counter is broken, not the compiler.
* **Byte-identical when off.** The corpus must be unchanged with the flag disabled.
* **lit + QEMU suites green** with the flag on (serialised; shared `rootfs.ext2` write-lock).
* **On silicon:** the SQLite domain that currently traps at `_start+0xF4874` completes. N >= 4
  draws, behind a known-good control, with the per-draw image sha verified inside the boot.

## What this proposal does NOT claim

* It does not fix S-12. The hardware defect is unchanged and every other triple on the machine —
  in any program, not just SQLite — remains exposed.
* It does not generalise to hand-written assembly, to the monitor, or to any binary not built by
  this compiler.
* It has not been shown that all 68 sites are reachable, nor that the S-12 site is the only one
  that has ever fired. A second site firing later would not contradict anything here.
