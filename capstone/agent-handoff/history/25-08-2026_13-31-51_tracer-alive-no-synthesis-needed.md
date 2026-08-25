# The tracer is alive: no synthesis is needed, and the defect was in the arming route

**Outcome: NO RTL change, NO synthesis, NO reflash.** Three RTL changes were designed over the
course of this investigation and none of them will be built. Total cost of establishing that: two
auditor runs, one RTL simulation, and three board boots that were spent anyway.

## What was believed, and what is true

The S-07 LDC recorder is first-wins and one-shot, and boot software consumes its slot before any
domain runs. Two RTL fixes were proposed to scope it, and a third to scope the STC side.

**None is necessary, because `core/tracer.sv` already logs what the investigation needs** — group 2
captures every `LDC` and `STC` commit with its PC and the **real tag bit**
(`tracer.sv:126-131`) — and it is present in the bitstream already on the board.

Three board boots with every group enabled (`0x1FF`) returned an **empty ring**, which looked like
a dead tracer. It is not. Directed RTL simulation, a real pass at 585 cycles rather than a timeout:

```
TRACER-DBG: trace_enable_i changed to 00000004 at time 484
TRACER-DBG: CAPTURE port 0 group 2 pc ...14e payload 0
TRACER-DBG: CAPTURE port 0 group 2 pc ...152 payload 1
TRACER-DBG: CAPTURE port 0 group 2 pc ...156 payload 0
TRACER-DBG: CAPTURE port 0 group 2 pc ...15a payload 1
```

Four capability accesses, four captures, correct group, correct PCs, and **payloads that vary** —
which matters more than their values, because a bit that is always 0 or always 1 is what a
tied-off signal looks like.

## The actual defect: the arming ROUTE, not the logic

The arming path is intact and was verified end to end at the resident revision `80843404c`:

    csr_regfile.sv:1787   CSR 0x810 write: trace_enable_d = csr_wdata[31:0]
    csr_regfile.sv:1111   default hold:    trace_enable_d = trace_enable_q
    csr_regfile.sv:3078   trace_enable_q <= trace_enable_d
    csr_regfile.sv:2862   trace_enable_o = trace_enable_q
    cva6.sv:2060 / :969   -> tracer.trace_enable_i

`trace_enable_d` is assigned in exactly **two** places, so **nothing clears the mask but hardware
reset**. That is what makes the empty ring diagnostic rather than ambiguous: if the write had ever
landed it would still be there, so it never landed.

The difference between the working simulation and the failing board is the **route**: in sim the
mask is written by an architectural `csrw 0x810` executed by the core; on the board the only route
was GDB, and the readback was taken at the same halt as the write — which cannot distinguish the
hardware register from the debugger's own copy. An unconfirmed mask makes an empty ring say nothing
about the logic.

## The fix, and why it is better than the one first proposed

The first suggestion was to read `0x810` back from inside the CONTROL domain, on the grounds that
perturbation is free there because that domain completes.

The better version, adopted: **arm it from Linux userspace, in the host `.user` program, before
entering the domain.** CSR `0x810` has `bits[9:8] = 00`, i.e. **User** privilege by the standard
CSR address encoding, and CVA6 enforces exactly that — `csr_regfile.sv:2644`,
`if (access_priv < csr_addr.csr_decode.priv_lvl)`, which for privilege 0 is never true. The second
check at `:2700` is also satisfied for a U-mode CSR.

This is strictly better because:

- **zero instructions are added to any domain image**, so the perturbation problem that removes the
  fault does not apply to *either* arm rather than only the control one;
- no monitor or firmware change, and no CAPENTER-ordering question — the
  `sbi_capstone_init.S` "supposed to invalidate all CSR setups" TODO becomes irrelevant rather than
  a risk to reason about;
- the mask is never cleared, so arming once in the host covers the wedge arm too.

## What stays designed and unbuilt

If the tracer had been dead, the batch was: drop `&& !s07_ldc0_valid_q` from `load_unit.sv:769`
(a strict fanin reduction restoring the rolling capture that already shipped in
`caplifive_s07debug_18august.bit` before `83a7d061f` reverted it); a watchpoint-granule filter on
the STC capture and its `clobbered` arm using `CSR_WATCHPOINT_ADDR` 0x811
(`csr_regfile.sv:214,:1788,:450`), because first-wins alone would latch the first of five `stc` in
the window rather than the subject; and the `ariane_pkg.sv:592` comment fix.

**Recorded so it is not re-derived**, not because it should be built.

## Two open items, flagged not concluded

- Whether a 16-byte `stc` presents usably on the commit paddr for a watchpoint filter. Unanswered;
  only matters if the batch is ever revived.
- The simulation showed **STC payload 0 followed by LDC payload 1** on the same granule — the
  inverse of the S-07 direction. Most likely an artifact of the directed test, where `CAPCREATE`
  alone leaves the register untagged until bounds and permissions are set. Flagged rather than
  built on.

## The process point

Three RTL changes were designed and none built. The first was refuted for being *correct but
insufficient*; the second and third were made unnecessary by an instrument that had been in the
bitstream the whole time. **The cheapest bitstream is the one nobody builds**, and the thing that
prevented all three was refusing to treat an unproven instrument as a dead one.
