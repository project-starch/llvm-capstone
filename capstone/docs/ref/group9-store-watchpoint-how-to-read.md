# The group-9 store watchpoint: what it sees, what it cannot, and how to aim it

Reference for `core/tracer.sv` group 9 and `CSR_WATCHPOINT_ADDR` (0x811), written 2026-08-25 after
it produced the first like-for-like store/reload comparison on silicon. Line numbers are at
`80843404c`, the revision the flown bitstream was built from.

## What it is

A store watchpoint, selective by **physical address** rather than by opcode class. That is what
makes it usable where group 2 is not: the monitor's trap path issues an `LDC` on every timer tick,
so any opcode-keyed group is scavenged out of a 256-entry ring continuously — measured, and
**identical in a wedging and a non-wedging arm**, so it separates nothing. An address-keyed group
the monitor never touches does not flood, and 256 entries becomes headroom instead of a ceiling.

    cva6.sv:904-906
      watchpoint_hit = lsu_commit_commit_ex
                     & (st_commit_paddr[PLEN-1:3] == watchpoint_addr[PLEN-1:3])
                     & st_commit_be[watchpoint_addr[2:0]]

## The payload is the CURSOR, not the metadata

Established two ways, which is why it can be relied on:

**Structurally** — `store_buffer.sv:197` `st_commit_data_o = ....data`; `:118` `.data = data_i`;
`store_unit.sv:462` `data_i = st_data_q`; `:369` `st_data_n = data_align(..., lsu_ctrl.data)`.
The **metadata is a separate lane**: `store_unit.sv:377`,
`st_user_n = (op inside {STC} || sel_dom_switch) ? lsu_ctrl.user : '0`, with an in-source comment
explaining they must stay disjoint so an ordinary `sw` holding stale metadata presents zero.

**By measurement** — a directed test storing a capability with a distinctive cursor reports that
exact value as the group-9 payload.

**So a group-9 payload is directly comparable with `tval` at a capability trap**, which carries the
rs1 cursor (`ex_stage.sv:488`). Same quantity, same width, same half.

## THE BLIND SPOT: it only sees WORD 0 of a granule, and this cuts both ways

The address compare is `[PLEN-1:3]` — **64-bit words** — and a 16-byte `stc` occupies a *single*
store-buffer entry whose `.address` is the granule base. So:

- **Aiming.** A watchpoint set anywhere in `G+8 .. G+15` compares against word 1's tag while the
  entry carries word 0's, and **silently never fires**. Always aim at the granule base. A
  capability store must be 16-byte aligned, so a capability slot's address *is* the granule base —
  but "the address I care about" and "an address this can match" are different statements that
  merely coincide here.
- **Absence.** Plain `sd`/`sw` stores *are* visible — nothing in the hit condition tests `is_cap`,
  and `st_be_n = lsu_ctrl.be` is set for every store — **but only if they hit word 0.** So an
  absence of entries proves nothing nulled **word 0**, not "nothing nulled the slot". State it that
  way. The cursor lives in word 0, so for cursor questions the distinction costs nothing; for
  metadata questions it is fatal.

## Arming, and the trap that wastes a boot

`0x810` (trace enable) and `0x811` (watchpoint address) are both **U-mode accessible** — bits[9:8]
of each address are `00`, and `csr_regfile.sv:2644` gates on `access_priv < priv_lvl`, never true
for privilege 0. **So arm from the host `.user` program in Linux userspace**, not from a domain and
not from the monitor:

- zero instructions are added to any domain image, so the perturbation that makes this fault
  disappear applies to *neither* arm;
- no monitor change, and no CAPENTER-ordering question;
- `trace_enable_q` is assigned in only two places (`csr_regfile.sv:1787` and the `:1111` hold), so
  **nothing clears the mask but hardware reset** — arm once and it covers every later arm.

**Never arm via GDB and verify at the same halt.** That read cannot distinguish the hardware
register from the debugger's copy; three boots were spent on an empty ring under a mask that had
never landed. Verify by reading the CSR **from the running core** and reporting it out.

## DBAS IS NOT STABLE. Carry the VA offset, never the physical address

`DBAS` differs between boots *and* between arm positions in one boot. A watchpoint address computed
from one run's DBAS and used in another points into a different allocation, group 9 fires on
nothing, and **an empty result reads as "the store never happened"** — a fabricated root cause that
looks clean.

    va_offset  = slot_va - DBAS_of_that_run     <- stable for one binary
    slot_paddr = DBAS_of_THIS_run + va_offset   <- recompute every run

Range-check the derived address against `[DBAS, DBAS + domain_size)` and refuse to run if it falls
outside, so a wrong address is loud rather than silent. This is not hypothetical: it fired on the
first real attempt, where one arm landed at `0x82400000` against a value derived from `0x82800000`.

## Reading a result

- **Ring never wrapped** (entries << 256) means the dump is *every* committed store to that address
  for the whole run, not a window onto the tail. That turns an absence into an exhaustive one, and
  it is a much stronger statement than any single comparison — lead with it.
- **Chronological order of the printed indices is NOT established.** Do not infer it from the index
  numbers; the ring's read-pointer semantics decide it, and a cross-iteration comparison would be
  a different claim.
