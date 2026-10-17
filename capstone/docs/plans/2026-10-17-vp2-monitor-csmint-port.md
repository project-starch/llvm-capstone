# vp2: porting the CSMINT emulation into the board monitor (plan, 2026-10-17)

**Owner:** the board lane. **Status:** planned, not started. It is not needed for the current bitstream.

**The decision it implements** (the runtime and RTL lanes, 2026-10-17): on vp2 an S-mode CSMINT traps to M (cause 2),
and the monitor emulates it. The emulation exists, validated end to end in RTL simulation, as the M-mode handler of
`tests/rtl-smoke/vp2-kit-2026-10-17/csmint-emu/va-csmint-emu.S`:
- capstone-ariane virtual-profile-2-d bf3db3175;
- 31 of 32 readings as pre-registered, and the 32nd a wrong prediction;
- every refusal in QEMU `helper_csmint`'s order.

This plan moves that routine into capstone-sbi on the exact-regions line, which is vp2's monitor (capstone-bootstrap
234ee66 and its successors).

## Where it goes

The monitor's capability-mode trap entry (`sbi_capstone.S`, `_cap_trap_entry`):
- saves S's 31 GPRs to a frame on the monitor stack, then calls `handle_exception(cause)` (`sbi_capstone.c`, FPGA
  build);
- if the return value is not -1, writes it to `x[rd]` in the frame (rd from mtval[11:7]) and advances mepc by 4;
- if it is -1, writes nothing and leaves mepc alone. The CPMP-swap path returns -1, so the access re-executes.

Today the CAUSE_ILLEGAL_INSTRUCTION case emulates only the `time` CSR read, and reports and spins on anything else
(I-4 site ILLX). CSMINT becomes a second emulated instruction there.

## What the arm did that the monitor cannot do as-is, and the port's answer

| arm | the monitor |
|---|---|
| reads rs1 and rs2 straight from the live registers, by an operand convention | reads `x[rs1]` and `x[rs2]` from the trap frame. `handle_exception` gains the frame capability as a second argument (`MOVC(a1, sp)` before the call; the stack capability is non-linear, as the existing `MOVC(a0, sp)` before `save_smode_context` already relies on). |
| reads the descriptor with three `ld` under MPRV | **walks the descriptor's VA in software too, and reads the words physically.** A page fault on an MPRV load inside `handle_exception` would trap into `_cap_trap_entry` again, which is not reentrant: it swaps sp through CSCRATCH. A software walk turns that fault into a refusal the routine reports. The descriptor is 24 bytes at an 8-aligned VA, so it spans at most two pages. Each doubleword is translated separately, as QEMU's helper does. A load needs V, R, A, and U only with SUM, or MXR for an execute-only page. |
| prints the status and skips the instruction | **injects** the cause into S with S's trap state: scause, sepc = mepc, stval, sstatus.SPP/SPIE/SIE, then mepc = stvec and MPP = S. It returns -1, so the entry neither writes rd nor advances mepc. The causes are 4 (descriptor misaligned), 13 (descriptor page fault), 6 (slot misaligned), 15 (slot page fault), 29 (bounds, perms, unrepresentable, or CSNODE's corrupt table) and 30 (CSNODE's reserve). The monitor has no redirect to S today, so this is new code. |
| stores through a NONLIN capability over all RAM | stores through **Linux's region capability**, the region that covers the slot's PA, used in place: cursor moved, STC, cursor restored. That region is LINEAR, so it cannot be copied the way the arm's NONLIN RAM capability was. If it sits in a CPMP slot, it is read and written back as `swap_cpmp` and `create_region` do. |
| catches CSNODE's 30 or 29 inside the window with a flag register | the same, with the flag in a monitor global. CSNODE faults in M, inside `handle_exception`, so the trap re-enters `_cap_trap_entry`. **That re-entry needs its own answer before the port is written:** either the entry checks the flag before swapping CSCRATCH and takes a short path that restores mepc and mstatus, or the routine checks the free-identity count itself before the window (the header's capacity, next-unused and free count). The second keeps CSNODE's fault as a defence only. The RTL lane's ruling on reading the table first (the barrier) applies. |

The routine's other steps carry over from the arm unchanged:
- the decode, from mtval or from the instruction at mepc when mtval reads 0 (the latter becomes a software-walked
  physical fetch too, for the same reentrancy reason);
- both alignment checks;
- the slot walk's A/D rule;
- the representability rule;
- the mint with CAPCREATE/CAPTYPE/CAPBOUND/CAPPERM;
- the three-instruction `csupctl.table` window.

## How it is tested before vp2

- **Not on QEMU as it stands.** QEMU executes CSMINT itself, so the trap never happens.
  - **Option 1:** a QEMU property, say `x-capstone-csmint-trap`, that makes S-mode CSMINT raise cause 2 with the
    bits in tval, as the RTL does since bf3db3175. Then the monitor path runs under the existing Linux-on-QEMU flow,
    and its rd and slot readings compare against QEMU's own CSMINT on the same input. That is a capstone-qemu change
    for the lane that owns the QEMU tree.
  - **Option 2:** an RTL simulation of the monitor's routine inside a bare harness that links the monitor's object.
    It is cheaper than a Linux boot, but it does not test the trap entry.
  - Recommendation: Option 1. The trap entry, the frame and the S-injection are exactly the new parts.
- **Then the board**, on vp2 with Tier 1's contract.dom, which calls CSMINT through the launcher.

## Before the board can run it (the other vp2 firmware items, from the virtual-profile plan)

- The board moves onto the exact-regions monitor line (capstone-bootstrap). The shared checkout still runs the
  classic pins d04bd83 / df0e9a8 / 93440d5.
- The 496 MiB memory map:
  - CMA moves inside [0x8000_0000, 0x9F00_0000);
  - the monitor's bound on Linux's region (0xC0000000, R-55) moves to the new top;
  - the boot-carve replay (`r55-regions/replay.py`) is re-run on the new map before the flash.
- The page-table walker's PMP needs no change: OpenSBI's Domain0 already permits S/U over all memory.
- The supervisor's CPMP coverage needs no change: genesis CPMP entries plus `swap_cpmp`, settled with the RTL lane.

## Open questions, and whose they are

1. QEMU's trap option (Option 1): the QEMU tree's lane.
2. The re-entry answer for CSNODE's fault: the board lane, with the RTL lane's ruling on reading the table first.
3. Whether contract.dom's CSMINT descriptors and slots stay in pages Linux has already touched, A and D set. If a
   slot's page is clean, the emulation injects 15, Linux sets D and returns, and CSMINT retries. That is correct, but
   it changes the count of traps per launch.
