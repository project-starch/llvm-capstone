# vp2 item D: `CSRUNV`, the event kinds, `CSMINT`/`CSRETIRE` and FP state on the RTL (design, RTL lane, 2026-10-17)

*Status: DECIDED 2026-10-17 by the semantics owner (the runtime lane), all three on the simplest path: (1) the frame
is the context storage, 2 KiB on silicon, with the software rule that while a context is paused the frame below +1024
is the processor's and software writes only the reply slot +224 (silicon reads the other slots back on resume where
QEMU ignores them: a Q-entry, and a directed arm writes a register slot after entry to show silicon honouring it);
the ABI header's size assertions move to 2 KiB (compiler lane); (2) action 3 is illegal on silicon (cause 2) and
collection is software: the module's collector (compiler lane) walks the listed pages, clears the tag of every
granule whose node is dead, then the saved contexts, then STROBES THE TRACKER FLUSH (a write of csupctl bit 2 fires
revnode_ns_flush; the strobe comes after the last tag clear and before action 0), and the launcher takes the cause-2
trap on action 3 as "collect in software"; (3) outside a context, S-mode CSMINT/CSRETIRE and the supervisor's own
lifetime instructions use the table at the current srevroot when it is non-zero, the physical pool stays M-mode's,
and a directed arm shows an S-mode CSMINT landing in the table while an M-mode one lands in the pool. Kind 5's
reserve is 256 (QEMU reserves per instruction; the number goes into item A's Q-entry), pinned by the exhaust arm's
257-allocates / 256-refuses reading. The RTL work starts on the switcher's physical bypass and the frame walk.*

*Earlier status: PROPOSED. Three decisions below were the project lead's or the semantics owner's; the RTL
work starts on the parts that do not depend on them. Reference: QEMU `virtual-capstone` d35af47b163c
(`capstone_supervisor.c`, `capstone_table.c`), the ISA notes `docs/design/virtual-capstone/isa.md`, and the RTL at
capstone-ariane `virtual-profile-2` 724e8d393 (item A step 2). File:line citations are to those.*

## What QEMU does, in one paragraph

`CSRUNV rd, rs1, rs2` (opcode 0x5b, funct3 1, funct7 0x24) takes a PHYSICAL frame address in rs1 and an action in
rs2, from trusted S or scalar M (the Linux module `capstone_vm.ko` on the reference runtime; U, V=1, legacy C, an
active supervised context and actions above 3 are illegal, cause 2). Action 0 on an unknown frame is a first entry:
the frame (576 bytes, 16-aligned, RAM; 640 with options; 656 for collection) supplies satp (Sv39) at +0, srevroot
at +8, options at +56 (bit 0 resumable ECALL, bit 1 resumable page faults), the PCC at +64 and x1..x31 at +64+16i,
each slot tagged from the physical tag map; the 32 slots are then zeroed and untagged. The context starts in C with
every other register and CSR zero, mstatus from the caller's SXL/UXL with FS Initial, fp_status cleared to RNE. The
caller's whole state is snapshotted in the slot (not in memory), rd is remembered, a 5 ms quantum is armed. An
escape (`escape()`, sup.c:402-440) snapshots the context in the slot, restores the caller, writes the event
{kind, cause, pc, addr} at frame+16, the raw a0 at +48, the a0..a7 views at +576 when options != 0, and sets
x[rd] := kind; no trap CSR is written. Kinds: 1 quantum (paused; action 0 resumes), 2 terminal fault (only action
1 frees the slot), 3 service ECALL (options bit 0, cause 11; action 2 reloads a0 from frame+224 and advances the pc
by 4), 4 page fault (options bit 1, causes 12/13/15; action 0 re-executes), 5 node pressure (a RECYCLE table with
capacity - next + free_count <= 256 at CDERIVE/MREV/SPLIT; pc restored to the allocating instruction; cause 30;
action 3 then 0). Action 1 discards the slot (rd 0); action 3 collects the paused namespace (software-shaped: it
untags saved registers whose nodes are dead, clears dead tags in a listed page set, rebuilds the free list; rd =
reclaimed count or all-ones). `CSMINT rd, rs1 = slot VA, rs2 = descriptor VA` (funct7 0x50; S or M, not C) builds a
LINEAR capability from {base, end, perms}, allocates a child and an ancestor (cap_rev_create + cap_rev_mrev), stores
the capability to the slot with its tag, rd = the ancestor id; causes 4/6/7/29/30. `CSRETIRE rd, rs1 = id` (funct7
0x53; S or M, not C) retires the id's subtree and the id (a read-only bounded walk first), rd = 0 ok / 1 failed,
never faults.

## What the RTL has (item B and the supervised CALL)

The supervised CALL is armed by `CSSUPERVISE rs1 = seal, rs2 = save area` and entered by `CALL`: the switcher saves
the monitor's register ids 3..66 to the save area, exchanges ids 0..7 with the seal window, and on an escape
(commit_stage.sv:253-301, 747-770) does the reverse, writing csupstatus/csupcause/csupepc/csuptval and x[rd] := the
sealed seal. The quantum is `csupquantum` in cycles. Item B added `csupctl.virt`, and `vctx = sup_active &
csupctl.virt & satp.mode == SV` turns on translation at U permissions for the domain; item A put the lifetime table
at srevroot. The switcher saves no FP state; its save area is 944 bytes of 1024; the id map ends at 66 and the CSR
file swallows ids above 56.

## The design: CSRUNV as the virtual profile's supervised entry on the same switcher

1. **The supervisor is Linux in S mode** (the reference runtime: `capstone_vm.ko` invokes CSRUNV; `isa.md` and
   `virtual-capstone-runtime-r0.md:45` "trusted S or scalar M"). So the switcher's frame traffic is PHYSICAL while
   S-mode translation is on: the switcher's requests bypass satp (a `dom_switch_busy` term on
   `en_ld_st_translation`, the mirror of item B's `vctx_eff = vctx & ~busy`), and their PMP check is at S (the
   monitor's all-permitting entry before the first virtual context, the plan's memory-map section, covers it). The
   M-mode monitor path (CSSUPERVISE/CALL) is untouched.
2. **One slot in the CSR file**: {frame address, used, paused, terminal, event_kind[2:0], options[1:0], rd}. A
   second CSRUNV while a context runs is illegal (as today's `sup_active`); a frame that is not the slot's while the
   slot is used and not discarded is refused (29).
3. **Context storage is the frame, on silicon.** QEMU keeps both snapshots in the slot; the RTL has no 2 KiB of
   flops to spare, and the frame is the natural home: (a) the paused CONTEXT's registers live in the frame's own
   slots +64..+575 (x1..x31 and the PCC, tagged), its FP state at +656..+919 (fpr0..31) and +920 (fcsr), its CSRs
   (mstatus, trap CSRs, satp, srevroot) at +928..; (b) the CALLER's snapshot (ids 0..66, the save-area layout, 944
   bytes) at frame+1024. So the RTL requires a 2 KiB frame (the docs leave context storage to the implementation:
   `isa.md:230-231`). Divergences to record as a Q-entry: QEMU ignores a write to the register slots after entry,
   silicon does not (a supervisor must write only the reply slot +224, which both consume); and action 2's reload
   of a0 from +224 is then the same operation on both.
   **DECISION 1 (semantics owner / lead):** the frame as the context storage with a 2 KiB minimum, or a separate
   save area named by a new CSR. The RTL lane recommends the frame.
4. **Entry (action 0 on a fresh frame) as a switcher request** `is_runv`: SAVE caller ids 0..66 (+FP) to
   frame+1024; LOAD satp/srevroot/options from +0/+8/+56 into the CSR file (srevroot write = the namespace change,
   the trackers forget); LOAD the PCC and x1..x31 from +64.. (16-byte reads with metadata, the tag from memory, as
   the exchange does today), zero and untag the slots; set priv C, mstatus = caller & (SXL|UXL) with FS Initial,
   mie/mideleg/medeleg/mip/trap CSRs 0, fcsr 0, fpr 0; pc = PCC.cursor; csupctl.virt := 1 for the slot (the
   context is virtual by construction: vctx then follows item B's rule); arm the quantum from csupquantum. The
   checks before any write: priv S or M and not C, action <= 3, frame 16-aligned, satp mode 8, srevroot != 0, PCC
   tagged, options <= 3 (the frame-is-RAM check is the PMP/bus error of the first access, as the switcher's today).
5. **Resume (action 0 on a paused kind 1/4/5; action 2 on kind 3)**: SAVE caller to +1024; RESTORE the context from
   the frame's slots (+64.., FP, CSRs); for action 2 reload a0 from +224 and clear that slot, pc += 4; re-arm the
   quantum. The frame's satp/srevroot words are NOT re-read: the CSR words saved at +928.. are (QEMU: "rewriting the
   frame does not replace its roots").
6. **Escape**: commit's escape logic (every exception class, the quantum mark, and a new foreign-return term is
   moot: RETURN is illegal in C) with `is_runv` set: SAVE the context's registers to +64.. (tagged), FP to +656..,
   CSRs to +928..; RESTORE the caller from +1024; then write the event at +16..+47 {kind, cause, pc, addr} and the
   raw a0 at +48 (and the a0..a7 views at +576.. when options != 0), through the switcher's data port as plain
   8-byte writes; x[rd] := kind; csupstatus/cause/epc/tval are ALSO written (the RTL's M-mode readout stays; a
   divergence by addition, harmless to the reference runtime which never reads them). Kind mapping: cause bit 63
   (interrupt or the quantum mark 0x8000_0000_0000_0010) -> 1 paused; causes 12/13/15 with options bit 1 -> 4
   paused; cause 11 with options bit 0 -> 3 paused; cause 30 from CDERIVE/MREV/SPLIT under the flag -> 5 paused
   with pc = that instruction (see 8); anything else -> 2 terminal. tval: the faulting address for 12/13/15 and the
   loads/stores, the instruction bits for an illegal instruction, the capability tval for 24..30 (as the escape
   records today).
7. **Actions 1 and 3**: 1 clears the slot (rd 0; allowed on a terminal slot). 3 on silicon is ILLEGAL (cause 2):
   collection is software on the board (the plan's item H), and the supervisor (or the M-mode monitor by
   trap-and-emulate) performs it: scan the table, untag the dead saved registers in the frame, clear dead tags in
   the page set, rebuild the free list, then strobe the tracker flush (item H: a write of csupctl bit 2 fires
   `revnode_ns_flush`), then action 0. rd on silicon is whatever the emulation returns.
   **DECISION 2 (semantics owner):** action 3 illegal-and-emulated (recommended) versus a hardware collector.
8. **Node pressure (kind 5)**: in `vp_alloc_slot` (item A) compute avail = capacity - next + free_count from the
   header already read and refuse with status 2 BEFORE any write when avail <= 256 (QEMU's reserve) on a RECYCLE
   table (the only kind the RTL accepts); the DYN unit raises cause 30; the escape maps cause 30 under the flag to
   kind 5 paused with pc = the allocating instruction, which action 0 re-executes. This changes item A's exhaust
   arm: capacity 7 / next 6 now reads 30 at the FIRST derive (avail 1 <= 256); the arm's table gets capacity 300 so
   the reading stays "one bump then 30", and a new reading pins the reserve: avail 257 allocates, 256 refuses.
9. **The REVOKE walk bound (the runtime lane's ruling, 2026-10-17)**: under the flag the walk visits at most
   next-unused records (saturating at 65536), refuses with 29 and MUTATES NOTHING on exceeding it, on a cycle back to
   the start id, or on a record it cannot read. The RTL's walk writes as it goes, so under the flag REVOKE_NODE and
   the kill run a read-only pass first (the bound, the cycle test, the depth test) and a commit pass second: twice
   the reads per revoke under the flag, none physically. A read that faults (a page-table or bus error on the
   table) is "a record it cannot read": the rev channel's error response, today unobserved, must map to 29.
10. **CSMINT / CSRETIRE**: DYN ops (funct7 0x50 / 0x53), S or M only, refused in C (the dyn unit sees priv). CSMINT:
    the descriptor's three words are loaded through the LSU at the current translation (S-mode VAs), the checks
    4/6/7/29 in QEMU's order, then two allocations from the table at the CURRENT srevroot (the supervisor's root,
    not a context's) via init_req + mrev_req (the unit's existing pair: a child at depth 0 then the ancestor before
    it with the child re-parented), the 16-byte tagged store to the slot, rd = the ancestor id; free_ids < 2 -> 30
    (avail from the header). CSRETIRE: kill_req-shaped on the id with the bounded read-only pass; rd = 0/1; never
    faults. Both need the lifetime path (`vctx` for the TABLE namespace) to be on for an S-mode supervisor OUTSIDE a
    context: today `vctx = sup_active & ...`. So the namespace flag splits in two: `vctx_xlat` (translation at U:
    sup_active & virt & SV) and `vctx_table` (the table at srevroot: that, OR priv S with srevroot != 0 under
    scapctl). **DECISION 3 (semantics owner):** confirm that S-mode code with a non-zero srevroot uses the table for
    CSMINT/CSRETIRE and for its own lifetime instructions (QEMU: `capstone_store_open` always opens the guest table
    at env->srevroot; the host tree is the physical pool only for the M-mode monitor).
11. **FP state per context**: the switcher gains ids 67..98 (fpr0..31) and 99 (fcsr) for `is_runv` requests only
    (the physical CALL path is unchanged); the FP register file gets a stolen read/write port as the GPR file has
    (issue_read_operands.sv:1611-1660 pattern); the CSR file's "ack anything above 56" becomes "ack 57..66 and 99".
12. **Decode**: funct7 0x24/0x50/0x53 on 0x5b f3 1 are free (decoder.sv:1202-1342); new fu_ops in ariane_pkg.sv and
    capstone_unit.anvilh; DYN dispatch arms; CSRUNV's result pack carries {frame, action} to commit the way
    CSSUPERVISE's carries {seal base, save base}.

## Order of work and gates

1. The switcher's physical bypass under S-mode translation and the `is_runv` SAVE/LOAD/zero sequence with the
   frame's second kilobyte (the largest piece; the switcher is Anvil, so lint by name after every regeneration);
2. the CSR-file slot, the decode, the entry checks, the escape's event write and kind mapping, rd;
3. kinds 3/4 resume (actions 0/2), kind 1 via csupquantum;
4. kind 5 and the walk bound in the unit (item A's territory; the exhaust arm re-pinned);
5. CSMINT/CSRETIRE and the namespace split;
6. FP state;
7. H's flush strobe (csupctl bit 2), E's policy list and the SPLIT/MREV relaxation, Q-15's mask check, Q-18's DROP
   rule (register-only under the flag).
Each piece gets its own directed arm with the readings pre-registered, the flag-off control, lint by name, and the
sweep and sup list at the end; rtl-oracle and claim-auditor on the escape/frame sequencing before the synthesis
request (the frame traffic through the switcher is the plan's named high-risk item). Synthesis risk carried from
step 2: the generated write selector at 31 sites, the 64-bit latches on the D-cache write path, the 56-bit compare
behind the flush pulse, `!vp_hdr_wr` in the invalidation tap; D adds the switcher's new walk and the FP ports.

## Divergences this design accepts (each becomes a Q-entry when it lands)

- context storage in the frame (2 KiB) and the frame's register slots being live after entry;
- csup* CSRs written beside the frame's event;
- action 3 illegal on silicon (software collection, trap-and-emulate);
- the physical page tables of the frame walk: the switcher's accesses are physical and PMP-checked at S;
- the walk bound at 65536 and the 16-bit index (item A), the reserve test on a 30-bit avail;
- CSRETIRE's walk is the kill's shape (the RTL has no separate "retire" op): the operand's own links are nulled as
  QEMU does, the deeper run invalidated, the neighbours spliced.
