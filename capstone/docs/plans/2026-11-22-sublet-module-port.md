# The native route on the Sublet format (B5b-1 and later): what the kernel module and the monitor must change

Status: design requirements, 2026-11-22, runtime lane. Nothing here is built. They were settled with the rtl lane
while it staged B5b-1 (the lead's Line B option (C), programme doc). The rtl side is recorded at llvm-capstone
land/rtl-b5b1 (docs/plans/2026-11-22-b5b1-format.md). The reference is capstone-qemu c1985bb529 (x-capstone-ids=63).

## What the format takes away

Under ids=63 the reference makes 14 instructions illegal: TIGHTEN, DELIN, SEAL, MREV, INIT, SPLIT, DROP, REVOKE, CALL,
RETURN, CAPENTER, the CJALR domain switch, CSSUPERVISE and CSRETIRE (trans_capstone.c.inc:25-41). A capability has no
type and no rights, and its cursor is below 2^38. CSMINT stays, in S and M, memory to memory (:568). CSRUNV stays. No
CSCHECK exists in the reference decoder.

## The monitor

1. **It runs in integer M and holds no capability of its own.** It cannot CAPENTER, so there is no genesis and no
   cpmp capability. The M-only debug mints (CAPCREATE, CAPTYPE, CAPBOUND, CAPPERM) are RTL-only and are not used,
   because they would make the twin and the silicon diverge on the monitor's own path.
2. Today's hard-coded END of 2^40 has no Sublet counterpart once CAPENTER is gone, so the redesign drops it rather
   than ports it. The places are region_nend[2] (wrap sbi_capstone_dom.c:35), cap_above (:124-125) and the boot
   SPLITs to cpmp(2) (sbi_capstone_init.S:47-48). The genesis end, if one is ever needed, is 2^38 (ruling Q2).
3. CSMINT is emulated today; under Sublet it is in hardware (B5b-1 step 3), and the emulation goes.
3a. **No capability-CSR access.** Under Sublet capmode is never set: its only setter is a committed CAPENTER, which is
   illegal there. A capability-CSR write requires capmode (csr_regfile.sv:2212: capmode_q && ccsr_en && ccsr_we_i &&
   !privilege_violation), so in integer M a CCSRRW writes nothing. Today's trap entry swaps S's sp through CSCRATCH
   and advances CEPC with CCSRRW (sbi_capstone.S:14, :88-90). The redesign uses the plain mscratch and mepc. The CPMP
   checks and entries are unreachable for the same reason, and the step-2a tie-off guards dead logic (rtl lane, at
   source; board lane's gate: about 21k LUT pruned, a Sublet saving and not a like-for-like comparison).
4. CSCHECK, the checked copy, is emulated today on page capabilities derived from the monitor's genesis, so it cannot
   exist as written. **Open, and the runtime lane's decision at step 3**, with the reference read first:
   - a hardware CSCHECK, added to the reference before the RTL (twin first);
   - or the B5b-2 route: raw tag moves plus a software record walk ("S accesses checked through references").

## The kernel module (capstone_vm)

5. **Retire becomes store, fence, edge.** Today the module ends an arena's lifetime with CSRETIRE on its ancestor
   when the domain unmaps the arena or the context is torn down (retire(), capstone_vm.c:2097-2130). That is the munmap
   half of temporal safety. CSRETIRE is illegal under ids=63, and the module holds no capability to CREVOKE with. So
   the module ends a lifetime itself:
   - an integer store advances the record's generation in the context's table memory;
   - then `fence rw, rw`;
   - then the srevroot edge (csrw 0x5c1) it already makes around every window (R-58).
   The fence is required. The rtl lane read that a CSR write does not drain the store buffer (commit_stage.sv:750-770,
   :836), and the lifetime unit reads records through its own D$ port, outside store forwarding. So without the fence
   the unit's next read can return the old generation.
   The RTL's side of the contract, recorded by the rtl lane:
   - the srevroot edge invalidates every cached record and verdict: the LSU cache and its ALLOW/DEAD verdicts, the PC
     verdict, the record TLB, and step 2's record cache;
   - no table word is kept across operations without invalidation on the edge (today's vp_hdr0_q / vp_alloc_q is the
     counter-example);
   - under CAPSTONE_SUBLET_FMT the CPMP trackers hold no lifetime id, tied off at reset (capstone-ariane b5b1-format
     8cbf393d4), so a kernel store needs no CPMP broadcast.
   Still to check against the reference before step 2's RTL depends on it: that capstone-qemu re-reads the table
   after an S store plus the srevroot edge.
6. **Grant bounds below 2^38.** Under Sv39, TASK_SIZE is exactly 2^38, so access_ok admits a grant ending at 2^38. A
   grant whose cursor is its end, as the stack grant's is, would then carry an unencodable cursor. Today the address
   layout prevents it: dom_reserve maps far below STACK_TOP. In the Sublet build the module refuses it explicitly at
   grant time: cursor < 2^38 and end <= 2^38 (capstone_vm.c:2229-2231's checks).
7. **The CSRUNV actions it uses stay:** 1 (discard), 3, 4 (detach) and 5 | kind << 8 (adopt) (capstone_vm.c:1186,
   1198, 1205, 1229, 1894, 3780), with c5a_model=1 and native_slot_transport=1. The switcher's hidden caller
   snapshot (+1024..+1976) keeps a lossless internal format, not Sublet (ruling (A)), because a kernel-VA caller pc
   cannot be stored as a zero-extended 38-bit cursor. No software reads the snapshot's capability words under c5a.
   The rtl lane's va-runv-snap arm shows an upper-half cursor surviving the round trip, and the zext38 hazard under
   its control.

8. **Record pages are published retired** (the rtl lane's step 2a spec, land/rtl-b5b1 83462497f0ef). Before a record page
   becomes reachable, the module writes every record with word1 = 0xffffffff << 32 and word0 = 0, the first page's
   reserved slots 0 and 1 included. A never-used slot then reads retired (dead) in the hardware and in QEMU alike;
   sublet_alloc already writes word1 = 0 on a fresh slot. Without the rule, a capability that names a never-used slot
   whose page happens to hold a matching generation reads live in the hardware and dead in QEMU. Twin first: the
   reference adopts the same rule, or is shown to read such a slot dead regardless, before the differential relies
   on it.
9. **The 256-slot reserve stays, with the reference's behaviour** (corrected 2026-11-22: the first wording here,
   "a domain CDERIVE fails without a fault at the reserve", is a third behaviour that neither model has; the rtl lane
   read the pin). cap_rev_table_alloc_decision proceeds only while avail > (can_grow ? reserve : 0) (capstone-qemu
   c1985bb529):
   - inside a context, with avail at or under 256 and a growable table, an allocation returns GROW: cause 30, or in
     a virtual-C context the pressure escape (op_helper.c:1286-1300), and the module grows the table;
   - with a table that cannot grow, the reserve does not apply, and an allocation fails without a fault only at
     zero.
   So the reserve keeps the kernel's growth path, and its own mints (CSMINT from step 3), from meeting an empty
   table.
   Item 8's twin-first condition already holds by code: sublet_resolve and sublet_get refuse index >= t->next, so
   QEMU reads a never-used slot dead whatever its page holds. The publish rule makes the hardware agree.

10. **Tables only in RAM.** The module publishes only tables and record pages it allocated from RAM (page_to_phys of
    its own pages). The step-2a unit has no range check (the rtl lane's divergence D3: hardware touches a table that
    QEMU would refuse as outside RAM), so this obligation is what closes it. Divergences D1, D2 and D4 are fail-open
    only on a corrupt or unpublished table, which needs a kernel bug. Each gets a directed arm in the rtl lane.
11. **Encoding:** under Sublet funct7 0x54 decodes as CRENEW (the reference's), so CSNODE no longer exists. The board
    module never issues CSNODE; it appears only in a comment, capstone_vm.c:478.
12. **Record pages stay put during an op** (the rtl lane's step-2a fix F2, an auditor finding). The module never unmaps,
    frees or reuses a record page while a lifetime op can be walking it. Under c5a, a page leaves a context's table
    only between windows, after the srevroot edge. The unit answers 29 for an unreachable record in CREVOKE's commit
    pass, as sublet_reached does, but a page that changes under a walk is a kernel bug the hardware does not fully
    contain (a D3-class row).

## What the census found (2026-11-22)

No software on the native route creates or holds a capability whose cursor is at or above 2^38:
- capabilities are user VAs (grants, checked by access_ok) or physical addresses (the monitor's, below 0xc000_0000);
- kernel VAs appear only as untagged integers (CSCRATCH, CEPC, ecall operands, addresses the monitor walks);
- the only edge case is item 6.

## Order

Nothing here reaches a board before:
- B5b-1 step 3 (hardware CSMINT);
- the checked-copy decision (item 4);
- the module port (items 5-7).
Until then, B5b-1's evidence is synthesis and directed simulation, and its bitstreams carry no silicon key.
