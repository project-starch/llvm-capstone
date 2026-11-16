# Sublet Section 5, B5b: what the collaborator's kernel needs, against what the lead named

Status: scope note, 2026-11-17, rtl lane. No RTL is written for B5b yet. One scope question is open for the project
lead; it is stated at the end with a recommendation, and work continues on the part that is common to both answers.

## The request

The lead (16 Nov, relayed by the runtime lane) asked for the rest of Section 5 in RTL. B5a is the record cache, now
at its synthesis-only gate (capstone-ariane 6e82d7bd2, plan `2026-11-16-sublet-phase2-record-cache.md`). B5b was
named as:
- the 63-bit capability format with a 128-bit record;
- single-operand CREVOKE, and CRENEW;
- deletions: types, rights, SEAL/DELIN, CSRUNV, the supervisor.

The Section 5 plan (spec-silicon `2026-11-15-sublet-section5-rtl.md`) puts these in its Phases 3 and 4. It says
the collaborator's kernel needs both phases, because the kernel checks the 63-bit format at boot and "enters
protected processes by the trap route, not CSRUNV".

## Three sources

- **Paper Section 5:** the specification of record.
- **QEMU `address-space-root`, 4bfbf02fbe..9289114e06:** the reference model. Its header
  `target/riscv/cap_rev_table_abi.h` gives the record, the paged root and the allocation rules bit by bit.
- **The kernel:** llvm-capstone `native-kernel-new-qemu`, `capstone/kernel/patches/` (31 patches in `series`).
  This is the one consumer. It pins capstone-qemu at 9289114e06, the branch tip, so the model and the consumer
  agree by construction.

## What the kernel requires of the processor

Read from the patches. Each line marked (v) was re-read here at source; the rest come from a read-only survey with
quoted evidence.

1. **Format and table.**
   - A capability is generation 127:96, slot 95:65, bounds 64:38, address 37:0 (0011's `sublet-format.h`). The
     address is zero-extended from 38 bits.
   - A record is generation 127:96, previous 95:65, next 64:34, depth 33:3, reserved 2:0.
   - The table is paged 5:9:9:8 from a root page that holds the header and 32 root entries, all physical addresses.
     Its magic is NODPAGE2.
   - The kernel builds the pages and the header and grows the table on cause 30. It reads records but never writes
     them.
2. **Instructions** (opcode 0x5b):
   - LDC, STC, LCC (selectors 1, 3, 4, 8) and CINCOFFSET;
   - CSMINT (funct7 0x50; rs1 is a memory slot, rs2 a descriptor with 0x10f), CDERIVE (0x51) and CREVOKE (0x52, one
     operand: `.insn r 0x5b, 1, 0x52, zero, obj, zero`) (v);
   - CSSHRINK (0x01).
   - **Not emitted:** CRENEW, CSRUNV, and every instruction QEMU deleted.
3. **Raw tag moves** (opcode 0x0b, in 0001): CLDRAW, CSTRAW, and CPTAG (v). CPTAG queries and acknowledges a sticky
   per-physical-page flag, "may hold tags", that "Hardware owns: every capability publication, including a raw copy,
   sets it".
4. **srevroot (0x5C1) as the protection switch** (v):
   - `switch_mm` writes the next mm's table or 0, and every return to U checks it.
   - The design depends on the QEMU rule "U is protected exactly when srevroot is nonzero" (a3006acfa6). In
     protected U, every load, store and fetch through a reference is checked for bounds and lifetime against that
     table.
5. **A tagged sscratch** (v):
   - entry does `csrrw tp, sscratch, tp` and then `csrrw s5, CSR_SCRATCH, x0`, and expects the user's tagged tp back
     with its tag;
   - csrr reads the address untagged, and an integer write clears the tag.
   - The firmware must keep register tags across M traps (0013 Kconfig).
6. **S accesses through references** (QEMU rule 1, 500905eab2): loads, stores, AMOs, LR/SC, LDC and STC in S whose
   base register holds a reference are checked against srevroot's table, using S's SUM translation. The kernel's
   user-memory routines are written for exactly this (0033).
7. **Causes 24-30 delivered to S**, which needs medeleg from the firmware.
8. **Platform:**
   - DT node `capstone,unpinned-v1` with id-bits 63, exact-bounds 0 or 1, page-size 4096 and granule-size 16 (v);
   - one hart;
   - the boot argument `sublet.dom_binfmt=1`.

## What the RTL has today (P1 lineage, 6e82d7bd2)

Re-read at source here:
- **Plain loads and stores are capability-checked only in M** (`load_store_unit.sv:1390-1393`:
  `ld_st_priv_lvl_i == riscv::PRIV_LVL_M`). S and U get the CPMP region check alone (`pmp_data_if.sv:385-390`).
  There is no "protected U" mode and no scapctl.
- A CSRUNV context runs at M with the MMU's permissions forced to U (`csr_regfile.sv:2124`; `cva6_mmu.sv:142-143`).
  That is how the SPEC native route is protected today.
- **sscratch is an integer register** (`csr_regfile.sv:351`, `:1463`). mscratch holds a capability only as half of
  CCSR CSCRATCH.
- **Custom-0 decodes only the debug print** (`decoder.sv:503-518`). CLDRAW, CSTRAW and CPTAG are illegal, and no
  per-page tag summary exists.
- CSSHRINK exists (SHRINK, funct7 0x01). CSMINT is illegal at decode (the monitor emulates it). 0x54 is CSNODE.
- From the census:
  - ids are 55 bits (slot 25, generation 30), with 70 width and position sites across the RTL and Anvil;
  - the record is 137 bits (35 sites);
  - type and permission are read in every FLU and DYN handler (dyn 52 and 10 references, flu 36 and 1).

So **the trap route is an addition, not a deletion**. Protected U, S rule 1 and the tagged sscratch give the
processor a per-access check outside M that it has never had. And CPTAG is a new hardware structure.

## R-75 does not block the trap route

R-75 is the S trap below M after a CSRUNV service resume. The discriminating arms are on capstone-ariane
rtl/record-cache 90e439a7f (tests only), run on 6e82d7bd2's RTL:
- **`va-rc-bare` and `va-rc-bare-nosum` pass.** S holds srevroot = the table it uses (0x80009000) and ecalls with no
  context ever entered, with SUM set and without. `csrr mcause` retires at mtvec.
- **The controls read as filed:** `va-fetchsplit-s` (one entry and one escape) passes, va-rc-trap-z passes, and
  va-rc-trap loops.

The defect needs the service resume and the second escape. A trap-route S, which runs with srevroot non-zero by
design, does not meet it on this RTL. If CSRUNV is deleted, the path R-75 lives on goes with it.

## The scope question for the lead

The list named for B5b deletes CSRUNV and the supervisor, but adds no other way to run a protected process. The
kernel that is to run on B5b enters protected processes only by the trap route, and it also needs the raw tag moves.
So the named list alone produces a bitstream on which neither stack runs:
- today's SPEC native route needs CSRUNV;
- the collaborator's kernel needs items 3-6 above.

The options:
- **(A) Run the collaborator's kernel.** B5b is the named list plus the trap route (items 4-7) plus the raw tag moves
  and CPTAG (item 3). Legacy C mode, the CPMP, CALL/RETURN/CJALR/CAPENTER and CCSRRW are made unreachable rather
  than removed in this bitstream: the kernel never uses them, and removing them is the plan's "biggest risk". This is
  the largest bitstream of the project.
- **(B) Keep the current stack running.** B5b changes the format and the record semantics behind CSRUNV, which stays,
  and the native route is ported to the new instructions (the runtime lane's work). The deletions and the trap route
  move to a later bitstream, together with the kernel.
- **(C) Staged.** B5b-1 is the format, the record and the instructions (CREVOKE single-operand, CSMINT in hardware,
  the NOGROW failure, the release-time generation), with CSRUNV kept. B5b-2 is the trap route, the raw tag moves and
  the deletions.

**Recommendation: (C).** (Premise corrected on 2026-11-17, below in "What proceeds now": B5b-1 likely needs a monitor
port, because the Sublet format has no types. C's remaining advantage is that the format gets its own synthesis gate.)
- B5b-1 is common to all three options, and it carries the format's synthesis risks: the 63-bit id on every compare
  site and the record width. The bounds arithmetic is unchanged; see step 1 below.
- Kept apart from the trap route, its gate isolates them. The SPEC native route keeps a measurable baseline on it.
- B5b-2 adds the trap route on a format that has already passed synthesis.
- It costs one more synthesis cycle than (A).
- If the lead wants the collaborator's kernel on silicon in one step, (A) is the alternative. Its risk is that a
  failed gate cannot say which half caused it.

## What proceeds now (common to A, B and C)

B5b-1's design, in this order. Each step gets a synthesis-only gate before anything is built on it.
1. **The codec, alone:**
   - `compress_bounds`/`decompress_bounds` and the pack/unpack for {generation 32, slot 31, bounds 27, address 38};
   - the cursor zero-extended from 38 bits;
   - the R-60 window limit (`cursor_outside_window`, E < 50) re-derived for a 38-bit address;
   - the genesis range (`flu:504-505`, 2^40) brought under 2^38.
   It goes to a synthesis-only gate by itself. Its key is mapping in P1's class.
   - **The bounds arithmetic does not change.** QEMU's Sublet `cap_compress`/`decode` (`cap_compress.c` at
     9289114e06) use the 27-bit field {iE, t, tE, b, bE} and the renormalisation step of the RTL's
     `compress_bounds` (`ariane_pkg.sv:645-651`, `:817-832`, which says the two encoders "stay bit-identical").
   - What changes is the packing: lo = bounds[25:0] << 38 | address[37:0]; hi = generation << 32 | slot << 1 |
     bounds[26]. The id also widens from 55 to 63 bits at about 70 width and compare sites.
   - So the compress-to-decompress round trips behind the 94 GB mapping precedent are not what is new here. The new
     risk is the 8 extra bits on every id compare (the LSU verdict cache, the CPMP trackers, the PC tracker).
   - A register lane of {tag, generation 32, slot 31, bounds 27} is 91 bits, against today's 90.
   **Step 1's design (2026-11-17):**
   - **Today's split.** The memory granule is hi = `cap_metadata_t` {generation 30 | perm 4 | type 3 | bounds 27}
     and lo = slot[24:0] << 39 | address[38:0] (`ids55_pack_lo`, `ariane_pkg.sv:917-919`). In the core the slot
     rides in the register lane's top bits: lane = {slot 25, tag, metadata 64}, CAP_MD_W = 90.
   - **Sublet keeps the same split.** hi = generation 32 | slot 31 | bounds[26], and lo = bounds[25:0] |
     address[37:0]. In the core:
     - the lane becomes {bounds[25:0], tag, hi 64}, so CAP_MD_W = 91;
     - the cursor word is the zero extension of address[37:0];
     - pack and unpack exchange the lane's top field with lo[63:38], as `ids55_pack_lo`/`ids55_unpack_word` do with
       the slot today.
     So the ~70 lane-slicing sites change a width, not a structure.
   - **The fat capability:**
     - its id becomes {generation 32, slot 31};
     - `decompress_bounds`/`compress_bounds` are unchanged (same 27-bit field);
     - `cursor_outside_window` and the genesis range are bounded by 2^38;
     - a cursor at or above 2^38 is unrepresentable and loses its tag (QEMU `capstone_cursor_representable`).
   - **Types and rights through a decode shim first.** QEMU's own transition did this (4bfbf02fbe, "A decode shim
     makes every decoded value NONLIN with all rights ... so LDC never consumes a slot by a stored type"). B5b-1
     can do the same:
     - every decoded capability reads NONLIN with RWX;
     - type and rights writers become no-ops or illegal;
     - the 52 + 10 DYN and 36 + 1 FLU type and permission references stay as they are and fold to constants in
       synthesis.
     The real deletion moves to B5b-2. This keeps B5b-1's edit to the format and the record, which is the point of
     staging it. Cost: the shim's constant-folding is an inference about synthesis, which the gate reads (own
     footprint by name, and DYN/FLU LUT not above B5a's).
   - **OPEN, and it bears on option C itself.** The Sublet capability has no room for a type: all 128 bits are
     generation, slot, bounds and address. So under B5b-1 every capability is effectively NONLIN with all rights,
     on the monitor's side too.
     - Today's monitor uses types: its CPMP0-2 genesis capabilities are LINEAR (`sbi_capstone_dom.c:8-15`, read in
       R-76's audit), and CSRUNV frames and CALL/RETURN rely on sealed and linear types.
     - So "option C keeps the current stack running" holds only if the monitor and the runtime's CSRUNV route work
       with every capability NONLIN and RWX. Linear moves would become copies, and sealing would be gone.
     - That needs a census of the type and rights uses on the native route (monitor, module, runtime) before B5b-1
       is built. If it fails, C's first step cannot keep the stack, and the choice is between B5b-1 with a ported
       monitor or option A.
     - **First census, the monitor only** (the shared checkout's capstone-sbi, opensbi df0e9a8 of 2026-10-15; not
       established as the board's revision; `sbi_capstone.c` and `sbi_capstone_dom.c`, counts of identifiers):
       - `__linear` 37, `seal` 21, `seal_cap_writer` 8, `CAP_TYPE_UNINIT` 5, `CAP_TYPE_LINEAR` 4, `CAP_TYPE_REV` 3;
       - `seal_region` 2, `seal_region_end` 1, `sealed` 1, `C_INIT` 1, `CAPENTER` 1.
       The monitor depends on types and sealing. **So C's first step most likely cannot keep today's stack without a
       monitor port.** C still isolates the format's synthesis gate, but its "baseline for the native route" argument
       no longer holds as stated.
     - The module's and the runtime's uses are not counted yet.
     - This was not known when the scope question went to the lead, and the lead is told now.
2. **The record and the walker:**
   - 137 bits become 128;
   - liveness becomes "generation matches" (no VALID, FREE or LINEAR);
   - the free list links through next, and the generation advances at release;
   - the walker splits 5:9:9:8 at slot[30:0], and the record TLB and B5a's cache tags widen from slot[24:8].
3. **The instructions:**
   - CREVOKE single-operand, releasing its operand;
   - CSMINT in hardware with descriptor 0x10f;
   - the four-step allocation decision (29, a slot, 30, or an untagged 0) with NOGROW at root+48;
   - retirement at generation 2^32-1.
   - CRENEW is in the named list and in the paper, but the kernel never emits it. It goes last in B5b-1, and is
     dropped if the gate is tight.

## Open questions (lead or collaborator)

- The scope above.
- CPTAG's per-page flag: its size and where it lives (tag memory or a new table). This is needed only for option A
  or C's second step.
- Firmware for the kernel: medeleg for causes 24-30, register tags kept across M traps, and the DT node. Which
  firmware does the collaborator's QEMU test use (`-bios $fw`)?
- U-mode ccsrrw is now R-76 (measured in simulation; first observed on 2026-11-13 on `perf/ccsrrw-priv`). With F-A's
  debug mints, it is refused only under supervision. Under the trap route, protected U is the defended boundary, so
  B5b-2 (or A) must refuse both below M.
