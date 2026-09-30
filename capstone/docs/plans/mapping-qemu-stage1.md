# M1: Stage-1 caplified mapping tables in capstone-qemu

Status: IMPLEMENTED ON THE LANE, 2026-09-30: commits 1-5 of §3 are on
`qemu/mapping-stage1` (capstone-qemu, from `c128-qemu-merge` b277660a45) with
their tests on `delegation-memory-qemu` (parent), 45 bare-metal tests passing;
§4 records the regression result. Written as a plan on the same day. It
implements Stage 1 of the [mapping candidate](../design/caplified-mapping-tables.md)
under the [encoding decision](../design/caplified-mapping-encoding-decision.md)
in the emulator only. Nothing here qualifies RTL, and the emulator runs one
vCPU, so the cross-hart clause of the candidate's §8 is not exercised.

## 1. What the emulator does today (survey of 2026-09-30)

Line numbers are from the `qemu/mapping-stage1` worktree at b277660a45 where
marked (w), otherwise from a read of the main checkout at 408fd839 (m).

| Fact | Where |
|---|---|
| Capability mode is `priv == PRV_C && cap_mem`, and `PRV_C` is `PRV_M`; the monitor and the domains share privilege level 3 | cpu_bits.h:622, translate.c:127-129 (m) |
| Every integer load/store and `ldc`/`stc` in capability mode passes `_helper_access_with_cap`, which checks tag, revocation, UNINIT-on-load, 16-byte alignment and bounds, then returns `cursor + imm` as the address the TCG memory op uses; mmu index 3 is an identity map, so the returned value is the physical address and the `cm_map` key | op_helper.c:1579-1695 (w) |
| That helper checks neither permissions ("TODO: bounds check only for now") nor the REV, SEALED or SEALEDRET types; a revocation handle can be dereferenced | op_helper.c:1662 (w) |
| FP loads/stores and LR/SC/AMO use the raw cursor and skip the capability check entirely; instruction fetch has no pc-capability bounds check | trans_rvd.c.inc:50-66, trans_rva.c.inc:26-90, translate.c:1244 (m) |
| Tags and exact bounds live in `cm_map`, keyed by the address the helper returns; the revocation tree and `cm_map` are per vCPU | cap_mem_map.c, cpu.h:391-392 (m) |
| Nodes: `prev, next, depth, valid, linear, refcount`; MREV and SPLIT clone through `_cap_rev_tree_dup_node_before`; REVOKE returns LIN if every revoked node was non-linear, else UNINIT; DROP does not touch the tree | cap_rev_tree.[ch], op_helper.c (m) |
| Sealed regions: the synchronous `call`/`return` swaps only the C-effective set (pc, ctvec, cscratch, mstatus, mideleg, medeleg, mip, mie; 0x58 bytes) and passes GPRs through; the asynchronous path swaps x1..x31 at 0x170 and the CSRs to 0x3B0; nothing but linearity protects the region's contents | capstone_helper.c:131-215 (w), op_helper.c:2843 (w) |
| A synchronous fault inside a domain prints the registers and calls `exit(0)` | cpu_helper.c:1880-1908 (m) |
| Opcode 0x5B, funct3 001, funct7 in use: 0x00-0x0d, 0x20-0x23, 0x40-0x48; free 0x0e-0x1f and 0x24-0x3f | insn32.decode:954-995 (w) |
| `lcc` selectors 0-7 are used (6 is `async`, 7 is `reg`); 8-31 return 0 | op_helper.c (m) |
| Genesis: `cscapenter` mints a third capability ending at `2^63`; `csdebuggencap` mints any bounds, unprivileged | op_helper.c:3021, 3040 (w) |
| No bare-metal Capstone test exists; the submodule's probes run as domains inside the Linux guest through the superproject harness | tests/ (m) |

Three of these are pre-existing gaps the milestone must close because its
security gates depend on them: permissions and handle types on the access path,
and the unchecked FP and atomic paths. Two it records and leaves: the sealed
region's protection and the domain fault exit; both are outside the mapping
contract. The per-vCPU tree and map are irrelevant at one vCPU.

## 2. ISA additions

All new instructions live under opcode 0x5B, funct3 001. Operands beyond the
three register fields come from fixed integer registers, the convention
`cscapenter` already uses for its extra inputs; the RTL cost of that is M4's.

| Instruction | funct7 | Register fields | Fixed operands | Effect |
|---|---|---|---|---|
| `csmapcreate rd, rs1, rs2` | 0x0e | rs1 = root page (LIN physical, W, one aligned 4 KiB page); rs2 = domain handle (SEALED) | a0 = id, a1 = lo, a2 = hi, a3 = max protection (R, RW), a4 = delivery register index 1..31 | Registry entry reserved for id with gen+1; root page zeroed, tags cleared, its capability moved into the entry; the mapping capability (LIN, logical, `[lo, hi)`, rights = max protection, cursor lo) written into the domain's delivery slot; rd = detach handle (REV over the mapping's senior node, node flagged mapping-senior). rs1 consumed |
| `csmappopulate rs1, rs2` | 0x0f | rs1 = detach handle; rs2 = frame (LIN physical, W plus the mapping's rights, one aligned page) | a0 = v (page address inside the mapping); a5 = table page register index or 0 | Frame zeroed and tags cleared, PTE(v) = frame; if the path's leaf table is absent, the register named by a5 supplies one LIN physical W page, zeroed and linked; all or nothing. Inputs consumed |
| `csmapdetach rd, rs1` | 0x10 | rs1 = detach handle | | Every node below the mapping's senior node invalidated; translation cache flushed; entry DETACHED; rd = token (the same REV) |
| `csmapunmap rd, rs1` | 0x11 | rs1 = token | a0 = v | PTE(v) locked; cache flushed; rd = UNINIT over the frame; token unchanged |
| `csmapdestroy rs1` | 0x12 | rs1 = token | | Entry freed for a new generation; token consumed; nothing returned |
| `lcc rd, rs1, 8` | | | | The node's 32-bit binding word, 0 for a physical capability |

Faults use the existing causes: 24 for an untagged operand, 26 for a wrong
capability type or state, 27 for insufficient permission, 5/7 for an access the
mapping does not back, and 25 for a revoked capability. A refused instruction
leaves every operand in place.

**Delivery slot.** The sealed region gains a 24-byte slot at offset 0x3B0: a
16-byte capability and an 8-byte register index. `csmapcreate` writes it and
refuses an occupied slot; `cscall` into that context moves a tagged slot into
the named register after the C-effective swap and clears the slot. The
minimum sealed size becomes 0x3C8. The monitor never holds the mapping
capability.

**Registry and tables.** One global registry of 4096 entries (id 12 bits):
gen, state, class, max protection, lo, hi, root capability with tag. Tables are
two levels of 256 sixteen-byte entries, so one mapping spans at most 256 MiB in
this prototype; the walker is written for a level count so a third level is a
constant change. Entries are stored with the existing `store_cap` path, so
their tags and exact bounds live in `cm_map` like any other stored capability.

**Access path.** In `_helper_access_with_cap`: after the tag and revocation
checks, permissions (new, cause 27) and a type check refusing REV, SEALED and
SEALEDRET (new, cause 26); then, if the cursor lies in the logical region:
binding word nonzero, `registry[id].gen == gen`, entry ACTIVE, root node live,
address inside `[lo, hi)`, walk root then leaf, PTE tagged and live and
physical, PTE rights cover the access, physical address = frame base + page
offset. A small direct-mapped translation cache keyed by (binding, page) is
consulted first and flushed by every REVOKE, UNMAP, DETACH and DESTROY, the
conservative strategy of the candidate's §5.2. FP loads/stores and LR/SC/AMO
are routed through the same helper in capability mode, closing the gap above
for every kind of capability.

**Genesis and minting.** `cscapenter`'s third capability ends at `2^56`;
`csdebuggencap` refuses any bound at or above `2^56`.

**Binding propagation.** Node fields `binding` and `flags`; set by
`csmapcreate`; copied in `_cap_rev_tree_dup_node_before`; zeroed for lone
nodes. REVOKE refuses a node whose flag is set.

## 3. Commits, in order, each building and passing its tests

1. **Harness and pre-existing checks.** `capstone/tests/mapping-qemu/` with a
   bare-metal runner (`-M virt -bios none -kernel`, verdict through the test
   device, trap handler at `ctvec` reporting the cause), an instruction macro
   header, and the first tests: permission fault on a store through an R-only
   capability, cause 26 on a load through a REV handle, and their positive
   controls. QEMU: the permission and type checks in the access helper.
2. **Genesis confinement and binding fields.** `cscapenter`, `csdebuggencap`,
   node fields, `lcc 8`. Tests: mint above `2^56` refused, `lcc 8` reads 0 on
   physical capabilities.
3. **Registry, CREATE, delivery slot.** New `cap_mapping.[ch]`, `csmapcreate`,
   the slot in `cscall`. Tests: CREATE with a monitor destination refused;
   overlap, partition and geometry refusals; delivery lands in the named
   register after `call`; a second CREATE for the id refused.
4. **POPULATE and translation.** `csmappopulate`, the walker, the cache, the
   access-path hook, FP and atomic routing. Tests: store then load through a
   logical capability; supplier bytes and a stored capability are gone after
   POPULATE; a PTE that is none faults; a non-linear or R-only frame refused;
   three scattered pages read as one interval; AMO on logical memory.
5. **DETACH, UNMAP, DESTROY, revocation from above.** Tests: after DETACH every
   logical capability faults with cause 25; UNMAP yields UNINIT that must be
   scrubbed before it reads; UNMAP twice refused; DESTROY without DETACH refused;
   frame revoke from above locks the PTE and returns UNINIT; table page revoke
   makes the subtree fault and the page returns through its handle; DESTROY then
   CREATE for the same id gets a fresh generation; DELIN then DETACH yields the
   token only; REVOKE on the detach handle refused.
6. **Refinement and regression.** The eleven model scenarios that make sense
   at the ISA level rewritten as tests; the superproject smoke and sqlite gates
   on the new binary to catch fallout of the permission and type checks; the
   submodule pin bumped on the parent lane with the survey table updated.

The candidate's §10.2 rows that need two harts (the §8 window, foreign issue
during a barrier) are recorded as not testable here. No number in this plan is
a measurement.

## 4. Result

Commits 1-5 landed as planned; [the harness README](../../tests/mapping-qemu/README.md)
maps every test to its §10.2 row or model scenario. Deviations from §2, all
recorded in the emulator's commit messages: atomics already took the
capability-checked path on `c128-qemu-merge`, so only floating-point accesses
were routed; UNMAP writes a tagged UNINIT marker without a node as the locked
state rather than a bit in the entry; the two-level walker refuses a mapping
above 256 MiB at CREATE. The permission and type checks changed the emulator
for every existing domain, which is what commit 6's regression run on the
prebuilt SQLite memory gate checks; its outcome is in the parent commit that
bumps the submodule pin.
