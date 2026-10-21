# ISA and context interface

[Guide](README.md) · Previous: [Runtime](runtime.md) · Next: [Development](development.md)

This chapter describes the experimental interface introduced at
[QEMU revision 9bf9c1f28653][qemu], plus the paged node-store, collection-list and opt-in exact-bounds
extensions and the Sublet lifetime operations below. The superproject QEMU pin selects their
implementation; the older source links describe the original interface.
Numeric allocations and the context-storage
ABI are not frozen. It supplements the [academic-spec amendment][spec-patch];
it does not claim conformance with the deployed physical RTL encoding.

## What the processor adds

The virtual profile combines existing capability operations with:

- virtual address translation under the context's `satp`, using **user PTE
  permissions**, including for instruction fetch;
- a guest-memory lifetime table selected by the context's `srevroot`;
- resumable full-capability contexts, service/fault/quantum events, and
  trusted root minting and retirement;
- physical tag tracking and consuming transfers on the protected paths;
- namespace collection before recycling lifetime IDs.

QEMU internally represents C mode with the M-mode privilege number and
capability memory enabled. A separate virtual-context flag selects the
restrictions above. Applications do not gain ordinary M-mode powers.
The opt-in machine property is `x-capstone-u-mode=true`; its historical name
also covers this virtual C implementation.

## Instructions and CSRs

The following R-type encodings all use opcode `0x5b`, funct3 `1`.

| Instruction | funct7 | Inputs | Result |
|---|---|---|---|
| `CSRUNV rd, rs1, rs2` | `0x24` | Physical frame address, action | Event kind or action result |
| `CSMINT rd, rs1, rs2` | `0x50` | Destination slot VA, descriptor VA (v7: base, end, perms+type, cursor, second slot, 0) | Scalar private ancestor ID; the tagged child (LINEAR or NONLIN, at the cursor) in the slot, and in the second slot when given |
| `CDERIVE rd, rs1, rs2` | `0x51` | Non-linear parent with cursor at child start, scalar size | Fresh non-linear child with inherited data rights and MANAGE |
| `CREVOKE rs1, rs2` (rd=0) | `0x52` | Management-capable non-linear parent, non-linear child reference | Invalidate that direct child and its subtree |
| `CSRETIRE rd, rs1, rs2` | `0x53` | Ancestor ID; conventionally zero in rs2 | Zero on success, one on rejected retirement |

Virtual applications can execute `CDERIVE` and `CREVOKE`. Both decode only in the virtual profile
(`x-capstone-u-mode=true`); the physical profile raises illegal-instruction fault 2 for them, as the
RTL, which does not implement them, would. `CSRUNV` and
`CSRETIRE` require trusted S or scalar M mode and reject C mode. `CSMINT`
rejects U and virtual C; its decoder also accepts legacy physical C when
the extension is enabled. It must remain a trusted root-creation operation.
The current `CSRETIRE` helper ignores rs2.

| CSR | Address | Meaning |
|---|---|---|
| `srevroot` | `0x5c1` | S/M read/write physical lifetime-table base, rounded down to 4 KiB |
| `urevavail` | `0xcc0` | **Illegal instruction by default (Q-21, as on vp2; QEMU from 3b8040d5ff6e).** Under the test-only property `x-capstone-test-csrs` only: read-only available IDs. The adapter computes `capacity − next + free_count` from the table header; the application runtime never reads it (availability inside a context is the kind-5 pressure event) |
| `scapctl` | `0x5c0` | **Illegal instruction by default (Q-22, as on vp2; QEMU from 3b8040d5ff6e).** Under `x-capstone-test-csrs` only: bit 0 selects the protected-U experiment, read-only bit 8 advertises exact shadow bounds. The adapter never touches it and reads exact bounds as 0 |

`scapctl` is not how the launcher enters virtual C. `CSRUNV` binds both roots
when admitting a context, then restores them from saved state on resume.
Virtual code may read `urevavail` but cannot replace the roots. Physical
C/M retains the legacy host-tree backend; the virtual context uses its guest
table. There are no implemented `CSCHECKR`/`CSCHECKW` instructions at this
revision. Linux syscall and VM service numbers belong to software ABIs.

### Minting and retiring an arena

`CSMINT` reads an 8-byte-aligned descriptor of six little-endian u64 values (**descriptor v7, ruled
2026-10-18 by the semantics owner; Q-23**): `base`, exclusive `end`, permissions (`X=1`, `W=2`, `R=4`,
`MANAGE=8`; bit 8 set selects a NONLIN child, clear a LINEAR one), `cursor` (`base <= cursor <= end`,
else 29), a second slot address (0 = none; a non-zero second slot with a LINEAR type is 29, since a
linear capability cannot be in two places), and a reserved word that must be 0 (else 29). The
destination slot, and the second slot when given, must be writable, 16-byte-aligned ordinary-RAM
slots; both receive the SAME capability, one node. Descriptor words and the slots are translated with
the trusted caller's permissions. The three-word descriptor (LINEAR, cursor = base, one slot) is v6
and is no longer accepted.

Why six words: trusted S performs no capability memory access on the RTL (`LDC`/`STC` with an
integer base read 24 in every mode), so the kernel cannot retype, re-cursor or copy a minted
capability itself; the mint produces exactly what the adapter needs, and the adapter's register slot 0
(the context's PC capability, which the switcher loads from frame +64) is filled by the second slot.

Refusal order of record for v7 (QEMU's helper and the monitor's emulation agree): operand
misalignment 4 (descriptor) and 6 (either slot); an unreadable descriptor word 13 (tval the word's
VA, in order); an unwritable slot 15 (tval the slot), the first slot's checks before the second's;
then, with tval 0, 29 for `base >= end`, permission bits outside `0x10F`, a non-zero reserved word,
a cursor outside `[base, end]`, and a second slot with a LINEAR type; then the representability
refusal 29; then fewer than two free IDs, 30. The slot's shape is checked before the descriptor's
meaning, so a misaligned second slot with a LINEAR type reads 6.

It requires `base < end`, permissions at most 15, two available IDs, and an
exact compress/decompress round trip of the initial bounds and cursor in the
default profile. With `x-capstone-exact-bounds=true`, the existing physical
shadow bounds are authoritative, and the round-trip restriction is lifted.
Context restoration also loads those shadow bounds. This opt-in QEMU profile
supports mallocng object geometry without allocator-specific instructions;
it requires more metadata than a tag bit and is not an RTL encoding.
All checks precede table and slot mutation. Success creates a private senior
ancestor and a child of the descriptor's type (LINEAR by default, NONLIN with bit 8). The scalar ancestor lets the adapter later
retire every application derivation under that arena without keeping a REV
capability in ordinary kernel C state.

`CSRETIRE` checks and walks the descendants, invalidates and unlinks them,
then invalidates and unlinks the ancestor. It does not unmap pages, erase
payloads or immediately recycle IDs. Those are ordered adapter duties.

### Sublet child lifetimes

`CDERIVE` requires a tagged, live NONLIN parent and an untagged, positive
size. The range `[parent.cursor, parent.cursor + size)` must lie inside
the parent's bounds; subtraction checks the range without overflow. The
compact profile also requires an exact bounds round trip. The exact-bounds
profile retains the requested bounds in the existing shadow metadata.
The operation creates a non-linear child node immediately after its parent
in preorder and returns its capability. Data rights are inherited; MANAGE
is granted for the new node even when the parent lacks it. Sources are
unchanged except when explicitly selected as rd; rd=0 discards the result
but still allocates. Allocation uses the existing pressure/collection path.

`CREVOKE` checks both tags, that both operands are NONLIN references (cause 26
otherwise), both live nodes, MANAGE on rs1, and direct parentage of rs2's node;
the node table also refuses a linear node as the child. It then invalidates and
unlinks that child and its descendants. The type checks matter because `CSMINT`
can mint MANAGE onto a linear grant: without them, an `MREV` handle over that
grant could end the linear node behind it, and legacy `CSREVOKE` of the handle
would then find an empty subtree and return LIN instead of UNINIT, skipping the
re-initialization rule.
No payload bytes are accessed or initialized. Narrowed bounds, offsets and
data rights on either reference do not affect release authority. Foreign
children, grandchildren and self-revocation fault with cause 29; missing
MANAGE faults with 27; stale operands fault with 25. All checks precede
mutation. Nonzero rd is reserved and causes illegal-instruction fault 2.
This two-operand operation is distinct from legacy one-handle `CSREVOKE`
(funct7=0), which retains its reclamation semantics.

`CSTIGHTEN` accepts four rights bits and cannot add any of them. Non-linear
`CSSPLIT` keeps the original node in both resulting references; allocating
a sibling here would let a client escape its original lifetime. Linear
splits still create siblings. `CSMREV` records the new parent and increases
all existing subtree depths. In the virtual profile, non-linear stale references
retain their tags and IDs when loaded from live storage; dereferencing them faults
with cause 25 where it used to fault with 24 (an untagged reload). Collection still
clears their tags before ID reuse: the adapter's sweep over registers, saved
contexts and memory. The physical profile is unchanged: a revoked capability of
any type reloads untagged (24). That differs from the RTL, whose `LDC` forwards
the loaded value verbatim for every type (`docs/ref/ISSUES.md` Q-11), and is left
for Q-11's own fix so that recorded physical results stay reproducible.

**Authority model.** MANAGE is the right to end direct children. Creating a child
is unprivileged: `CDERIVE` needs no MANAGE, so any non-linear reference can create
lifetimes and consume nodes, and `capstone_cap_without_manage` stops a client from
ending lifetimes but not from creating them. The right to end a child does not
depend on the parent reference's bounds, and MANAGE survives `CSSHRINK`,
`CSCINCOFFSET` and non-linear `CSSPLIT`. An allocator must therefore keep this
invariant: **every pointer that leaves the allocator is a `CDERIVE` child, or has
had MANAGE removed.** A narrowed or offset view of a pool that escapes with MANAGE
can end every live object in that pool.

**Overlap.** Live children may overlap: two `CDERIVE`s over the same bytes both
succeed. Linear carving made disjointness hold by construction; with `CDERIVE`
it is an allocator duty, and an allocator bug that issues a chunk twice is
invisible to the ISA.

**Representability (compact profile).** `CDERIVE` refuses bounds that do not
round-trip exactly (cause 29), and there is no instruction to ask what is
representable. With the 27-bit bounds field, a length L of 4 KiB or more needs
base and end aligned to 2^(msb(L)-9), applied to the length after that rounding:
8 bytes for 4 KiB, 128 bytes for 64 KiB, 2 KiB for 1 MiB, while a length just
below a power of two, such as 8191, rounds up to it and takes its alignment (16).
Shorter lengths are exact. Allocators with large, oddly aligned objects must round.
The exact-bounds profile has no such restriction.

**SHRINK's result on the compact profile (ruled 2026-10-20, semantics owner).** CDERIVE refuses an inexact child; SHRINK does not. vp2's SHRINK checks only causes 24, 26 and 29 (`capstone_flu_unit.anvil` SHRINK, 8522d72fe:254-290), and its result passes through the register file's compressed form. So silicon holds the requested bounds rounded OUTWARD (base down, end up, at the alignment above), while QEMU's `helper_csshrink` stores them exactly and keeps them until an escape compresses the register. The compiler's global-capability initialiser forms every global as `shrink(scc(gp, &G), sizeof G)`, so the difference is real in every image: in one cpython session `_PyRuntime` (0x80f00 bytes) rounds at 1 KiB on silicon, so its capability starts up to 0x3f0 bytes before the object (compiler lane, 2026-10-20). Rules: (1) QEMU's compact profile rounds the result of SHRINK, and of every other bounds-forming op that does not refuse, through the same compress and decompress at writeback, with the RTL's compress_bounds normative and an enumeration cross-check. The prediction kits then model what silicon holds, and QEMU's bounds no longer depend on when the first escape lands. SCOPE (2026-10-20): every bounds-forming result is rounded EXCEPT those formed by the QEMU kit's own M-mode monitor outside any context. That monitor is a model artifact, which relies on exact physical bounds and fails its first domain return when rounded, and the board runs a different monitor. The exemption is a known model deviation. It is named by what it exempts and never by privilege, because a context runs at raw M. Measured (compiler lane, capstone-qemu compact-round-results-v7, binary 2c6d3f24): a context SHRINK of a global to [+0x10, +0x5013) reads [+0, +0x5020), as silicon holds it (exact on the unrounded batch binary 9818c197), and Tier 1 reads PASS 30/45 with the same 15 FALSE and the same 24 exits as without rounding. The enumeration against both trees' compress/decompress reads 0 mismatches over 210,946 regions. (2) The compiler pads and aligns each global to representable bounds, as Q-15's tail pad does for the heap, so that both sides hold the object's exact bounds. Outward rounding is an AUTHORITY GAIN over neighbouring objects, not only a detection difference. (3) Until (2) is in an image, a global overrun inside the rounding slack may read MISSED on vp2 where QEMU reads CAUGHT, and such a difference is this deviation (the global analogue of R-60's rule (2)). (4) The intended ISA rule is CDERIVE's: SHRINK refuses bounds that do not round-trip exactly (29). The RTL adopts it only after (2) is measured to leave no inexact SHRINK in the corpus images, because before that it would fault every image at initialisation.

The C interface is [capability.h](../../../runtime/include/capstone/capability.h):
`capstone_cap_derive(parent, offset, size)` interprets the offset relative
to the parent's base and rejects offset overflow before setting the cursor;
`capstone_cap_revoke_child(parent, child)` releases the child; and
`capstone_cap_without_manage(cap)` removes only MANAGE. These wrappers use
raw `.insn` encodings and require no compiler backend, mnemonic or intrinsic
addition. `capstone_cap_derive` and `capstone_cap_without_manage` check that
their operand is a NONLIN reference first, and refuse anything else atomically
with a zero-size `CDERIVE` (cause 26, or 24 if untagged), because the `CSSCC` and
`CSTIGHTEN` they use would otherwise move a linear operand out of its register.
Allocator bookkeeping must still prevent overlapping live children and protect
management-capable copies. This ISA change does not migrate the allocator
adapters or qualify RTL.

**Open items** (deferred design choices, recorded at review):
- whether ending a child should depend on the parent reference's bounds (CREVOKE
  requiring the child to lie within them, which needs the child node's own extent),
  or narrowing should drop MANAGE;
- `CDERIVE` with rd=0 still allocates a node nothing can name; reserving rd=0 as
  illegal, as `CREVOKE` reserves rd≠0, would remove it;
- NOT_CHILD and a corrupt node table both raise cause 29 with tval 0, so a
  program defect cannot be told from an infrastructure fault; a distinguishing
  tval would separate them;
- a representability query for the compact profile (see above);
- one mnemonic everywhere: the QEMU decoder says `cskill`, and `CREVOKE` differs
  by one letter from legacy `CSREVOKE`;
- assembler and disassembler support (`llvm-objdump` shows both as `<unknown>`).
- a guest-level test that a collected and reused ID cannot be reached through a
  stale reference reloaded from memory; the model exercises the table, not the
  adapter's sweep, and the bare-metal gate has no collection.

## Saved context protocol

The trusted caller passes a pinned physical RAM frame, aligned to 16 bytes.
The main runtime frame is kernel-owned; child startup frames may reside in
registered user mappings under the [thread protocol](runtime.md#threads-share-memory-and-lifetimes).
Their roots are supplied by the module. These input frames are distinct from
QEMU's internally saved continuations.

| Byte offset | Size | Field |
|---|---|---|
| 0, 8 | 8 each | Initial Sv39 `satp`, nonzero `srevroot` |
| 16, 24, 32, 40 | 8 each | Event kind, cause, PC, fault address/data |
| 48 | 8 | Legacy raw low-word a0 view |
| 56 | 8 | Options: bit 0 resumable ECALL, bit 1 resumable page faults |
| 64 | 16 | Initial tagged PCC |
| `64 + 16*i`, i=1..31 | 16 each | Initial full-width x1..x31 |
| 576..639 | 64 total | Scalar cursor/value views of a0..a7 |
| 640, 648 | 8 each | Collection page count, physical page-list address |

**The register slots after an escape (ruled 2026-10-18, Q-30; the RTL is the reference).** On every escape the
context's x1..x31 are written WITH their tags into the frame's own register slots (+64 + 16·i, the PCC at +64) and
the caller is restored over the register file; the resume loads the slots back (the pc from +64, +4 after a reply),
and zeroes only +224 (x10's slot) after a reply (action 2); a non-reply resume zeroes nothing, and the restore read
does not consume, so after every resume the frame keeps tagged copies of x1..x31, LINEAR ones included, beside the
live registers until a later ENTRY zeroes them (a linearity residual on silicon, recorded with the S-entry below;
the RTL item for the next bitstream is a consuming resume). The trusted adapter therefore treats the register slots as the context's live
state and OVERWRITES +224 with the reply; a check that the slot is empty is QEMU-shaped and wrong on silicon (vp2's
boot 3-T1 repeat stopped on it). QEMU's escape, which kept the callee in its own save state and left the slots
zero, matches the RTL from b290f5656600 (every kind writes the slots before the caller is restored; the resume
loads them; the hidden words +920..+944 and the FP area stay QEMU-internal, a recorded residual); the +576..+639
**An out-of-window cursor on silicon (R-60, characterised on vp2 2026-10-20; ruled by the semantics owner).** The
ISA rule is QEMU's: a load or store through a capability whose access lies outside its bounds traps (cause 28), whatever
the cursor. vp2 deviates: CINCOFFSET neither clears the tag nor checks representability, the register file holds the
compressed form, and the LSU checks an access against the bounds decoded AT the live cursor, which for a cursor k window
spans away (S = 2^(E+14)) are the true bounds shifted by kS. An access passes iff (cursor - base) mod S <= len - size,
touching memory kS away without a trap (silicon: a load read and a store overwrote a byte 64 KiB away). Rules that
follow until the RTL is fixed: (1) generated and library code must never DEREFERENCE a capability whose cursor lies
outside its representable window (an in-register transit that returns before any access, a bounds-deriving op, or a
compressed save is a separate matter: the first is harmless on silicon, the second refuses with 29, the third loses
bounds on QEMU's compact profile); (2) a corpus cell whose defect overruns far past its object may read MISSED on vp2
where QEMU reads CAUGHT, and such a difference is this deviation, not a model error; (3) the RTL fix belongs with the
ISA's intent, rather than a widened LSU check. RULED 2026-10-20 (semantics owner): CINCOFFSET and SCC CLEAR THE TAG
of their result when its cursor is not representable, and do not trap; the fault then arises at any later use as a
capability (an untagged operand), never at the arithmetic. Representable is defined by the decode itself: the result's
cursor c is representable iff decompress_bounds(the source's compressed bounds, c) returns exactly the source's
[start, end) (ariane_pkg.sv's decompress, the function the LSU checks against, is normative; QEMU's compact profile must
implement the same predicate on the same inputs, and both are cross-checked by enumeration at the window edges). Why
clear rather than refuse: pointer arithmetic that is never dereferenced (a loop's end pointer, a hoisted offset) stays
total, as in CHERI, while every dereference of such a pointer faults; refusing would trap code that computes but never
uses an out-of-window value. Consequence: a cursor that leaves its window and comes back has LOST its tag, so a transit
out of the window and back (undefined in C beyond one past the end, and the pattern removed from the runtime at
232d13b220cc) is no longer tolerated; the compiler must not create one (rule 1 is therefore strengthened, once this
fix is in silicon, from "never dereference" to "never leave the window"). **Refined 2026-10-20 (semantics owner), the reference bounds:** "the source's [start, end)" means the bounds the source's compressed form decodes to at the source's OWN cursor, i.e. decompress_bounds(compressed, c_old), the bounds silicon actually holds, and the tag survives iff decompress_bounds(compressed, c_new) equals that. On silicon this is the rule as first written, since the register file holds only the compressed form. It matters on QEMU's compact profile, where a register can carry fat bounds that no cursor decodes back to (an INEXACT source: the compiler lane's window counter read escape_inexact 44,312 in one cpython session, register samples at the context's escapes; CORRECTED the same day, the session's 32 inexact bounds-deriving ops were all the QEMU kit's M-mode monitor at boot, not the domain, which made none, so where the domain's inexact capabilities come from is open). Compared against those fat bounds, the predicate would fail even at the source's own cursor and untag every move of such a capability, +0 included, which is not R-60's subject. QEMU computes the compressed form with the same cap_compress its escape save uses. The inexact fat bounds are a separate QEMU-vs-silicon divergence: silicon cannot hold them, because bounds are rounded or refused when the capability is formed. It stays counted on its own (op_inexact, escape_inexact) and is not folded into the tag rule. **The op set and where the predicate is evaluated (ruled 2026-10-20, semantics owner, on the RTL lane's enumeration).** The rule covers every op that writes an arbitrary cursor onto a tagged capability WITHOUT changing its bounds: CINCOFFSET, CINCOFFSETIMM (the same instruction's immediate form, and the one the compiler prefers, so leaving it out would keep the defect reachable), SCC (an absolute set) and INIT. **CORRECTED 2026-10-21 (an adversarial audit of the RTL lane's item 4, verified at c7e6f604f): the first version listed three ops and said INIT's cursor was in-window by construction. That is false:** INIT keeps rs1's bounds (modify_metadata_type copies start and end) and sets the cursor to rs2 + start with rs2 an unconstrained register (capstone_flu_unit.anvil, the INIT body), so without the check it yields a TAGGED LINEAR capability at any cursor, and R-60's alias survives through INIT, STC, LDC and a load a span away. SHRINK and CAPBOUND change the bounds, so the round trip is not the right test for them and they are excluded. CCSRRW is moot (its cap_rd is forced to zero). The predicate does not belong in the shared helper. **SEALED and SEALEDRET are excluded from it:** a sealed capability holds no bounds decode in a register (ariane_pkg.sv: bound_start = cursor, bound_end = 0, the field carrying reg_id/async), so the round trip would fire on non-bounds and destroy a correct capability's tag. Those types are refused by the ops or carry no window. The measured population below (LINEAR and NONLIN decodes) could not have caught that; it excluded the class that breaks it. The RTL may evaluate it at the pack (compress_cap), because a register's fat bounds are themselves a decode, either exact (lengths below 4 KiB) or granule-aligned, so compress(decompress(B, c)) == B there. This was measured on the lane's bounds model over 200,000 regions: 0 idempotence failures, 0 fires with the cursor unchanged, and 0 disagreements with the two-cursor definition over 106,115 moves, with a fired control (bounds perturbed off the granule change the word 100,000 of 100,000 times). **Never evaluate it against EXACT requested bounds:** compress_bounds rounds the base down and the top up, so a round-trip test on exact bounds would clear the tag of every inexact capability (191,953 of 200,000 in the same model). The predicate is correct only on decoded bounds, and a later "simplification" to the requested bounds would silently break every inexact capability. QEMU's compact profile covers the same three ops. **What a cleared result holds (ruled 2026-10-20):** the computed cursor as an integer and a ZERO metadata half, exactly what CINCOFFSET's own untagged path already builds (`metadata = cnull.metadata; cursor = new cursor`), and never the capability's old bounds. The RTL's compress_cap returns bound_start verbatim for a NOT_CAP type (the S-06 raw-data stash), so the clear must not be done by retyping the fat capability before the pack: the stale bound would ride in the metadata half and be byte-copied onward as data. QEMU's untagged result carries the same zero upper half (its scalar_hi), so a differential agrees on both halves. A cleared result is indistinguishable from any other untagged value: every query (LCC's type query included) reads it as plain NOT_CAP, with no marker of how it lost its tag.

**CPMP entries are physical (ruled 2026-10-20, semantics owner; S-23, R-60's batch item 1).** A CPMP entry holds only
a capability of the PHYSICAL namespace. Every supported producer is the monitor's M-mode region path (swap_cpmp, the
boot carves, region grants), which runs with csupctl.table clear; a context's virtual nodes live in its registers, frame
and memory, the CPMP is INIT-zeroed inside a context, and the CSMINT emulation writes its re-noded capability into a frame
slot, never a CPMP entry. Consequences: (1) the monitor must never install into a CPMP entry a capability obtained from a
context, a frame or any table-namespace operation (a monitor bug if it does; the capability word has no bit by which the
hardware could check it). **"Physical" here means M-MODE-MINTED, not "minted by whoever is not a context":** an S caller with a non-zero `srevroot` is ITSELF inside the table namespace by `vctx_table_o`'s second clause (`csr_regfile.sv:2885-2887`), so it can derive virtual nodes without being a context, and a capability it hands to M is exactly the case consequence (1) forbids. The RTL lane's measured arm has the caller in that state throughout, which is why its tracked id must be described as M-mode-minted rather than as the caller's; (2) a revocation in a context's lifetime namespace can never legitimately invalidate a CPMP
entry, so the RTL fix for S-23 is an invalidation broadcast that carries one namespace bit, with the CPMP trackers acting
only on physical broadcasts while the LSU revocation cache and the PC tracker consume both; (3) one bit suffices only
because a change of table base flushes every tracker (ex_stage.sv revnode_ns_flush), so that flush must not be narrowed.
Until that fix, the module keeps a context's virtual ids above the physical ones (the node floor, dev 794d7c5815f5). **Companion monitor invariant (ruled 2026-10-20, with batch item 1):** the M-mode monitor must never issue a PHYSICAL-namespace revocation (kill, retire, REVOKE of an M-minted node) while csupctl.table is set. The broadcast's namespace bit is the rev-node unit's held copy of the table-namespace flag, sampled at the handler's accept and frozen until the handler is idle (ex_stage.sv: vctx_rev_q resamples only when !rev_busy, and the broadcast's own write cycle is a busy cycle), so it cannot change mid-walk; but it is only as right as the flag at accept. Outside a context the only clause that reads table while issuing from M is `M && csupctl.table`, so a physical revoke issued inside a csupctl.table bracket would be stamped virtual, and the CPMP entry it should invalidate would keep stale authority. So the monitor brackets csupctl.table tightly around its own table operations (the CSMINT emulation) and issues no physical revocation inside the bracket. This is a software obligation, not checked by the hardware, and it is the same shape as "never install a context's capability into a CPMP entry". VERIFIED for capstone-sbi f13dbc5 with the board wrapper aab63d5 (board lane, 2026-10-20): csupctl.table is set only at capstone_csmint.S:279 and cleared at :281, or as csm_window_trap's first instruction (:310); the window holds one CSNODE (:280), with MIE 0 by trap entry. The batch firmware's pre-registration re-runs the same check on its own commit. **And the namespace flush must no longer touch the CPMP trackers (ruled 2026-10-20, from an independent audit of item 1).** pmp_data_if's `if (revnode_ns_flush_i) { tracked id := 0; valid := 1 }` (present since vp2 item A step 2) fires on every change of the held flag or srevroot, i.e. at every context entry and escape, and the next cycle's adopt re-arms the entry as valid. Before item 1 that re-arm was what healed S-23's false denies. Once CPMP trackers act only on physical broadcasts there is no false deny left to heal, and the same re-arm UNDOES a real physical revocation: a node the monitor revoked, tracked by a CPMP entry, would be trusted again after the next context switch. So, in the same bitstream as item 1, the CPMP trackers ignore the namespace flush: a tracked id and its validity change only by adopt (an entry rewritten with a different id) and by a physical broadcast. The PC tracker's and the LSU revocation cache's handling of the flush is checked for the same resurrection, and each must not report a node valid that a broadcast has cleared, other than through a re-check of the record. Checked by the RTL lane at 7784b37ab: the LSU revocation cache is safe (a flush empties it and gates any new ALLOW, so a flush can only force a re-probe); the PC tracker has the SAME defect (commit_stage.sv: the flush zeroes the tracked id and sets valid, which also suppresses the cause-25 trap in that cycle, and srevroot changes raise the flush with no switch at all, e.g. a plain S-mode write), so it is ruled the same way: the PC tracker ignores the namespace flush entirely, with its validity changed only by adopt (a new PCC id) and by a broadcast. Wording note: a classic supervised domain, and U with srevroot non-zero, read as physical to the bit, which is harmless to the CPMP because their writes go to the physical pool as well.

**Generations in virtual node ids (the lead's decision 2026-10-20: IN the next bitstream as batch item 6, co-designed with item 1; semantics owner's ruling, REVISED the same day on the RTL lane's source read).** Until now a virtual id has been the bare index (generation 0), and the kill's free-list push is gated off under the virtual-context flag. Records are therefore never recycled in a context, and a table exhausts (the module's -ENOMEM). cpython lt1 is the first corpus arm that needs recycling, and the 16-bit index rules out a larger table. Rules: (1) a virtual id is {generation[29:16], index[15:0]}, the packing of a physical id. A record stores its generation in word 1 bits [59:46]. **CORRECTED 2026-10-21 (the RTL lane's source read):** "no earlier text did" was wrong. The shared ABI header cap_rev_table_abi.h ("shared with the trusted OS adapter") assigns word 1 as depth[29:0] and parent[59:30], and QEMU packs and unpacks it so; the RTL writes only parent[15:0] and depth[16:0], which is why [59:46] looked free from the hardware's side. RULED (options (i) and (ii) of the RTL lane's four): in the FLAT table format, the one the virtual profile uses on silicon and in every prediction kit, capped at 65,536 records, a cap the module, the RTL's 16-bit index and CSRETIRE already impose, word 1 is {reserved[63], FREE/LINEAR/VALID[62:60], generation[59:46], parent[45:30] (16 bits), depth[29:0]}, and the ABI header gains that flat-format layout beside the paged one. The PAGED format (QEMU-only) keeps parent[59:30] and its tag-sweep collection, and needs no generation. A FREE record is "FREE set, the generation field preserved, every other field zero" on BOTH sides, replacing the exact `FREE << 60` shape (QEMU cap_rev_table.c's push at :277 and its free-list validation at :100; the RTL's vp_rec_free_exact), because the pop reads the generation out of the free record. The module (the trusted adapter) parses no record's parent or depth, so the narrowing touches only QEMU and the RTL. Two constraints on the push, found implementing it (RTL lane, 2026-10-21), which bind QEMU equally: (a) the push and its header publish happen only in a RECYCLE table, since slot 1 is an ordinary record otherwise; (b) the push loads THIS table's free-list header before linking, because the unit's cached free head and count belong to whichever table last allocated, and linking to them would splice another context's index into this table's free list. The recycling must also cover a killed child's DESCENDANTS, not only the child, because the runtime heap and the cpython and postgres Sublet patches free only through CREVOKE, and their CREVOKEs end subtrees. (2) Recycling is in HARDWARE, mirroring the physical path: under the flag the kill (and a retire) pushes the record onto the TABLE's free list (slot 1, vp_write_hdr1), and the virtual pop hands out {generation + 1, index} and rewrites the record with it. A record at generation 16383 is not pushed (the physical push's guard, mirrored). The fresh path hands out generation 0. Software never re-lists records and sweeps no tags. A bitstream without this item simply never recycles, so it fails closed by construction and no module probe is needed. (3) LOOKUPS compare the FULL id: every cache or tracker that answers "is this capability's node valid" from a stored entry (the LSU revocation cache, the PC tracker) must tag the entry with all 30 bits, or a stale {g, i} copy would hit an entry made for {g + 1, i}. INVALIDATIONS may stay index-only (a broadcast masked to the index clears every generation of that index, which is conservative), and the record's own validity check is already generation-aware. Under item 1 the CPMP trackers act only on physical broadcasts and leave this question. (4) The record layout is a module/model contract (the module's table code, QEMU's cap_rev_table.c), not the image ABI. Domain images do not read records, so no image is rebuilt for it unless someone finds an image-side reader. (3a) COMPOSITION sites, which the lookup/invalidation split does not cover (independent RTL audit, 2026-10-20, each read at c7e6f604f): every place that FORMS an id or WRITES a record's generation must carry the real generation, and every admission gate must accept an id whose generation is at least 1. Known instances: the virtual record-read tap composes {14'd0, index} (ex_stage.sv:1729), which would make the LSU's 30-bit probe resolution (load_store_unit.sv :1433) deny every recycled node for good; CSNODE's create writes both new records with generation 14'd0 (capstone_rev_node.anvil create_req) instead of the generation of the id the pop returned; and CSRETIRE refuses any id at or above 65536 before its request (capstone_dyn_unit.anvil CSRETIRE), which every id of generation 1 or more is, while its record check compares no generation. CSRETIRE of a stale {g, i} must answer 1 and change nothing. (3b) CSNODE (RTL funct7 0x54, the primitive behind the monitor's CSMINT emulation) is admitted at S whenever the table-namespace flag is set and no supervised context runs (decoder.sv, the same predicate as CSRETIRE; measured executing at S by the RTL lane's va-srevroot -DSREV_CSNODE arm), and its create never reads rs1's prior node, so it re-nodes any capability it is handed, a revoked one included. RULED by design, not a defect, under the trusted-S model: S already holds CSMINT, which mints ANY capability from a descriptor onto fresh nodes, so re-noding grants S nothing CSMINT does not, and trusted S holds no capability in a register in any supported flow (Q-23). Revocation protects against U contexts, never against the trusted kernel. What would make it a defect, and so must stay true: U and a running supervised context can never issue it (the decoder excludes both), and no flow hands S a context's capability in a register. (5) QEMU's virtual profile implements (1)-(3) identically, the push at kill included, and the flat virtual build stops using CSRUNV action 3's page-sweep collect. (6) The rule this REPLACES, from the first version of this ruling (02c84b8b7a6a): software re-listing by the module plus a fail-closed load-time probe. It is withdrawn, because the hardware push already exists on the physical path and a software path would be unsafe on every bitstream without the item.

**The frame is never domain-reachable (S-21/S-22, settled 2026-10-18 from the module's source).** The main runtime frame is a kernel page outside the process's address space (`capstone_vm.c:1756`, `:1821-1822`; the launcher grants nothing on it), and a child startup frame, which the domain's runtime acquires as a one-page heap grant, is pinned and its arena's ancestor CSRETIREd by CV_THREAD_CREATE before the first entry (`:1520-1559`; the create fails if the retire fails; the arena must be exactly that page, `:351-358`; a later overlapping grant is refused, `:772-775`), so after the create the domain reaches 0 bytes of the frame; the caller's snapshot is written at entry, after the retire. The retire is the control, and it covers the satp and srevroot words at +0/+8 and the hidden words at +928..+944 as well as the snapshot at +1024, which a narrowed grant would not; a ruling of the same day that narrowed the grant to [frame, frame + 1024) was withdrawn as unnecessary and weaker. A kernel-only save base in a CSR (the snapshot addressed from a base the trusted side writes before entry) is a design note for the day the ABI maps a frame into a domain, not a bitstream item. The register slots at +64..+576 are the context's own saved state (the escape writes x1..x31 there, the resume loads them) and stay frame-addressed, so no software clear has a window; on bf3db3175 the restore does not consume them (the linearity residual above). Design of record for the next bitstream: a CONSUMING resume in hardware, reusing the entry's zero pass over the slots, flat unless a tagged-only clear is measurably cheaper, after which a tagged copy never outlives the resume that loaded it. **Address-space installs and the TLB (ruled 2026-10-18).** On bf3db3175 the switcher installs satp on entry, on resume and on the escape's restore of the caller without a TLB flush. The RESUME direction is an ABI rule, because the trusted side runs before it issues the CSRUNV and can fence for itself: after a resolved fault it issues `sfence.vma` on the faulting address, on the resuming hart, immediately before the resume (the kernel's own fault paths flush only on the hart that changed the entry, and a repeat fault on an unchanged entry flushes nothing); a trusted caller that changes a mapping a paused context will use fences before resuming it. DEMONSTRATED on bf3db3175 (S-29, va-satpstale): a virtual context under its own root reads the caller's physical page through a stale data translation (the escape walk's own trace shows the live satp as the context's root while the read is served from the caller's chain), and after an escape the caller's first data access reads the context's page, one retired instruction away from the control that reads its own. For DATA accesses the hole is an ABI rule in BOTH directions, because every escape returns to CSRUNV+4 whatever its kind (the walk restores ids 0..66 from the save area with no kind-dependent branch; observed for the ecall kind, static by design for the others): the trusted side fences once before the entry and once at CSRUNV+4, and a context switch is the time to assume nothing about the TLB. The surface no fence covers is the caller's first instruction FETCH after the escape: the fence at CSRUNV+4 is itself fetched under the caller's satp through an instruction TLB the context's fetches populated under its own root (cva6.sv:1910/:1912 OR the context flag into both the fetch and the load/store translation enables), which is S-23's recorded fetch shape, now DEMONSTRATED in simulation on bf3db3175 (satpu2): without a post-escape fence the caller's first U-mode fetch takes cause 1 (instruction access fault) with mtval the PHYSICAL page the context's chain maps that address to, one byte past the caller's only capability region, and with one fence it takes cause 12 with the virtual address. A stale instruction translation therefore surfaces as an access fault naming a physical address the code never used, indistinguishable to the caller from a capability violation; translations do not survive a context switch and no caller may rely on them, which is what the hardware flush-all is justified by. S-23's DATA face (a kernel access at S refused at an address inside a loaded region) has no mechanism established; an index-only CPMP invalidation in the R-38/R-44 family is a hypothesis. QEMU is the stricter side: restore_state flushes the soft TLB whenever satp, offsetmmu or the mstatus translation bits differ, and virtual C translates on its own MMU index (capstone_supervisor.c:167-222 at b290f5656600), so the model's behaviour is the reference. Design of record for the next bitstream: every satp install by the switcher (entry, escape, resume) flushes ALL entries of the instruction, data and shared TLBs through an unqualified flush-all input (CVA6's flush_tlb_i is qualified by the asid and vaddr registers that only a real sfence.vma loads, so a pulse into it flushes at most one arbitrary entry; the first fix attempt did that and the pre-registered arm caught it); cost measured by proxy at ~91 cycles per flush, ~2 % of a context round trip. Bounds of the demonstration: a stale DATA translation (S-23's fetch mechanism strengthened, not shown); both pages inside one capability region, so containment was not tested. Corpus consequence: none, because the trusted caller runs in the process's own context and the context's satp is the process's, so no switch in the kit crosses roots; any design with a root per context (the deferred paged-table/32-context design) binds on the fix. The context's satp is read from frame +0 at ENTRY only; the escape saves the live satp to the hidden word +928 and every RESUME restores it from there (srevroot +936, options +944 likewise), so a rewritten +0 changes nothing after the first entry. CVA6 honours 16 ASID bits, so a kernel ASID rollover while the process is switched out between steps would resume the context under an ASID since given to another address space; the trusted side therefore writes the address space's CURRENT satp into +928 before every resume (the frame is kernel-owned), which repairs the ASID where a flush cannot. QEMU's resume restores an internal copy; the pin that makes +928..+944 the resume's source is the next model pin. **Exception order on a capability load (Q-31, ruled 2026-10-18; the RTL is the reference).** On the RTL a misaligned access wins the exception merge (cause 4, the integer cursor as tval), a non-canonical address page-faults in the MMU (cause 13), and only then does the tag check raise 24 or 28; QEMU tests the tag first and raises 24 for the same instruction and tval. No verdict depends on it (no aligned canonical integer address passes the tag check on either side), so the model's order is a recorded divergence until a later pin aligns it; a prediction that compares causes models the RTL's order. **A CPMP entry cannot be rewritten while a supervised context is live (found 2026-10-20 by an arm that wedged on its own repair).** `csr_regfile.sv:3018-3021` makes a CCSRRW to CSR 0x001 or to any of 0x010..0x01F a privilege violation while `sup_active_q` is set, and `CCSR_CPMP(n)` is `0x10 | n`, so the whole CPMP bank is in that range; its own comment records that the violation reaches commit as an illegal-instruction exception and becomes a fault event, never a trap, and that an earlier form of the gate let a supervised domain's CCSRRW cpmp0 through. The consequence for trusted code is that an M trap handler entered WHILE a context is still active cannot repair or re-arm a CPMP entry: CTVEC, CEPC and CSCRATCH stay legal, the bank does not. A handler that needs to re-arm an entry must therefore clear the supervision first, or the repair silently does nothing and a refused access repeats indefinitely -- which is how this was found, an arm looping to the cycle limit with byte-identical counts across a fix to the repair's operands (capstone-ariane `be040f067`, va-fetchctx). **Capability tags exist only in the tagged physical region (vp2/tree A).** The shadow-tag store covers physical addresses in [0x80000000, 0xBC2D2D2D) only (the region is fixed at build; tag memory and the node pool sit above it); an STC to an address at or above it writes no tag, so the capability is silently stored untagged rather than refused. Within the region one scalar store of any offset clears its 16-byte granule's tag (the tag is one bit per granule, and the shadow-tag address drops address bit 3), which is what a trusted scrub of a page relies on. **A RESUME writes the switcher's refusal literal into the context's x(rd) (S-27, found 2026-10-18 on bf3db3175; the RTL fix is on the accumulating list).** The commit stage builds one switcher request for the entry and both resumes with `set_ra = rd` and `ra_data` = untagged 2 (`commit_stage.sv:606-624`, the comment at :613 restricting the 2 to the bad-frame REFUSE), and the switcher's finish() exempts only a GOOD ENTRY (mode 1), so an acknowledged resume in mode 3 (plain) or mode 4 (reply) writes integer 2, tag 0, into register 25 + rd AFTER the context's x1..x31 are loaded, i.e. into the resumed context's x(rd). Every resume, the asynchronous ones after a quantum or ecall escape included. A resume also RE-LATCHES rd (`:627`, `csr_regfile.sv:2110-2113`), and the next escape writes its kind through the latched value (`:839`). Adapter rule, mandatory on bf3db3175 and harmless on the fixed RTL and on QEMU: the trusted caller issues every register-restoring CSRUNV (action 0 on a live context, action 2) with rd = x0, so nothing is written (finish() skips on `set_ra == 0`); it reads the event kind from frame +16 ALWAYS, the entry's first event included, since the latched rd is then x0; it writes a sentinel outside the kind set {0..5} (0 is the domain-return kind) to +16 before every issue and reads the sentinel back as REFUSED, because a refusal is not an escape and leaves +16 stale; no code path takes a kind or any value from a CSRUNV's result register after any resume. The discard (action 1) never reaches the switcher and keeps its rd. QEMU writes nothing to rd on a resume and the kind to +16 on every escape, so a QEMU record cannot show this defect; the matched pair is the board (the old module's cause-6-at-0x12 and cause-24/29-at-0x2 faults, all through a5 = the resume site's rd, against the new module's). **Collection under the table namespace is software, and on bf3db3175 it is not available (ruled 2026-10-18; Q-27, kit blocker 6).** On both models a CSRETIRE under the table namespace only clears the record's VALID (RTL reclaimer's else branch, "retirement is not a push; collection is software"; QEMU `cap_rev_retire` via table_put); `free_count` and `next` do not move, so `capacity − next + free_count` is unchanged by retirement. Allocation pops `free_head` when non-zero, else bumps next-unused, and the RTL re-reads slot 1 {free_head, free_count} at every allocation, so a software push is honoured. The liveness compare is generation-aware at its single site (`valid && generation == id[16+:14]`, fronting LCC, SPLIT, REVOKE, TIGHTEN, CALL, RETURN, LDC and STC), and the PHYSICAL pop bumps the generation and packs it into the id; but the VIRTUAL pop and the fresh path hand out the bare index, so every virtual record and reference carries generation 0 and a reused index REVIVES every stale capability naming it. QEMU prevents that with `virtual_collect`'s tag sweep over the context's pages (capstone_supervisor.c:806-822) and register scrub (:858-866) before the free-list push (cap_rev_table.c:255-289); trusted S cannot sweep tags on the RTL (Q-23). Therefore on bf3db3175 the module's collect returns -ENOMEM with a must-fire line (CSRUNV action 3 is illegal at execute, `rs2.cursor > 2`), a context that exhausts its 65,536 records ends, and an arm that collects on QEMU is a "no reading on bf3db3175: needs collection" row. Design of record, DEFERRED (ruled 2026-10-18; not in the next bitstream, because the corpus census read 0 collections on every bundle and only the contract's churn rows need reclamation): the flat record has NO generation field today (the glue composes the record with a literal 14'd0 where the generation belongs, `ex_stage.sv:1402-1404`; the 16-byte record of `cap_rev_table_abi.h:24-25` is fully packed but for bit 63 of word 1), so the generation takes word 1 bits [59:46], the parent field's unused upper 14 bits (the unit keeps parent in 16 bits), with no record growth, no index arithmetic change and the capacity unchanged; a fresh table zeroes them. Four coordinated changes when it is pulled forward: the glue's read and write of those bits; the virtual pop bumping and packing the generation exactly as the physical pop does, with the A6 masking rule and saturation at 16383 (a record at 16383 is retired permanently, never relinked); the model's ABI header and table code; the module and the adapter. The module's collect then becomes the table walk alone: for each record neither VALID nor FREE and below 16383, set FREE, link next to the old head, update slot 1 {free_head, free_count}; no tag sweep. Until then, -ENOMEM at exhaustion is the virtual profile's behaviour on bf3db3175 and on the next bitstream. The walk reads the records' VALID/FREE bits and never trusts a software retired count: frees invalidate records without a CSRETIRE (the stopgap's gate read retired 1 at exhaustion with 65,024 nodes consumed). QEMU's default refusing action 3 is the next pin, so the model stops implementing what the RTL refuses. views and the event words are unchanged. A context's instruction fetch is permission-checked as USER on the RTL
(`cva6_mmu.sv:140`, `:377-379`), so a context can never execute from a page without the U bit.

**The caller's snapshot at +1024..+2304 must never sit in a page the application can reach (found
2026-10-18 by the board lane's auditor, filed as an S-entry).** On the RTL every escape restores the S caller
(pc, x1..x31, the S and M CSRs, FP) from frame+1024 with no check, so a domain holding a capability to a
child thread's frame page could write the kernel's saved pc/sp/satp/stvec and have the next escape load them.
QEMU keeps the caller in its own slot and cannot show it. Rule: the trusted adapter makes a child thread's
frame unreachable by the domain before its first entry (it retires the frame arena's ancestor after pinning the
page and keeps the page kernel-only; the main thread's frame is a kernel page already); and the RTL item for the
next bitstream keeps the caller snapshot out of any domain-reachable page. Item D's decision 1 placed the
snapshot in the frame on QEMU's assumption that the frame is not mapped into the application.

The base frame is 576 bytes; nonzero options require 640 bytes; collection
requires 656 bytes. Unknown options are rejected. First entry consumes and
clears all 32 initial PCC/GPR slots. Later resumes use the saved continuation;
rewriting the frame does not replace its roots or registers. The reply slot
is the explicit exception described below. Platform capability registers
start empty; FP/vector state, when present, is also preserved by the context.

| Action in rs2 | Meaning |
|---|---|
| 0 | First entry, or resume a paused non-service event at its saved PC |
| 1 | Forget the saved context; does not retire its mappings |
| 2 | Complete a paused ECALL: consume frame slot 224 into a0 and advance PC by four |
| 3 | Collect the paused namespace; return reclaimed count, or `UINT64_MAX` on failure |

On escape, the processor saves the application and restores the caller's
registers, privilege and roots. It writes an event and returns its kind in rd.

| Kind | Event | Adapter response |
|---|---|---|
| 1 | Quantum/asynchronous escape | Let Linux schedule; action 0 resumes |
| 2 | Terminal fault | Forget and clean up; cannot resume |
| 3 | Service ECALL | Supply reply, then action 2; cause is 11 in this encoding |
| 4 | Page fault | Resolve a permitted fault, then action 0 retries the instruction |
| 5 | Node pressure | Sweep and reclaim before retrying; fail if capacity remains insufficient |

There are currently 32 supervisor slots on the hart, shared with the physical
supervisor. This is a QEMU resource limit. A terminal context must be forgotten
before its frame is reused. The frame format does not by itself specify a
future RTL context-storage implementation.

## Access and instruction policy

Data access requires a valid tag, the appropriate capability type and rights,
the complete accessed span inside bounds, and a live node. Translation then
requires the corresponding user PTE permission. Scalar stores clear tags on
overlapping physical 16-byte granules, including through kernel aliases.

PCC must have an allowed type, execution permission, a live node and bounds
covering the whole instruction. QEMU checks before reading instruction bytes
for translation and again when executing cached translated code. Virtual
translation blocks currently contain one instruction, so revocation cannot
leave a long block executing with stale PCC authority.

Ordinary capability arithmetic, lifetime operations, loads/stores and jumps
remain available. Virtual code rejects `SEAL`, `CALL`, `RETURN`, `CAPENTER`,
`CCSRRW`, supervisor entries and debug instructions. Privileged SYSTEM
operations other than ECALL/EBREAK, supervisor CSRs, and legacy Capstone
CSRs `0x800..0x803` are rejected with illegal-instruction cause 2. This
profile uses its own supervised entry/service boundary instead of domain calls.

## Linear transfers and faults

Protected `LDC` consumes a tagged source slot when the transferred type is
not NONLIN. `STC` similarly consumes its tagged non-NONLIN source register.
A consuming load needs read **and write** authority and a writable PTE.
Complete-span, alignment, permission and ordinary-RAM checks happen first.
Read and consume must use the same translated physical granule, including
when a PTE was changed without a translation fence. A fault leaves bytes,
tags and registers unchanged; a retry moves once.

**A and D are software's on vp2 (S-24, found 2026-10-18).** The RTL's walker never sets the accessed or dirty
bit: a load or fetch on a leaf with A clear, and a store on a leaf with W or D clear, raise a page fault and
install no TLB entry (`cva6_ptw.sv:453/468/477-481`, `cva6_mmu.sv:142/602`). Linux's GUP on a present page marks
the struct page, not the PTE, so the trusted adapter resolves a kind-4 STORE fault on a present page through the
kernel's write-fault path (`fixup_user_fault` with FAULT_FLAG_WRITE, which sets D and fences the address) before
pinning; a read fault's populate sets A. The prediction kit runs QEMU with svadu=false so hardware A/D never hides
this class.

**On vp2 (bf3db3175) a capability store must not be the first access to a page (S-25, found 2026-10-18).** The
RTL's store syncer hands the DYN unit a response with no exception field, so an `STC` whose translation faults
retires with no exception and no kind-4 escape while commit still counts it: a ghost entry in the store buffer's
commit queue that never drains, wedging every later DYN operation (the sparse steps) or, after four, every store
(the kit applications). Confirmed by a matched pair in simulation (va-runv8 vs its plain-sd control). The RTL fix
(the store exception carried to the STC's writeback, as R-56 did for loads) is the next bitstream's; until then the
trusted adapter pins a whole arena on its first kind-4 fault (every allocator-owned arena is first touched by
enframe's plain prefix load, so no stc reaches an unpinned page through malloc'd memory), and a region whose
first access is an stc through a pointer obtained without the allocator (a direct mmap) is not supported on this
bitstream (the kit's sparse steps are that case). QEMU escapes the faulting STC as kind 4 and cannot show this.

**Trusted S performs no capability memory access (ruled 2026-10-18, Q-23).** An `LDC` or `STC`
whose base register is an integer reads 24 (tval = the base register) in every mode, as on the RTL.
QEMU's earlier experimental "trusted-kernel context spill" branches (an S-mode scalar-address `LDC`,
and the matching `STC`, which QEMU also had) survive only
under the CPU property `x-capstone-test-s-ldc=true`, default off, which exists for four directed
suites (`virtual-capstone-m1`, `movc-untagged-source`, `trusted-linux-u-access`, and
`virtual-capstone-runtime`'s `run.sh` and `index8.sh`) whose bootstrap or scaffolding loads minted
capabilities into registers from S; it is a test scaffold with no silicon equivalent, no production
kit sets it, and their passes say nothing about trusted S. The follow-up that retires the property
is rewriting that bootstrap onto the `CSRUNV` route. The reference QEMU for the virtual profile is
capstone-qemu `virtual-capstone-v7` b290f5656600 (four commits on d35af47b163c: descriptor v7 and the
scalar-S gate at f2f15f8fc70c; strict DELIN and `x-capstone-supervisor-slots`, default 1 as on
silicon, clamped 1..32, at 852c5bc76774; `scapctl` and `urevavail` illegal by default under
`x-capstone-test-csrs` at 3b8040d5ff6e; the escape writing the frame's register slots and the resume
loading them, Q-30, at b290f5656600). Tier 0 on it at the suites' own settings: virtual-capstone-m1
74/74, movc 67/67, sublet-lifetimes 119/119, runtime run.sh 71/71 (69 + frame_slots ×2), index8 PASS,
fp 60/60 (three isolation cases set 32 slots explicitly), exact-bounds 4/4, trusted-linux-u-access
71/71 (m1, movc and trusted-linux set `x-capstone-test-s-ldc` and `x-capstone-test-csrs`: their
bootstrap selects protected U through `scapctl`). No prediction kit sets any of the three properties;
the kit gate parses their values; the prediction kit runs with `svadu=false`. The adapter obtains
every capability it needs through `CSMINT` (descriptor v7) and never holds one in a register.
The Q-12 protected-path changes did not change legacy physical C transfers.

**`DELIN` of a non-LINEAR capability faults with 26 (ruled 2026-10-18).** The RTL raises 26 for any
non-LINEAR rs1 (`capstone_dyn_unit.anvil:794-801`); QEMU's earlier no-op for a NONLIN operand is
replaced by the same fault in the default profile, so a prediction run sees what silicon sees. The
image gp is NONLIN by ABI (every launch grant is). The `-capstone-image-gp` materialization of a
global address today is `scc`/`cincoffset` off gp followed by a `delin` that is dead on QEMU and
faults on vp2 (boot 2 of the virtual profile, 2026-10-18: 525 of contract.dom's 530 delins): this is
C-27, which ISSUES-ARCHIVE recorded FIXED while the backend still emits it unconditionally. Dropping
the `delin` is sound only under the asserted NONLIN-gp ABI (two consumers rely on the base being
NONLIN: the copy hazard and the SHRINK double read), and that decision is reserved to the project
lead (`CapstoneISASemantics.md:52`); the semantics owner's recommendation is to drop it with the
contract asserted in the backend and the pin flipped.

| Cause | Capability fault |
|---|---|
| 24 | Unexpected operand type, including a missing tag |
| 25 | Invalid capability/lifetime |
| 26 | Unexpected capability type |
| 27 | Insufficient capability permissions |
| 28 | Out of bounds |
| 29 | Illegal operand value |
| 30 | Insufficient resources |

Capability memory faults report the effective VA; virtual PCC faults report
the faulting PC. Other capability faults and resource exhaustion report zero.
Explicit fault data is consumed on delivery so an earlier address cannot
leak into a later event. The direct-U experiment's `medeleg` changes are
separate from virtual C event delivery through `CSRUNV`.

## Lifetime storage and encoding

The guest table is kernel-owned writable RAM with a 4-KiB-aligned physical
`srevroot` and 16-byte records. The original flat format requires contiguous
backing. Slot 0 is a header, never a node.
The header contains capacity (bit 63 enables recycling) and the next-unused
ID. Recycling also reserves slot 1 for free-list metadata and allocation
statistics, so its first usable ID is 2. A node record's first word contains
u32 previous and next IDs. The second contains depth in bits 0..29, direct
parent in bits 30..59, and VALID/LINEAR/FREE in bits 60..62; bit 63 is reserved.
Parent 0 denotes a root. Host records use their existing UINT32_MAX sentinel.
Depth overflow refuses allocation. Node records remain 16 bytes.

The node-growth extension adds a paged format while retaining flat-table
support. Bit 62 selects paging and requires the recycling bit. The physical
root contains these little-endian fields:

| Byte offset | Field |
|---|---|
| 0 | Capacity in slots, including reserved IDs, and format bits 63/62 |
| 8 | Next unused ID |
| 16 | Free-list head (low u32), free count (high u32) |
| 24 | Cumulative allocations |
| 32 | Version magic `0x3245474150444f4e` (`NODPAGE2`) |
| 40 | Retired records not yet returned to the free list |
| 48, 56 | Reserved, zero |
| 64 | 16 physical directory pointers |

The 30-bit ID is split into `4:9:9:8`: root entry, two 512-entry directory
indices, then record index. Each directory and record page is aligned
ordinary writable 4-KiB RAM. Missing, misaligned or non-RAM pointers, and a
pointer back to an earlier page on the path, refuse lookup. The trusted adapter guarantees that pages do not
alias one another or other objects. All pages below the published capacity
must exist. Initializing pages and publishing links precedes the capacity
update, with all namespace execution stopped. Pages and the root remain
stable until all bound contexts have been discarded. Node accesses are
physical metadata accesses; they do not use `satp` translation.

This QEMU backend does not cache directory translations or implement large
leaves. It caches only a validated physical RAM section and its host pointer,
assuming a static physical memory map; directory and node bytes remain fresh.
RAM hotplug and migration are outside this prototype. Paging adds physical
lookups on capability checks; this qualification is not an RTL speedup claim.

Existing records and up to two allocation candidates are preflighted before
any node mutation; this includes fresh candidates straddling a page boundary.
Sublet changes the node record and capability encoding together. The
[shared format header](../../../capstone-qemu/target/riscv/cap_rev_table_abi.h)
is used by the QEMU backend and copied into the Linux module build.
The module's namespace and page quotas use that header's 30-bit limit.
Rebuild the module with this QEMU; old node/capability layouts are incompatible.
The paged magic rejects the old layout. The flat test layout has no version
magic and must be initialized afresh by matching software.

For scalable collection, bit 63 of frame word 80 selects a linked page list;
the remaining bits hold its total entry count, and word 81 holds its first
physical page address. Each list page contains a next pointer, count 1..510,
then physical page addresses. The final next pointer must be zero. QEMU
preflights the entire request and issued node pages before clearing tags;
no ID is released after an incomplete sweep. The original contiguous-vector
collection request remains valid. Dead PCC IDs remain pinned.

IDs must be allocated, below the high-water mark and valid; missing, free,
reserved and out-of-range records fail closed. The same depth-ordered list
algorithm operates over host and guest backends. Mutations preflight records
and capacity, and revocation walks have a visit bound. These checks assume
trusted metadata and the one-hart profile.

The QEMU memory encoding is 128 bits: ordinary cursor in the low 64 bits;
the upper word contains 27 bounds bits, 3 type bits, 4 permission bits and
a 30-bit node ID. It contains neither generation nor address-space bits.
The deployed RTL instead allocates 28 bounds bits and 30 node bits.
Moreover, some QEMU data paths retain uncompressed bounds beside the tag.
Exact minting alone does not settle general SHRINK/SPLIT/store/load
representability. [Encoding conformance remains open](guarantees.md).

Implementation: [decoder][decode], [CSRs][csr], [mint/retire][table],
[context and collection][supervisor], [instruction policy and translation][translate].

[qemu]: https://github.com/project-starch/capstone-qemu/tree/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9
[spec-patch]: https://github.com/project-starch/llvm-capstone/blob/e0313149630450c1a906cc3506b317add64266e0/capstone/docs/plans/virtual-capstone-isa.patch
[decode]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/insn32.decode
[csr]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/csr.c
[table]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/capstone_table.c
[supervisor]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/capstone_supervisor.c
[translate]: https://github.com/project-starch/capstone-qemu/blob/9bf9c1f28653632c2ce93a10d4bdd69eb26e04b9/target/riscv/translate.c
