# Line B step (a): census of the live monitor against the Sublet format (2026-11-21, read-only)

The first deliverable of programme 7.1 item 3's monitor port: no edits. The live monitor on every P1-family board kit
(bkit91, bkit100) is capstone-sbi `native/cscheck-emu-ids55-paged` 22a907f (the runtime lane's statement, 2026-11-21).
The kits build it with `BOARD_FW_DEFINES` = CAPSTONE_SUPERVISED_CALL, CAPSTONE_SUPERVISOR_CSR_EVENTS,
CAPSTONE_SUPERVISE_QUIET, CAPSTONE_VP2_CSMINT, CAPSTONE_CPMX_DIAG, CAPSTONE_VP2_FETCH_RETRY, CAPSTONE_QUANTUM_PRINT,
the two quanta, CAPSTONE_HPM_EVENTS(_PAGED), CAPSTONE_S23_REFUSAL_READ, CAPSTONE_S23_DATA_RETRY,
CAPSTONE_FETCH_RETRY_CAP, **CAPSTONE_CSCHECK_EMU, CSC_IDS55, CSC_TABLE_TREE** (bkit100's board.env). The census was
read in a worktree of that commit: capstone-sbi `lineb-monitor-port` = 22a907f, unchanged.

## The target format (capstone-qemu c1985bb529, the programme's reference revision)

`target/riscv/cap.h:226-246` and `cap_compress.c:73-105`:
- id bits 63: generation 32 | slot 31 | bounds 27 | address 38. The low word is bounds[25:0] << 38 | address[37:0]; the
  high word is generation << 32 | slot << 1 | bounds[26]. The id is generation << 31 | slot.
- The cursor is the ZERO extension of the 38-bit address. A cursor at or above 2^38 leaves the value untagged.
- **No type and no rights.** Every Sublet capability decodes as NONLIN with RWX. "Every instruction that would write a
  type or rights is illegal." Only bounds and lifetime decide an access; the page table holds read, write and execute.
- **The bounds field is today's 27-bit encoding** (the same `cap_compress`, the E from the length's high bit at or
  above 12), so representability is unchanged.
- Tables: the Sublet record and up to 2^31 slots (`cap_rev_table_abi.h`).

## Classification

**Format-free under Line B (carries over unchanged):**
- Representability. The 27-bit bounds encoding is today's, so these carry over: R-55's region granule and exactness
  helpers (`sbi_capstone.c:1527-1568`, repr_granule: 1 below 4 KiB, else 2^(hb(len)-9)), the INIT-fill notes on
  re-encoded ends (`:306-340`), and CSMINT's representability refusal (`capstone_csmint.S:40-41`, `:210-236`).
- The ISA-level field reads: `__capfield(c, 0..5)` (LCC queries: validity, type, cursor, base, end, perm;
  `sbi_capstone.c:263-266` and 15 uses), SCC, CINCOFFSET, LDC and STC. They use the instructions, not the bits. But
  see the type and rights rows below: what LCC TYPE and PERM RETURN changes (always NONLIN and RWX).
- Addresses. Every capability the monitor mints lies below 2^38: DRAM 0x8000_0000..CAPSTONE_DRAM_TOP 0xc000_0000
  (`capstone_target.h:34-37`), the UART, CLINT and PLIC MMIO, and the bootrom at 0x10000.
- The trap entry (`sbi_capstone.S`): CSR field arithmetic only (MPP, SPIE, mtvec mode).
- `process-abi.h`'s process ids (generation << 32 | slot): the monitor's own software ids, not hardware ids.

**Format-bound BITS, all in the two emulators (the part a step (a) port can actually change):**

| file:line (22a907f) | site | build | ids55 today | Sublet |
|---|---|---|---|---|
| capstone_cscheck.S:183-186, :315-318 | stored TYPE at hi[29:27] == CSC_STORED_NONLIN | CSCHECK_EMU | type field present | no type field; the check is void (every capability decodes NONLIN) |
| :198-211 | slot from the cursor word [63:39], read through a de-tagged scratch copy (B's load rule) | CSC_IDS55 | slot[24:0] at lo[63:39] | slot at hi[31:1]; the lo word is bounds|address |
| :199 | generation from hi[63:34] | CSC_IDS55 | 30-bit generation | 32-bit generation at hi[63:32] |
| :212-216 | id = hi >> 34, index/generation split at CSC_INDEX_BITS | !CSC_IDS55 (E5 flat) | 20/10 | not used by the live kits |
| :223-233, :347-351 | header word 0: capacity | RECYCLE << 63 | PAGED << 62; word 1 next | table | ids55 header | Sublet table header (NODPAGE2 root page with 32 root entries, per the scope note) |
| :245-253 | record word 1: VALID bit 60, FREE bit 62, generation at [59:30] (CSC_GEN_SHIFT/BITS) | CSC_IDS55 | the wide flat record | the Sublet record (generation 127:96, previous 95:65, next 64:34, depth 33:3) |
| :338-381 | the 5:9:9:8 tree walk (root entry slot >> 26, then >> 17 & 511, >> 8 & 511, & 255) | CSC_TABLE_TREE | 25-bit slots, so the root index is always 0 | the split already covers 31 bits (5+9+9+8 = 31) and carries over; only the slot's source (row :198-211) and the record format change |
| :256 | LCC PERM lacks R/W -> 27 | CSCHECK_EMU | rights present | always RWX: the check never fires; the page table holds the rights |
| :270-284 | address and end must be Sv39 (sext39: slli 25 / srai 25) | CSCHECK_EMU | sext39 | below 2^38, zero-extended |
| capstone_cscheck.S:81-85 and csmint's CSM_PHYS | the physical-page capability: CAPTYPE NONLIN + CAPPERM 6 | both | legal from M | ILLEGAL (no type or rights writes); the mint must be CAPCREATE + CAPBOUND only |
| capstone_csmint.S descriptor w2 (header :26-33) | perms bits 0-3, bit 8 NONLIN else LINEAR; CAPTYPE / CAPPERM on the minted capability | CAPSTONE_VP2_CSMINT | types and rights | the Sublet CSMINT descriptor (0x10f in the collaborator's kernel) and hardware CSMINT in Line B step (c) |
| capstone_csmint.S "fewer than two free identities" (header :38-44) | header words 0-2 as QEMU's cap_rev_table reads them | CAPSTONE_VP2_CSMINT | ids55 header | Sublet header |

**Format-bound SEMANTICS, in the C monitor: no expression in the Sublet format at all.** These are counts of mentions
in `sbi_capstone.c`, comments included. They size the problem; they are not an executed-site list:
- LINEAR, `__linear` or CAP_TYPE_LIN: 74;
- UNINIT, csinit or INIT: 34;
- the type read `cap_type()` or `__capfield(.., 1)`: 27;
- SPLIT: 19;
- REVOKE or MREV: 9;
- SEAL, CALL or RETURN: 7;
- SHRINK: 5;
- DELIN: 2;
- supervision or CSRUNV: 49.
The monitor's region carving (SPLIT demands LINEAR), its region reinitialisation (UNINIT, STC fill, INIT), its domain
entry (sealed frames, CALL/RETURN, CSRUNV) and its revocation (REVOKE/MREV of linear handles) all rest on types. Under
the Sublet format every capability is NONLIN, type writes are illegal, and the collaborator's kernel emits only LDC,
STC, LCC, CINCOFFSET, CSMINT, CDERIVE, CREVOKE and CSSHRINK (B5b scope note, "What the kernel requires").

## What this means for "the monitor is ported in step (a)"

- **What step (a) can port:** the emulators' bit sites above, behind a format define (a `CSC_SUBLET` beside
  CSC_IDS55). Those are about a dozen sites in capstone_cscheck.S and capstone_csmint.S. They can be tested against
  c1985bb529 once a Sublet table exists in the twin.
- **What it cannot:** the C monitor. Its design is typed end to end, and the format removes types. That is not a port
  but the redesign the B5b scope note's options A/B/C turn on ("option C keeps the current stack running" holds only
  if the monitor's CSRUNV route works without types). The census measures that statement on the live monitor: the
  route uses sealed frames, LINEAR region handles and UNINIT reinitialisation throughout.
- **Proposal:**
  - step (a) = this census plus the emulators' `CSC_SUBLET` sites, on capstone-sbi `lineb-monitor-port`, never on
    `native/cscheck-emu-ids55-paged`, which running kits pin;
  - the C monitor's redesign waits for the lead's A/B/C ruling, which this census informs.
