# ISA and context interface

[Guide](README.md) · Previous: [Runtime](runtime.md) · Next: [Development](development.md)

This chapter describes the experimental interface introduced at
[QEMU revision 9bf9c1f28653][qemu], plus the paged node-store and collection-list
extensions on `virtual-node-growth`. The superproject QEMU pin selects their
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
| `CSMINT rd, rs1, rs2` | `0x50` | Destination slot VA, descriptor VA | Scalar private ancestor ID; tagged linear child in slot |
| `CSRETIRE rd, rs1, rs2` | `0x53` | Ancestor ID; conventionally zero in rs2 | Zero on success, one on rejected retirement |

Virtual applications cannot execute these instructions. `CSRUNV` and
`CSRETIRE` require trusted S or scalar M mode and reject C mode. `CSMINT`
rejects U and virtual C; its decoder also accepts legacy physical C when
the extension is enabled. It must remain a trusted root-creation operation.
The current `CSRETIRE` helper ignores rs2.

| CSR | Address | Meaning |
|---|---|---|
| `srevroot` | `0x5c1` | S/M read/write physical lifetime-table base, rounded down to 4 KiB |
| `urevavail` | `0xcc0` | Read-only available IDs; zero for an invalid/missing table |
| `scapctl` | `0x5c0` | S/M read/write; bit 0 controls the separate protected-U experiment |

`scapctl` is not how the launcher enters virtual C. `CSRUNV` binds both roots
when admitting a context, then restores them from saved state on resume.
Virtual code may read `urevavail` but cannot replace the roots. Physical
C/M retains the legacy host-tree backend; the virtual context uses its guest
table. There are no implemented `CSCHECKR`/`CSCHECKW` instructions at this
revision. Linux syscall and VM service numbers belong to software ABIs.

### Minting and retiring an arena

`CSMINT` reads an 8-byte-aligned descriptor of three little-endian u64 values:
`base`, exclusive `end`, and permissions (`X=1`, `W=2`, `R=4`). The destination
must be a writable, 16-byte-aligned ordinary-RAM slot. Descriptor words and
the destination are translated with the trusted caller's permissions.

It requires `base < end`, permissions at most 7, two available IDs, and an
exact compress/decompress round trip of the initial bounds and cursor.
All checks precede table and slot mutation. Success creates a private senior
ancestor and a linear child. The scalar ancestor lets the adapter later
retire every application derivation under that arena without keeping a REV
capability in ordinary kernel C state.

`CSRETIRE` checks and walks the descendants, invalidates and unlinks them,
then invalidates and unlinks the ancestor. It does not unmap pages, erase
payloads or immediately recycle IDs. Those are ordered adapter duties.

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

These rules also cover trusted S-mode tagged transfers used by the module.
The Q-12 protected-path changes did not change legacy physical C transfers.

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
statistics, so its first usable ID is 2. A node record contains u32 previous
ID, next ID, depth and flags (VALID=1, LINEAR=2, FREE=4).

The node-growth extension adds a paged format while retaining flat-table
support. Bit 62 selects paging and requires the recycling bit. The physical
root contains these little-endian fields:

| Byte offset | Field |
|---|---|
| 0 | Capacity in slots, including reserved IDs, and format bits 63/62 |
| 8 | Next unused ID |
| 16 | Free-list head (low u32), free count (high u32) |
| 24 | Cumulative allocations |
| 32 | Version magic `0x3145474150444f4e` |
| 40 | Retired records not yet returned to the free list |
| 48, 56 | Reserved, zero |
| 64 | 32 physical directory pointers |

The 31-bit ID is split into `5:9:9:8`: root entry, two 512-entry directory
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
The same node-list algorithm and capability encoding are retained. The
[shared format header](../../../capstone-qemu/target/riscv/cap_rev_table_abi.h)
is used by the QEMU backend and copied into the Linux module build.

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
the upper word contains 27 bounds bits, 3 type bits, 3 permission bits and
a 31-bit node ID. It contains neither generation nor address-space bits.
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
