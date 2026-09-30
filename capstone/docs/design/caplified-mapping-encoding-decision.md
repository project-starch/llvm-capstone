# Caplified mapping tables: encoding and partition decision (M0)

Status: DECISION, 2026-09-30, on the `delegation-memory-encoding` lane. It
fixes the numbers and placements the [mapping
candidate](caplified-mapping-tables.md) left to "an encoding decision" (§4,
§10.3), so that the QEMU prototype (M1) and the delivery transport (M2) can
start. Nothing here is implemented in QEMU, the monitor, the compiler or RTL;
the hardware cost of every item stays with §10.3 and the RTL lane. The
[executable model](../../tests/mapping-model/README.md) applies the rules below
to the architectural constants in its `check_architectural_partition` scenario
and records them in its result.

## 1. Facts this decision rests on

| Fact | Source |
|---|---|
| The physical address width is 56 bits on both platforms | capstone-ariane `core/include/build_config_pkg.sv:36` (`cfg.PLEN = 56` for XLEN 64) and `core/include/riscv_pkg.sv:26`; capstone-qemu `target/riscv/cpu-param.h:13` (`TARGET_PHYS_ADDR_SPACE_BITS 56`) |
| Installed RAM lies far below that width: the FPGA's DRAM is `[0x80000000, 0xBC3C0000)` and QEMU's RAM also starts at `0x80000000` | [HOW-TO-LAUNCH-ON-FPGA](../ref/HOW-TO-LAUNCH-ON-FPGA.md) (the `reg = <0x0 0x80000000 0x0 0x3c3c0000>` line) |
| The 128-bit compressed capability has no spare bit: the second word is `bE:3 b:11 tE:3 t:9 iE:1 ty:3 perms:3 revnode_id:31` (bits 0..2, 3..13, 14..16, 17..25, 26, 27..29, 30..32, 33..63) | capstone-qemu `target/riscv/cap_compress.c:30-37` at the pinned commit `ac2837aa` and at the working copy `408fd839` |
| The RTL carries revocation-node ids in 30 bits, QEMU in 31 | capstone-ariane `core/commit_stage.sv:75`, `core/load_store_unit.sv:199` (`logic [29:0]`); R-35 in [ISSUES.md](../ref/ISSUES.md) records the split as a 14-bit generation and a 16-bit index |
| Six of the eight `ty` codes are used: LIN, NONLIN, REV, UNINIT, SEALED, SEALEDRET | capstone-qemu `target/riscv/cap.h:26-32` |
| A QEMU revocation node holds `prev, next, depth, valid, linear, refcount`; the pool defaults to 65,536 nodes and `CAPSTONE_REV_NODES` overrides it | `target/riscv/cap_rev_tree.h:14-26` |
| SPLIT and MREV allocate a new node for the derived capability | `target/riscv/cap_rev_tree.c:107`, `target/riscv/op_helper.c:1144` |
| The current QEMU keeps exact fat bounds in a side table on tagged stores, so compressed-bounds rounding is not observable there | [capability bounds model](capability-bounds-model.md), correction of 2026-06-29 |
| The compiler compares capabilities by their 64-bit cursor, lowers `ptrtoint` to a bare `mv` and `inttoptr` to an untagged capability carrying the integer as its address | `llvm/lib/Target/Capstone/CapstoneISelLowering.cpp:7992` and `:8003` (`lowerSETCC`), `:2264`, `:8084-8087` |
| The RTL's second word is `revnode_id:30 perm:3 cap_type:3 bounds:28`, and the 28th bounds bit selects between two bounds codecs, `full` (`iE:1 t:9 tE:3 b:11 bE:3`) and `cursorless` (`t:21 e:6`); there is no free bit | capstone-ariane `core/include/ariane_pkg.sv:612-642` (structs) and `:665-666` (`decompress_bounds`) at `f6ec6c1` |
| The RTL's packed 3-bit type field is fully assigned: `NOT_CAP, LINEAR, NONLIN, REVOKE, UNINIT, SEALED, SEALEDRET, EXIT`; `EXIT` occurs nowhere else in the core, and QEMU has no EXIT type. The type query answers `cap_type - 1`, the spec numbering 0..6 with 7 meaning "not a capability"; the compiler's tag test is `LCC(type) < 7` | capstone-ariane `core/include/ariane_pkg.sv:654-663`, `core/anvil_build/capstone_dyn_unit.anvil` (the S-06 comment on the type query); capstone-qemu `target/riscv/op_helper.c:910-927`; `llvm/lib/Target/Capstone/CapstoneISelDAGToDAG.cpp:1744-1752` |
| `LCC` selectors are Table 8 of the spec: 0 valid, 1 type, 2 cursor, 3 base, 4 end, 5 perms; none reads protected node metadata | `CapstoneISelDAGToDAG.cpp:1740-1741` |
| Every type but NONLIN moves linearly, REV included | `target/riscv/op_helper.c:1202` |
| QEMU's genesis leaves one capability ending at `2^63`, and the debug mint takes raw bounds | `target/riscv/op_helper.c:3021` (`helper_cscapenter`, "TODO: should be 2**64") and `:3039-3040` (`helper_csdebuggencap`), same at `408fd839`; the RTL's CAPENTER code capability is `[0x80000000, 0x80800000)`, `core/commit_stage.sv:200-201` |
| QEMU's full codec shifts a 32-bit `1` by `E + 14` when decoding, undefined for `E >= 17`, that is for lengths of 512 MiB and more; the fat-bounds side table hides it | `target/riscv/cap_compress.c:102-103` at `ac2837aa`; noted in the bounds model as "int-shift UB" |

## 2. Decisions

### E1 Address partition

| Region | Addresses | Meaning |
|---|---|---|
| Physical | `[0, 2^56)` | Today's capabilities; MMIO and RAM |
| Guard | `[2^56, 2^57)` | Never valid for any capability; catches boundary arithmetic |
| Logical | `[2^57, 2^63)` | Mapping ranges, globally disjoint, reserved by CREATE until DESTROY |
| Invalid | `[2^63, 2^64)` | Never valid; keeps every one-past cursor below `2^63` and the integer view of an address non-negative |

CREATE accepts `[lo, hi)` only if `2^57 <= lo < hi < 2^63`, both ends 4 KiB
aligned, and no reserved entry overlaps. The physical region is the
architectural width, not installed RAM: `0x80000000 + 0x3c3c0000` on the FPGA
is far below `2^56`, and the candidate's "one-past physical address" clause
means the region boundary must sit above `2^56` itself, which the guard octant
does. The partition is the same on QEMU and the FPGA because both use PLEN 56.

### E2 Kind follows the region, without an encoding bit

A capability is logical if its bounds lie inside the logical region and
physical if they lie inside the physical region. No capability straddles the
regions once genesis and minting are confined to the physical region, CREATE
produces ranges inside the logical region, and SHRINK, SHRINKTO and SPLIT only
narrow. That confinement is a change, not a fact: at the pinned QEMU the third
genesis capability of `helper_cscapenter` ends at `2^63`, and
`helper_csdebuggencap` mints whatever bounds it is given. M1 bounds the genesis
capabilities to `[0, 2^56)`, makes the debug mint refuse any range outside the
physical region, and gives both a zero binding; M4 checks the RTL's genesis set
beyond the code capability. Until then the region invariant does not hold at
start, and no claim of this document does either. The walker and the LSU
classify by address (`address >= 2^57`), which costs no bit of the 128-bit
format. The binding word of E3 is the second witness: it is nonzero exactly for
logical capabilities, and a capability whose region and binding disagree is a
hardware fault, never an access.

### E3 The binding lives in the revocation node

The (id, gen) binding of the candidate's §4 is a 32-bit word in protected node
memory, not in the capability: `id` in the low 12 bits, `gen` in the upper 20,
zero meaning "no binding" (gen starts at 1). CREATE writes it into the
mapping's senior node (the detach handle's) and junior node (the mapping
capability's). Every operation that creates a node for a logical capability,
SPLIT and MREV, copies the word from the parent; DELIN, SHRINK, SHRINKTO and
cursor arithmetic keep the node and therefore the binding. The access path
already reaches the node for the liveness check, so the binding costs no second
lookup: the check is node live, `registry[id].gen == gen`, the entry's root
valid, the address inside the entry's range, the PTE present. The TLB tag of §8
and §10.1 is the 30-bit node id plus this 32-bit word.

In QEMU this is one field in `CapRevNode`. On the RTL it widens node memory by
32 bits and the TLB tag by the same; that cost belongs to §10.3 and is not
estimated here.

### E4 Registry size and CREATE's overlap check

The registry has `2^12 = 4096` entries, one per id, in protected M-mode memory.
An entry holds gen (20 bits), state, class, max protection, the range as two
64-bit addresses, and the root page-table capability with its tag; 64 bytes per
entry, 256 KiB in total. Both widths are build constants: the id width bounds
the mappings alive system-wide, the gen width bounds how often one id can be
reused before it retires (`2^20 - 1` CREATEs, generation zero being "no
binding"; an exhausted id keeps its entry reserved, as the RTL's revocation
generation retires rather than wraps). CREATE scans all entries for range
overlap; that is a bounded loop in a rare instruction, and an interval
structure is an optimisation for later. Sixteen-bit ids (4 MiB of protected
memory) are deferred until a measured need.

### E5 Capability kinds and the `ty` field

The candidate needs two kinds the ISA lacks, and reuses two it has:

| Candidate object | Encoding |
|---|---|
| Page-table capability | No type code. The linear physical capability CREATE or POPULATE consumed, stored by hardware in its slot or registry entry; the walker reads slots by level. It is unreachable because no software capability covers a table page while it is one |
| Teardown token | No type code. The detach handle's REV, returned by DETACH, with the registry entry now DETACHED |
| Detach handle | Existing `REV` whose node carries a protected "mapping senior" flag set by CREATE. REVOKE refuses such a node; DETACH requires it |
| Domain handle | The existing `SEALED` capability of the target domain |
| Resume destination | A capability slot index inside that sealed domain context, read by the domain at resume |

There is no bit to widen `ty`, and no code to spare inside it. The RTL's second
word is full: 30 node bits, 3 permission bits, 3 type bits and 28 bounds bits,
the last of which selects between its two bounds codecs and does not exist in
QEMU, which spends it as a 31st node bit. Its eight type codes are all
assigned, the spec numbering runs 0..6 with `EXIT` at 6 (unreferenced in the
RTL beyond its enum and absent from QEMU, but a spec type nonetheless), and 7
is the total type query's answer for "not a capability", which the compiler's
tag test relies on. So the decision adds no type at all:

- **Table pages need no type.** A table page is protected by the absence of
  authority, not by a code: the linear capability that covered it was consumed
  by CREATE or POPULATE, the registry and the revocation-node table are
  hardware-private memory, and the parent slot that holds the page's capability
  lies in another table page, up to the root in the registry. No software
  capability covers a table page while it is one, so no load, store or LDC can
  reach a slot; I2 holds by authority. The walker knows the level it is
  reading: non-leaf slots hold page-table capabilities, leaf slots hold PTEs,
  both stored as the linear physical capabilities they were. The only authority
  over the page is the table-page handle, whose REVOKE returns UNINIT, so the
  contents are gone before anyone reads them.
- **The token needs no type.** The detach handle is a REV capability whose node
  carries the protected mapping-senior flag while the registry entry is ACTIVE;
  DETACH returns the same REV with the entry now DETACHED, and that is the
  token. REVOKE refuses a flagged node in either state; DETACH requires ACTIVE;
  UNMAP and DESTROY require DETACHED. A Stage-3 range token from SURRENDER is
  the same encoding with narrower bounds. REV moves linearly, as everything but
  NONLIN does, so the token has one holder.

The type field, the compiler's tag test and the RTL's word are untouched. A
later kind that must be register-visible, should Stage 4 need one, requires a
layout decision, dropping `cursorless`, narrowing the node id or retiring
`EXIT`, with its cost; that decision is not taken here.

### E6 Representability: an allocator rule, not a codec guarantee

This decision does not fix a bounds codec, and it does not claim exact mapping
bounds from the existing ones. QEMU's full codec has undefined behaviour when
decoding lengths of 512 MiB and more (`cap_compress.c:102-103`), which the
fat-bounds side table hides; an independent UBSan run of that codec turned `[L
+ 1 GiB, L + 2 GiB)` into `[L + 1 GiB, L + 3 GiB)` for `L = 2^57`. The RTL has
two codecs, and its cursorless decode rebuilds the far end from the cursor's
high bits, so a range that crosses the codec's alignment window can lose its
top; the review's hand computation of `[2^58 - 4096, 2^58 + 4096)` with the
cursor at the start gives an end of `2^58`. Neither codec has been
round-tripped at logical addresses.

Two things are decided. First, the monitor's range allocator reserves naturally
aligned power-of-two ranges with the base aligned to twice the size: a request
of `n` bytes reserves `2^k >= n` at a base that is a multiple of `2^(k+1)`.
Read against both codecs, this keeps the range and its one-past end inside one
alignment window, so a correct decoder reproduces the bounds exactly; it is
derived from reading the codecs, not from testing them, and it is what the
model's `reservation_ok` rule checks. The usable length is the request;
POPULATE backs only those pages, and the reservation above them stays none.
Second, M1 fixes the QEMU codec's shift and adds round-trip tests over bases
across the logical region, with the cursor at the start, inside, at the end and
one past, for the full codec; M4 repeats them for both RTL codecs. Until those
tests pass, "exact" is a target.

None of this is a security condition: the walker checks every access against
the registry entry's range through the capability's binding, so bounds that
decoded outward could never reach another mapping. Objects inside a mapping
round as they do today.

### E7 The compiler is unchanged for Stage 1

`lowerSETCC` keeps comparing the 64-bit cursor: logical ranges are disjoint
from each other and from the physical region, so equal cursors name the same
object address, and `NULL` (0) lies outside the logical region. `ptrtoint`
keeps returning the full 64-bit cursor and `inttoptr` an untagged capability,
so a logical address survives integer round trips as a value, without
provenance, as today. No new intrinsic, type or address space is needed. One
pre-existing caveat is unchanged: a physical capability whose cursor is 0
compares equal to `NULL`.

### E8 CREATE's operands

`CREATE(id, lo, hi, root page, max protection, PRIVATE, domain handle, slot
index)`, with the range checked by E1 and E4, the root page converted as §4 of
the candidate requires, and the mapping capability written into the named slot
of the sealed domain context inside the instruction. The monitor never holds
the result. The libc checks the delivered bounds and rights against its pending
request; the range alone identifies the mapping, since ranges are globally
unique (E1). To compare the binding with the (id, gen) the completion reported,
software needs a read the ISA lacks: the binding lives in the node (E3) and the
`LCC` fields stop at perms. M1 adds field 6, the binding word, read-only,
answering 0 for a physical capability and trapping on an untagged operand like
every field but the type. It is a consistency check for the libc and a
diagnostic, not the authority; the walker's binding check is.

## 3. Consequences for the milestones

- **M1, QEMU prototype:** the binding field in `CapRevNode`; the registry as
  protected emulator state; region classification on the access path; the
  CREATE overlap loop; the `ty` widening and node-id narrowing in
  `cap_compress.c`; the TLB entry tag. The fat-bounds side table stays as it
  is; representability is therefore not observable in QEMU and the E6 rule is
  exercised by the model and later by RTL.
- **M2, transport:** a range allocator in the monitor that keeps E6's alignment
  and issues ranges from the logical region, and the 256 KiB registry
  reservation in monitor memory. Per-domain arenas inside the logical region
  are an allocator convenience; global disjointness is enforced by CREATE
  regardless.
- **M4, RTL:** node memory and TLB tag widths of E3 plus the mapping-senior
  flag, the registry lookup on the access path, the CREATE loop, the genesis
  set, and round trips of both bounds codecs at logical addresses. These are
  the cost questions of §10.3; none is answered here.

## 4. Alternatives rejected

- **(id, gen) inline in the capability:** the second word has no spare bit, and
  the binding would then have to survive every 128-bit round trip and be
  checked against the node anyway.
- **Per-domain logical spaces:** D3 lets capabilities of different domains meet
  in one context after transfer; equal cursors would then name different
  objects (candidate §4).
- **A region above installed RAM:** RAM differs between platforms and a
  physical one-past address can equal `2^56`; only an architectural partition
  is stable.
- **A kind bit in the encoding:** unnecessary given E2, and it would spend one
  of the bits E5 needs.
- **Sixteen-bit ids now:** 4 MiB of protected memory without a measured need.

## 5. Open after this decision

The bounds codec at logical addresses, to be fixed and round-tripped in M1
(QEMU, after the shift fix) and M4 (both RTL codecs); the RTL's genesis set
beyond its code capability; the placement of the mapping-senior flag and the
binding word in RTL node memory and their cost (E3, E4); a layout decision for
any later register-visible kind; the future of QEMU's fat-bounds side table,
which decides whether representability can be tested there at all.
