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
regions: genesis physical capabilities lie below `2^56`, CREATE produces ranges
inside the logical region, and SHRINK, SHRINKTO and SPLIT only narrow. The
walker and the LSU classify by address (`address >= 2^57`), which costs no bit
of the 128-bit format. The binding word of E3 is the second witness: it is
nonzero exactly for logical capabilities, and a capability whose region and
binding disagree is a hardware fault, never an access.

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
reused before it retires (`2^20` CREATEs; an exhausted id keeps its entry
reserved, as the RTL's revocation generation retires rather than wraps). CREATE
scans all entries for range overlap; that is a bounded loop in a rare
instruction, and an interval structure is an optimisation for later.
Sixteen-bit ids (4 MiB of protected memory) are deferred until a measured need.

### E5 Capability kinds and the `ty` field

The candidate needs two kinds the ISA lacks, and reuses two it has:

| Candidate object | Encoding |
|---|---|
| Page-table capability | New type `TABLE`. Exists only in table slots and registry entries, written by CREATE and POPULATE; never in a register |
| Teardown token | New type `TOKEN`; linear; its node is the mapping's senior node, whose entry is DETACHED |
| Detach handle | Existing `REV` whose node carries a protected "mapping senior" flag set by CREATE. REVOKE refuses such a node; DETACH requires it |
| Domain handle | The existing `SEALED` capability of the target domain |
| Resume destination | A capability slot index inside that sealed domain context, read by the domain at resume |

The two spare `ty` codes would suffice for TABLE and TOKEN, but leave nothing
for later stages. The field therefore widens from 3 to 4 bits, taking the bit
the RTL never uses: `revnode_id` shrinks from 31 to the RTL's 30 bits. The
exact placement of the 33 low bits after this shuffle is an implementation
detail of `cap_compress.c` and of the RTL's field constants, not part of this
decision.

### E6 Representability

Under the compressed-bounds rule of `cap_compress.c`, a range shorter than 4
KiB is exact and a longer one is exact only if both ends are multiples of
`2^(E+3)`, where `E` is the highest set bit of the length minus 12. The
monitor's range allocator therefore hands out mapping ranges whose ends are
aligned to that grain, and the mapping capability's bounds are exact. This is a
convenience, not a security condition: the walker checks every access against
the registry entry's range through the capability's binding, so bounds that had
rounded outward could never reach another mapping. Objects inside a mapping
round as they do today. On the current QEMU the side table keeps fat bounds
exact, so the rule is checked by the model's arithmetic and by the RTL, not by
QEMU.

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
the result; the libc checks the delivered bounds, rights and binding against
its pending request.

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
- **M4, RTL:** node memory and TLB tag widths of E3, the registry lookup on the
  access path, the CREATE loop, and the field shuffle of E5. These are the cost
  questions of §10.3; none is answered here.

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

The bit placement inside `cap_compress.c` after E5; the monitor's range
allocator policy; the RTL cost of E3 and E4; the future of QEMU's fat-bounds
side table, which decides whether representability can ever be tested there.
