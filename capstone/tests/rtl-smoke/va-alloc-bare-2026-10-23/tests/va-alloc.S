# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-alloc.S  (vp2 item A, step 2: node allocation and retirement driven by the flat table's header, 2026-10-17)
#-----------------------------------------------------------------------------
#
# The step-1 table (va-flat.S) with a free list of two records, 5 -> 4 -> 0 (free_count 2) and next-unused 6, under the
# flag: the domain derives three children from its root (record 2) and the unit must allocate exactly as QEMU's
# table_alloc: ids 5 and 4 from the free list (head first), then 6 by the bump; root+8 becomes 7, root+16 becomes 0
# (head 0, count 0), root+24 becomes 3. Then the domain crevokes the third child (6): the unit clears VALID and sets
# nulls its links (retirement is not a free-list push; collection is software), so record 6's word1 has VALID clear and
# FREE clear and root+16 is unchanged. The domain stores each child's node-bearing reading through s1: the three ids are
# not readable by lcc, so the readings are the children's cursors relative to data (0, 64, 128: the derive placed
# them where the domain asked) and 0x19 for a load through child 6 after its crevoke.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order), flag on:
#   srevroot read back minus the table: 0; status 0; csupcause 2 (since item E, 2026-10-17: the domain's RETURN is refused in a virtual context; 0 before); through s1: 0, 64, 128, 0x19; csupstatus 5 (the escape; 1 before item E);
#   root+0: 0x8000000000000400; root+8: 7; root+16: 0; root+24: 3; record 5 word1 VALID 1, parent 2, depth 1; record 4 word1 VALID 1, parent 2,
#   depth 1; record 6 word0 prev 0 (the retirement nulls the retired record's links, as QEMU's cap_rev_retire);
#   record 6 word1 VALID 0 and FREE 0 ((word1 >> 60) & 7 = 0), parent 2, depth 1;
#   mcause 0.
# Flag OFF (-DVA_FLAG_OFF, the positive control): the same program allocates from the unit's own head in the physical
#   pool (3, 4, 5): through s1 the same 0, 64, 128, 0x19; csupstatus 1; the table as the monitor built it: root+0
#   0x8000000000000400, root+8 6,
# -DVA_CORRUPT (flag on): record 4's word1 is (FREE << 60) | 1, not QEMU's exact FREE record, so the second derive (the
#   pop of 4) is refused with 29 (status 3 -> ILLEGAL_OPERAND_VALUE under the table namespace since D6/Q-20, QEMU's
#   CAP_REV_CORRUPT; 25 before) after the first allocation went through: 0; 0; 29; s1: 0 0 0 0; csupstatus 5; root+0
#   0x8000000000000400; root+8 6; root+16 0x100000004; root+24 1; rec5 1 2 1; rec4 4 0 1; rec6 prev 0; rec6 0 0 0; mcause 0.
# -DVA_EXHAUST (flag on): capacity 263 | RECYCLE, next 6, no free list, records 5 and 4 zero: 257 free identities, one
#   above QEMU's reserve of 256 (D5): the first derive bumps (id 6, next 7), the second finds 256 and is refused with 30
#   (status 2 -> INSUFFICIENT_SYSTEM_RESOURCES, QEMU's node pressure, kind 5 under CSRUNV): 0; 0; 30; s1: 0 0 0 0;
#   csupstatus 5; root+0 0x8000000000000107; root+8 7; root+16 0; root+24 1;
#   rec5 0 0 0; rec4 0 0 0; rec6 prev 2 (live, not retired); rec6 1 2 1; mcause 0.
#   root+16 = (2 << 32) | 5, root+24 0; records 5 and 4 FREE ((word1 >> 60) & 7 = 4, parent 0, depth 0) and record 6's
#   words 0; mcause 0.

# 2026-10-17 (D4/D5): the table's capacity is 1024, not 64: inside a context QEMU's allocator refuses below 257 free
# identities (CAP_REV_TABLE_RESERVE, node pressure 30), and the RTL's does the same since D5; root+0 reads 0x8000000000000400.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define SREVROOT    0x5C1

#ifdef VA_FLAG_OFF
#define VA_CTL_ARM 0
#else
#define VA_CTL_ARM 2
#endif

#define PTE_LEAF_RWX 0xDF
#define PTE_PTR      0x01

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

MINT(s5, CAP_TYPE_LIN, savearea, savearea_end, CAP_PERM_RWX)
MINT(s1, CAP_TYPE_LIN, scratch, scratch_end, CAP_PERM_RW)
sd x0, 0(s1)
sd x0, 8(s1)
sd x0, 16(s1)
sd x0, 24(s1)

# ---------------------------------------------------------------- the flat table with a two-record free list
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
#ifdef VA_EXHAUST
ori t0, t0, 263                # capacity 263 | RECYCLE: 257 free identities, one above the reserve
#else
ori t0, t0, 1024               # capacity 1024 | RECYCLE (above the 256-id reserve inside a context)
#endif
sd t0, 0(s8)
li t0, 6
sd t0, 8(s8)                   # next unused 6
#ifdef VA_EXHAUST
sd x0, 16(s8)                  # no free list
#else
li t0, 2
slli t0, t0, 32
ori t0, t0, 5
sd t0, 16(s8)                  # free_head 5, free_count 2
#endif
sd x0, 24(s8)                  # allocations 0
#define REC(IDX, REG) li REG, (IDX)*16; CINCOFFSET(REG, s8, REG)
#define REC_READ(IDX) REC(IDX, t1); ld t2, 8(t1); srli t3, t2, 60; andi t3, t3, 7; CAPPRINT(t3); srli t3, t2, 30; li t4, 0xffff; and t3, t3, t4; CAPPRINT(t3); li t4, 0x1ffff; and t3, t2, t4; CAPPRINT(t3)
# item 6: the GENERATION, word 1 bits [59:46]. REC_READ above prints the flags, the 16-bit parent and the
# depth and stops, so without this the arm can see a FREE bit and not the generation -- and a free bit alone
# does not distinguish a push from a recycle. This is the virtual analogue of r12-recl-stale's "65536+index"
# witness: that arm reads the re-minted ID, this one reads the field the ID is composed from.
#define REC_GEN(IDX) REC(IDX, t1); ld t2, 8(t1); srli t3, t2, 46; li t4, 0x3fff; and t3, t3, t4; CAPPRINT(t3)
#define CSNODE(rd, rs1)       .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSRETIRE(rd, rs1)     .insn r 0x5B, 0x1, 0x53, rd, rs1, x0
REC(2, t1)
sd x0, 0(t1)                   # record 2: the root, prev 0 next 0
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)                   # VALID, parent 0, depth 0
#ifdef VA_EXHAUST
REC(5, t1)
sd x0, 0(t1)
sd x0, 8(t1)
REC(4, t1)
sd x0, 0(t1)
sd x0, 8(t1)
#else
REC(5, t1)
li t0, 4
sd t0, 0(t1)                   # record 5: next free = 4
li t0, 4
slli t0, t0, 60
sd t0, 8(t1)                   # FREE << 60
REC(4, t1)
sd x0, 0(t1)                   # record 4: next free = 0 (last)
li t0, 4
slli t0, t0, 60
#ifdef VA_CORRUPT
ori t0, t0, 1                  # not an exact FREE record
#endif
sd t0, 8(t1)
#endif
REC(6, t1)
sd x0, 0(t1)
sd x0, 8(t1)

MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)   # node 2

#ifndef VA_FLAG_OFF
MINT(s7, CAP_TYPE_NONLIN, pt_root, pt_end, CAP_PERM_RW)
lla t0, pt_l0
lla t1, pt_root
sub a7, t0, t1
lla t0, pt_l1
sub s9, t0, t1
li t1, 0x80000000
li t2, 0
1:
srli t3, t1, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, a7
CINCOFFSET(t0, s7, t4)
sd t3, 0(t0)
li t4, 0x1000
add t1, t1, t4
addi t2, t2, 1
li t4, 512
bne t2, t4, 1b
lla t1, pt_l0
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
CINCOFFSET(t0, s7, s9)
sd t1, 0(t0)
li t2, 1
2:
slli t3, t2, 21
li t4, 0x80000000
add t3, t3, t4
srli t3, t3, 12
slli t3, t3, 10
ori t3, t3, PTE_LEAF_RWX
slli t4, t2, 3
add t4, t4, s9
CINCOFFSET(t0, s7, t4)
sd t3, 0(t0)
addi t2, t2, 1
li t4, 512
bne t2, t4, 2b
lla t1, pt_l1
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_PTR
li t4, 16
CINCOFFSET(t0, s7, t4)
sd t1, 0(t0)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
lla t0, tbl
csrw SREVROOT, t0
csrr t1, SREVROOT
sub t1, t1, t0
CAPPRINT(t1)                   # 0
#endif

#ifdef VA_CSRET
# The CSRETIRE case: step 3's retire-OPERAND push, and the retire handler's generation gate. Runs at M with
# the table namespace set (csupctl bit 3, as va-csmint-emu does), before any context is entered, so the
# table is in the known state this file built: free list 5 -> 4, count 2, next-unused 6, both generation 0.
#
# EVERY VALUE HERE IS DERIVED from the allocator's own order, not measured, and a mismatch is as likely to
# be a wrong derivation as a defect -- read it that way on the first run. CSNODE allocates twice, "the
# child ... first ..., the ancestor the second" (capstone_rev_node.anvil:710), and rd takes the ANCESTOR.
csrsi CSUPCTL, 8
CSNODE(s4, x26)                # pair: child = pop 5 = {1,5}, ancestor = pop 4 = {1,4}
CAPPRINT(s4)                   # 0x10004: the ancestor id, generation 1 at index 4
CSRETIRE(t1, s4)               # retire the ancestor: pushes the OPERAND (4) and its RUN (5)
CAPPRINT(t1)                   # 0: accepted
CSNODE(s5, x26)                # pops them back: child {2,5}, ancestor {2,4}
CAPPRINT(s5)                   # 0x20004: generation 2 at index 4 -- only a push-then-pop gives this
CSRETIRE(t1, s4)               # THE STALE RETIRE: id {1,4} against a record at generation 2
CAPPRINT(t1)                   # 1: REFUSED, and the fresh pair below must be untouched
ld t3, 0(x26)                  # a load through the re-noded fresh owner must not fault
CAPPRINT(t3)                   # 0: data is zeroed; the ABSENCE of a fault is the reading
csrci CSUPCTL, 8
ld t2, 16(s8)
CAPPRINT(t2)                   # 0: the free list is empty again, both records re-popped
ld t2, 24(s8)
CAPPRINT(t2)                   # 4: four allocations, two per CSNODE; the pushes do not count
REC_READ(4)                    # 3 0 0: the ancestor, VALID and LINEAR, parent 0, depth 0
REC_GEN(4)                     # 2
REC_READ(5)                    # 3 4 1: the child, VALID and LINEAR, parent 4, depth 1
REC_GEN(5)                     # 2
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)
#endif

li t0, 0x100000
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, VA_CTL_ARM

CAPCREATE(a2); li a0, CAP_TYPE_LIN; lla a3, stub0; lla a4, stub0_end; li a5, CAP_PERM_RWX
CAPTYPE(a2, a0); CAPBOUND(a2, a3, a4); CAPPERM(a2, a5)
CAPCREATE(a1); li a0, CAP_TYPE_LIN; lla a3, seal0; lla a4, seal0_end; li a5, CAP_PERM_RWX
CAPTYPE(a1, a0); CAPBOUND(a1, a3, a4); CAPPERM(a1, a5)
STC(a1, a2, 0); li a0, 0x0000000A00040000; sd a0, 48(a1); sd x0, 56(a1); sd x0, 64(a1); sd x0, 72(a1); sd x0, 80(a1)
SEAL(a3, a1); csrw mcause, x0; CSSUPERVISE(t3, a3, s5); addi s2, t3, 0; CAPPRINT(s2); CALL(a3, a3)
csrr t0, CSUPSTATUS
csrr t2, CSUPCAUSE
CAPPRINT(t2)                   # 0
ld t1, 0(s1)
CAPPRINT(t1)                   # 0
ld t1, 8(s1)
CAPPRINT(t1)                   # 64
ld t1, 16(s1)
CAPPRINT(t1)                   # 128
ld t1, 24(s1)
CAPPRINT(t1)                   # 0x19
CAPPRINT(t0)                   # 1

# ---------------------------------------------------------------- the header and the records after the domain
ld t2, 0(s8)
CAPPRINT(t2)                   # 0x8000000000000400 (both: capacity | RECYCLE survives the slot-0 write; 0x...107 with VA_EXHAUST)
ld t2, 8(s8)
CAPPRINT(t2)                   # 7 (flag on) / 6 (off)
ld t2, 16(s8)
CAPPRINT(t2)                   # 0 (on) / 0x200000005 (off)
ld t2, 24(s8)
CAPPRINT(t2)                   # 3 (on) / 0 (off)
REC_READ(5)                    # 1 2 1 (on) / 4 0 0 (off)
REC_READ(4)                    # 1 2 1 (on) / 4 0 0 (off)
REC(6, t1)
ld t2, 0(t1)
li t4, 0xffff
and t3, t2, t4
CAPPRINT(t3)                   # 0 (both): record 6's prev, nulled by the retirement (QEMU cap_rev_retire) / untouched
REC_READ(6)                    # 0 2 1 (on: retired, not free) / 0 0 0 (off)
#ifdef VA_STALE
# item 6's authority readings. The two probes are in the s1 slots above: slot 0 is the STALE validity
# query and must read 0, slot 1 is the FRESH query on the SAME record and must read 1. These two lines
# are what exclude the confound: record 5 must read VALID here, so the stale refusal cannot be a
# valid == 0 refusal, and its generation must read 2, so the fresh id and the record agree while the
# stale id does not.
# VOID: both probes reading 0 means the fresh owner is refused too and the arm measures nothing about the
# generation. Both reading 1 means the generation is not consulted. Record 5 reading anything but VALID
# means the confound is back and the refusal says nothing.
REC_READ(5)                    # 1 2 1: VALID at the moment of the stale probe
REC_GEN(5)                     # 2: the record's generation, which the fresh id matches and the stale does not
#endif
#ifdef VA_RECYCLE
# item 6's witness, with its VOID condition stated beside it. Record 5 is popped, pushed and popped again,
# so its generation must read 2 -- a value NO software free list can produce and no bump can produce, only
# a hardware push between two pops. The header must read {count 1, head 4}: 5 was taken, given back, taken
# again, leaving 4 alone on the list.
# VOID: generation 1 with the header at 0 means the push never happened and the second derive popped 4
# instead; next-unused above moving off 6 means something bump-allocated. Either way the run says nothing
# about recycling, whatever the free bits read.
REC_READ(5)                    # 1 2 1: VALID again, same parent, same depth
REC_GEN(5)                     # 2: THE WITNESS -- pop, push, pop
#endif
#ifdef VA_SUBTREE
# item 6's subtree witness: the operand AND its junior run are both FREE, with their generations preserved
# for the next pop, and the header counts 2 again (5 pushed by the kill, 4 by the walk, so head 4).
# VOID/LEAK: record 5 FREE with record 4 reading 0 0 0 and the header at {count 1, head 5} is the operand
# push WITHOUT the walk push -- one record recycled per free and the subtree leaked.
REC_READ(5)                    # 4 0 0: the child, FREE with every other field zero
REC_GEN(5)                     # 1: the generation the pop composed, preserved across the push
REC_READ(4)                    # 4 0 0: the grandchild, FREE -- the walk pushed the junior run
REC_GEN(4)                     # 1
#endif
csrr t0, mcause
CAPPRINT(t0)                   # 0
CAP_PASS(s11)

trap_handler:
  csrr t0, mcause
  CAPPRINT(t0)
  CAP_PASS(s11)

.align 6
stub0:
  MOVC(s3, x1)
#ifdef VA_RECYCLE
  # item 6's RECYCLE case, and the witness is a generation of TWO, which is unreachable without a hardware
  # push. The table this arm builds already has a SOFTWARE free list, 5 -> 4 with record 5 written as
  # (FREE << 60), i.e. generation 0. So: the first derive POPS 5 and composes generation 1; the crevoke
  # PUSHES 5 back with that generation preserved (item 6); the second derive pops 5 AGAIN and composes
  # generation 2. Only the push in the middle can produce a 2.
  li t0, 64
  CDERIVE(s4, x26, t0)         # pops 5 from the software free list -> {1, 5}
  CREVOKE(x26, s4)             # item 6 pushes 5 back, generation 1 preserved in the free record
  CDERIVE(s5, x26, t0)         # pops 5 AGAIN -> {2, 5}
  LCC(t1, s5, x2)
  lla t2, data
  sub t1, t1, t2
  sd t1, 0(s1)                 # 0: the twice-recycled capability is usable at the asked-for place
  ld t3, 0(s5)                 # a load THROUGH it must pass; it is the ABSENCE of a fault that counts
  sd t3, 8(s1)                 # 0: data is zeroed
  li t0, 0x19
  sd t0, 16(s1)                # the marker, so a truncated run is visible
  sd x0, 24(s1)
#elif defined(VA_STALE)
  # item 6's AUTHORITY property: the stale reference is REFUSED and the fresh owner of the SAME record is
  # ACCEPTED, in one run, differing only in the generation carried in the id.
  li t0, 64
  CDERIVE(s4, x26, t0)         # pops 5 from the software free list -> {1, 5}
  CREVOKE(x26, s4)             # pushes 5 back; s4 KEEPS the now-stale {1, 5} -- revocation never touches
                               # the register file, which is the whole reason a stale reference can exist
  CDERIVE(s5, x26, t0)         # pops 5 AGAIN -> {2, 5}: the SAME record, VALID, one generation up
  LCC(t1, s4, x0)              # THE STALE PROBE, validity query: must read 0
  sd t1, 0(s1)
  LCC(t1, s5, x0)              # THE FRESH PROBE on the SAME record: must read 1
  sd t1, 8(s1)
  ld t3, 0(s5)                 # and the fresh owner is usable: a load through it must not fault
  sd t3, 16(s1)
  sd x0, 24(s1)
#elif defined(VA_SUBTREE)
  # item 6's SUBTREE case, which is the shape lt1's pymalloc and pg-pools actually free: a CREVOKE of a
  # child that has a descendant. The kill pushes the OPERAND and the walk pushes its JUNIOR RUN, so both
  # records must come back FREE. Both are popped from the software free list first, so both carry
  # generation 1 when they are pushed, which is what the pushed records must preserve.
  li t0, 64
  CDERIVE(s4, x26, t0)         # child: pops 5 -> {1, 5}, depth 1
  CDERIVE(s5, s4, t0)          # grandchild FROM the child: pops 4 -> {1, 4}, depth 2, in the child's run
  LCC(t1, s5, x2)
  lla t2, data
  sub t1, t1, t2
  sd t1, 0(s1)                 # 0: the grandchild exists before the revoke
  CREVOKE(x26, s4)             # kills the child AND walks its junior run: both must be pushed
  sd x0, 8(s1)
  li t0, 0x19
  sd t0, 16(s1)
  sd x0, 24(s1)
#else
  li t0, 64
  CDERIVE(s4, x26, t0)         # child A at data+0
  LCC(t1, s4, x2)              # its cursor
  lla t2, data
  sub t1, t1, t2
  sd t1, 0(s1)                 # 0
  CINCOFFSET(s6, x26, t0)      # data+64
  CDERIVE(s4, s6, t0)          # child B at data+64 (derived from the root's cursor moved: still the root's node)
  LCC(t1, s4, x2)
  sub t1, t1, t2
  sd t1, 8(s1)                 # 64
  li t0, 128
  CINCOFFSET(s6, x26, t0)
  li t0, 64
  CDERIVE(s4, s6, t0)          # child C at data+128
  LCC(t1, s4, x2)
  sub t1, t1, t2
  sd t1, 16(s1)                # 128
  CREVOKE(x26, s4)             # child C retired
  li t0, 0x19
  sd t0, 24(s1)
#endif
  MOVC(ra, s3)
  la a1, stub0
  RETURN(ra, a1, x0)
stub0_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
scratch:
  .zero 64
scratch_end:
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
pt_root:
  .zero 4096
.align 12
pt_l1:
  .zero 4096
.align 12
pt_l0:
  .zero 4096
pt_end:
.align 6
seal0:
  .zero 2048
seal0_end:
  TEST_DATA
RVTEST_DATA_END
