# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# objb-perf.S -- what the per-access lifetime check COSTS on bitstream 3 (e7af48457), measured against the ObjBounds
#                variant (CAPSTONE_NO_LIFETIME_CHECK), shape by shape (2026-10-30)
#-----------------------------------------------------------------------------
#
# The check on e7af48457 is a same-cycle lookup of a 4-way x 64-set cache of node ids (load_store_unit.sv, rvc_*),
# filled by watching the rev-node unit's own writes and reads. A HIT costs nothing. A MISS marks the access for
# replay, a probe reads the node's record through D-cache port 2, and commit flushes and re-fetches the access once
# the probe resolves. So the cost is set by how often accesses MISS, and every shape below is built to say how often:
#   SHAPE 1  one capability, 4,096 loads through it                      -> every access hits
#   SHAPE 2  a pointer chase over 64 nodes (ldc of the next capability)  -> one node per set, every access hits
#   SHAPE 3  the same chase over 1,024 nodes                             -> 16 nodes per set, 4 ways: every access misses
#   SHAPE 4  the 64-node chase right after a namespace flush (csupctl bit 2, what a context entry and escape raise)
#            -> the first pass misses once per node, the second pass hits
#   SHAPE 5  ObjLife-style churn: 1,024 objects each on its own cderive'd node, every odd one freed (crevoke) and
#            re-allocated, then two passes over all 1,024 live objects (ldc from the table, ld of the payload)
#            -> the churn's own accesses hit (fresh nodes and the table's node), the passes miss on every object
#   SHAPE 6  1,024 ldc through one capability;  SHAPE 7  1,024 stc through one capability
# CORRECTION after the first run (c7b020d7b / 3c5c43996, 2026-10-30): the cache arrows above hold only for PLAIN
# loads and stores (shapes 1 and 5). LDC and STC are DYN-unit ops and never reach the LSU's cache: the DYN unit asks
# the rev-node unit for rs1's node validity on every one of them (capstone_dyn_unit.anvil LDC/STC), uncached. So the
# chases (2-4) measure that query, and they read equal on both trees until site (e) (ex_stage.sv) answers it.
# Each shape prints, in order: the measured window's mcycle delta, its minstret delta (the control: equal on both
# trees, since a replay re-executes but retires once), and a value that must be equal on both trees (the chase's
# final cursor, or the payload sum). The cost of the check is the mcycle difference between the two trees; the
# number of record reads is the count of "R43 probe-sent" lines with +define+R43_TRACE (base: about one per miss,
# ObjBounds: none). Readings are under the simulated memory delay (S12_MEM_DELAY=12), so the PATTERN is the
# prediction (equal / higher on the base, per shape), not the cycle values.
# Registers: s2 the heap capability (NONLIN, all permissions), s4/s10 the trap handler's cause/count, t0 scratch
# that the handler clobbers. No access in this arm is expected to trap on either tree; the last two prints are the
# trap count and the last cause, 0 and 0.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#ifndef SHAPE
#define SHAPE 1
#endif
#define OBJ 64
#define NOBJ_BIG 1024
#define TBL_OFF (NOBJ_BIG * OBJ)

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s4, 0
  li   s10, 0

  lla  a0, _start
  lla  a1, _end_of_text
  CAPENTER(a0, a1)

  CAPCREATE(s2)
  li   t1, CAP_TYPE_NONLIN
  lla  a2, heap
  lla  a3, heap_end
  CAPTYPE(s2, t1)
  CAPBOUND(s2, a2, a3)
  li   t1, CAP_PERM_ALL
  CAPPERM(s2, t1)

#if SHAPE == 1
# ---------------------------------------------------------------- one capability, 512 x 8 loads
  li   t0, OBJ
  CDERIVE(s3, s2, t0)
  li   t1, 0x5a
  sd   t1, 0(s3)
  ld   t3, 0(s3)                 # warm
  csrr s5, mcycle
  csrr s6, minstret
  li   t2, 512
1:
  ld   t3, 0(s3)
  ld   t3, 8(s3)
  ld   t3, 16(s3)
  ld   t3, 24(s3)
  ld   t3, 32(s3)
  ld   t3, 40(s3)
  ld   t3, 48(s3)
  ld   t3, 56(s3)
  addi t2, t2, -1
  bnez t2, 1b
  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(s7)                   # cycles
  CAPPRINT(s8)                   # instructions (control)
  ld   t3, 0(s3)
  CAPPRINT(t3)                   # value: 0x5a
#endif

#if SHAPE == 2 || SHAPE == 3 || SHAPE == 4
#if SHAPE == 3
#define NRING NOBJ_BIG
#define PASSES 2
#elif SHAPE == 2
#define NRING 64
#define PASSES 16
#else
#define NRING 64
#define PASSES 1
#endif
# ---------------------------------------------------------------- a ring of NRING objects, object i holds the capability of i+1
  li   t0, OBJ
  CDERIVE(s7, s2, t0)            # object 0, the ring's head
  MOVC(s8, s7)                   # prev (NONLIN: the copy keeps s7)
  li   s5, 0                     # byte offset of the current object
  li   s6, NRING - 1
2:
  addi s5, s5, OBJ
  CINCOFFSET(t4, s2, s5)
  li   t0, OBJ
  CDERIVE(t5, t4, t0)
  STC(s8, t5, 0)                 # prev[0] = this
  MOVC(s8, t5)
  addi s6, s6, -1
  bnez s6, 2b
  STC(s8, s7, 0)                 # last[0] = head
  MOVC(s9, s7)
  li   t2, NRING                 # warm pass
3:
  LDC(s9, s9, 0)
  addi t2, t2, -1
  bnez t2, 3b
#if SHAPE == 4
  csrwi 0x7c4, 4                 # csupctl bit 2: the namespace flush, the revocation cache empties
#endif
  csrr s5, mcycle
  csrr s6, minstret
  li   t2, NRING * PASSES
4:
  LDC(s9, s9, 0)
  addi t2, t2, -1
  bnez t2, 4b
  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(s7)                   # cycles
  CAPPRINT(s8)                   # instructions (control)
#if SHAPE == 4
  csrr s5, mcycle                # the second pass after the flush: resident again
  csrr s6, minstret
  li   t2, NRING
5:
  LDC(s9, s9, 0)
  addi t2, t2, -1
  bnez t2, 5b
  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(s7)                   # cycles, second pass
  CAPPRINT(s8)                   # instructions, second pass
#endif
  CAPPRINT(s9)                   # value: the cursor the chase ends on (the head, heap + 0)
#endif

#if SHAPE == 5
# ---------------------------------------------------------------- ObjLife-style churn over 1,024 objects
  li   t0, TBL_OFF
  CINCOFFSET(t4, s2, t0)
  li   t0, NOBJ_BIG * 16
  CDERIVE(s3, t4, t0)            # the object table: one node, 16 bytes a slot
  csrr s11, mcycle
  li   s5, 0                     # i
6:                               # allocate every object, payload i at +8, its capability in slot i
  slli t1, s5, 6
  CINCOFFSET(t4, s2, t1)
  li   t0, OBJ
  CDERIVE(t5, t4, t0)
  sd   s5, 8(t5)
  slli t1, s5, 4
  CINCOFFSET(t3, s3, t1)
  STC(t3, t5, 0)
  addi s5, s5, 1
  li   t1, NOBJ_BIG
  blt  s5, t1, 6b
  li   s5, 1
7:                               # free every odd object
  slli t1, s5, 4
  CINCOFFSET(t3, s3, t1)
  LDC(t5, t3, 0)
  CREVOKE(s2, t5)
  addi s5, s5, 2
  li   t1, NOBJ_BIG
  blt  s5, t1, 7b
  li   s5, 1
8:                               # re-allocate them: fresh nodes (recycled indices), payload i again
  slli t1, s5, 6
  CINCOFFSET(t4, s2, t1)
  li   t0, OBJ
  CDERIVE(t5, t4, t0)
  sd   s5, 8(t5)
  slli t1, s5, 4
  CINCOFFSET(t3, s3, t1)
  STC(t3, t5, 0)
  addi s5, s5, 2
  li   t1, NOBJ_BIG
  blt  s5, t1, 8b
  csrr s7, mcycle
  sub  s7, s7, s11
  CAPPRINT(s7)                   # cycles, the churn (allocate, free, re-allocate)
  csrr s5, mcycle
  csrr s6, minstret
  li   s9, 0                     # payload sum
  li   s11, 2                    # passes
9:
  li   s8, 0
10:
  slli t1, s8, 4
  CINCOFFSET(t3, s3, t1)
  LDC(t5, t3, 0)
  ld   t6, 8(t5)
  add  s9, s9, t6
  addi s8, s8, 1
  li   t1, NOBJ_BIG
  blt  s8, t1, 10b
  addi s11, s11, -1
  bnez s11, 9b
  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(s7)                   # cycles, the two passes over the live objects
  CAPPRINT(s8)                   # instructions (control)
  CAPPRINT(s9)                   # value: 2 x (0 + ... + 1023) = 1047552
#endif

#if SHAPE == 6 || SHAPE == 7
# ---------------------------------------------------------------- one capability, 128 x 8 ldc (6) or stc (7)
# Added after the first A/B: LDC and STC are DYN-unit ops, which ask the rev-node unit for rs1's node validity on
# EVERY access (capstone_dyn_unit.anvil LDC/STC), uncached, so shapes 2-4 measured that query, not the LSU cache.
# These two isolate it on a single node: the base pays the query per op, ObjBounds pays only the channel handshake
# that answers it (ex_stage.sv), and the difference is the query's cost on a node that never leaves the caches.
  li   t0, OBJ
  CDERIVE(s3, s2, t0)
  STC(s3, s3, 0)                 # slot 0 holds a capability (s3 itself; NONLIN, so stc keeps the register)
  LDC(t5, s3, 0)                 # warm
  csrr s5, mcycle
  csrr s6, minstret
  li   t2, 128
11:
#if SHAPE == 6
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
  LDC(t5, s3, 0)
#else
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
  STC(s3, s3, 16)
#endif
  addi t2, t2, -1
  bnez t2, 11b
  csrr s7, mcycle
  csrr s8, minstret
  sub  s7, s7, s5
  sub  s8, s8, s6
  CAPPRINT(s7)                   # cycles, 1,024 ops
  CAPPRINT(s8)                   # instructions (control)
  LDC(t5, s3, 16)
  CAPPRINT(t5)                   # value: the cursor stored at +16 (7) or 0 (6, nothing stored there)
#endif

  CAPPRINT(s10)                  # trap count: 0
  CAPPRINT(s4)                   # last cause: 0
  CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  addi s10, s10, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 14                        # 16 KiB: every derived window (64 B objects, the 16 KiB table, the 80 KiB heap) is exact
heap:
  .zero (NOBJ_BIG * OBJ + NOBJ_BIG * 16)
heap_end:
RVTEST_DATA_END
