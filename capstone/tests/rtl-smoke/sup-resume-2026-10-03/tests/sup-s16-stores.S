# Copied unchanged from capstone-ariane a8045365a (branch sup-call), verif/tests/custom/capstone/sup-s16-stores.S -- the RTL lane's S-16 directed test.
# See LICENSE for license details.

#*****************************************************************************
# sup-s16-stores.S  (S-16 on silicon: the supervised switch that never finishes, 2026-10-03/04)
#-----------------------------------------------------------------------------
#
# A burst of NSTORES scalar stores to distinct granules, then the FPGA monitor's swap-out tail (`ccsrrw x0, cscratch,
# sp; li sp, 0`) and an armed CALL whose domain RETURNs at once. The switcher's first SAVE write (id 3) must find room
# in the store buffer's commit queue (depth 4). If the burst's last stores are still queued when it arrives, the
# store unit's room check -- keyed on the kind of the PREVIOUS store, so it answers for the speculative queue -- lets
# the write through, the push lands in the full commit queue over its head (a committed store is lost), the count
# runs one ahead of the valid bits, and the 4th later write (id 7) starves forever: the switch never finishes. On the
# board (2026-10-04): 24 stores before the CALL hang within 16 resumes, 12 complete; the wedge reads switcher idx 7,
# store unit WAIT_STORE_READY, commit count 4, the data request valid and not acknowledged.
#
# ESCAPE (the board's C5f, 2026-10-04: the real monitor with a fence before every domcall still hung, on an ESCAPE):
# the domain runs ESC_BURSTS bursts of NSTORES back-to-back stores under a short QUANTUM, so the quantum escape's SAVE
# walk (the domain's registers into the seal) starts while the commit queue is full of the DOMAIN's own stores -- the
# same push over the head, entered from the escape side, which no fence before the CALL can cover. The monitor resumes
# until the domain RETURNs. Readings: csupstatus 1, the number of resumes (>= 1, or the quantum never landed in a
# burst and the arm is void), the marker 0x77, mcause 0.
#
# Readings (CAPPRINT, in order): csupstatus after the CALL = 1 (kind 0, the domain's RETURN); the domain's marker
# 0x77; mcause 0. A hang: no readings, the simulation times out, and SUP_TRACE prints `stbuf dom-switch push:
# cnt_q=4` followed by `dom-switch data request starved`. NSTORES 4 is the control.
#
# ITER rounds of burst + armed call (default 3): the first round warms the seal's line, so from the second round the CALL's
# own seal load hits L1 and the first SAVE write follows the last store's commit within a few cycles -- the board's
# geometry, where the DDR drain cannot empty the queue in that gap. With the seal cold (one round) only N >= 32
# reaches cnt_q=4 in simulation; with it warm EVERY arm hangs on the resident logic, N=4 and the fence arm included,
# at the RETURN switch of round 2: the CALL's own SAVE tail plus the domain's one store fill the queue, and that
# ordinary store resets the kind the room check is keyed on. On the fixed logic all arms complete.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0

#ifndef NSTORES
#define NSTORES 16
#endif
#ifndef ITER
#define ITER 3
#endif
#ifndef QUANTUM
#define QUANTUM 1000000          /* no quantum escape unless ESCAPE sets a short one */
#endif
#ifndef ESC_BURSTS
#define ESC_BURSTS 4
#endif

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

# s5: the monitor's private save area, LINEAR RWX, 1 KiB
CAPCREATE(s5)
li a0, CAP_TYPE_LIN
lla a3, savearea
lla a4, savearea_end
li a5, CAP_PERM_RWX
CAPTYPE(s5, a0)
CAPBOUND(s5, a3, a4)
CAPPERM(s5, a5)

# s8: NONLIN RW over the burst buffer (one 16-byte granule per store)
CAPCREATE(s8)
li a0, CAP_TYPE_NONLIN
lla a3, storebuf
lla a4, storebuf_end
li a5, CAP_PERM_RWX
CAPTYPE(s8, a0)
CAPBOUND(s8, a3, a4)
CAPPERM(s8, a5)

# s3: the result word
CAPCREATE(s3)
li a0, CAP_TYPE_LIN
lla a3, result
lla a4, result_end
li a5, CAP_PERM_RWX
CAPTYPE(s3, a0)
CAPBOUND(s3, a3, a4)
CAPPERM(s3, a5)
sd x0, 0(s3)

# the seal: code capability over the stub, LINEAR over the seal region, images, SEAL
CAPCREATE(a2)
li a0, CAP_TYPE_LIN
lla a3, stub
lla a4, stub_end
li a5, CAP_PERM_RWX
CAPTYPE(a2, a0)
CAPBOUND(a2, a3, a4)
CAPPERM(a2, a5)
CAPCREATE(a1)
li a0, CAP_TYPE_LIN
lla a3, seal
lla a4, seal_end
li a5, CAP_PERM_RWX
CAPTYPE(a1, a0)
CAPBOUND(a1, a3, a4)
CAPPERM(a1, a5)
STC(a1, a2, 0)
li a0, 0x0000000A00040000
sd a0, 48(a1)
sd x0, 56(a1)
sd x0, 64(a1)
sd x0, 72(a1)
sd x0, 80(a1)
SEAL(a3, a1)

li t0, QUANTUM
csrw CSUPQUANTUM, t0
csrwi CSUPCTL, 0
li s9, 0                       # the number of resumes (ESCAPE)
li x30, 0xA5
li s7, 0                       # the round counter
round:
MOVC(sp, s8)                   # sp holds a capability, as the monitor's does
CSSUPERVISE(t3, a3, s5)

# the burst: NSTORES stores to distinct granules, back to back
.set s16_off, 0
.rept NSTORES
  sd x30, s16_off(s8)
  .set s16_off, s16_off + 16
.endr

# the monitor's swap-out tail (its CCSRRW flushes and re-fetches, which gives the queue ~50 cycles to drain; NOCSR
# drops it so the first SAVE write follows the last store as closely as the pipeline allows), then the armed CALL.
# FENCE is the board lane's workaround: drain the store buffer before the CALL.
#ifndef NOCSR
CCSRRW(x0, CCSR_CSCRATCH, sp)
li sp, 0
#endif
#ifdef FENCE
fence
#endif
CALL(a3, a3)
#ifdef ESCAPE
supervise_loop:
  csrr t0, CSUPSTATUS
  andi t1, t0, 6
  li t2, 2
  bne t1, t2, escapes_done     # kind != 1: the domain returned (0) or faulted (2)
  addi s9, s9, 1
  csrwi CSUPCTL, 1             # resume
  CSSUPERVISE(t3, a3, s5)
  CALL(a3, a3)
  j supervise_loop
escapes_done:
#else
csrr t0, CSUPSTATUS
#endif
CAPPRINT(t0)                   # 1: the domain returned (kind 0), once per round
addi s7, s7, 1
li t1, ITER
bne s7, t1, round
#ifdef ESCAPE
addi t0, s9, 0
CAPPRINT(t0)                   # the number of resumes: >= 1 or the arm is void
#endif
ld t0, 0(s3)
CAPPRINT(t0)                   # 0x77
csrr t0, mcause
CAPPRINT(t0)                   # 0: no trap was taken
CAP_PASS(s11)

.align 4
trap_handler:
  csrr s4, mcause
  CAPPRINT(s4)                 # a trap ends the test: its cause is the last reading
  CAP_PASS(s11)

.align 4
stub:
#ifdef ESCAPE
  li x20, 0
  li x21, ESC_BURSTS
1:
  .set s16_doff, 0
  .rept NSTORES
    sd x30, s16_doff(s8)       # the domain's burst, back to back (x30 and s8 are inherited)
    .set s16_doff, s16_doff + 16
  .endr
  addi x20, x20, 1
  bne x20, x21, 1b
#endif
  li a1, 0x77
  sd a1, 0(s3)
  la a1, stub
  RETURN(ra, a1, x0)
stub_end:

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN
.align 6
savearea:
  .zero 1024
savearea_end:
.align 6
result:
  .zero 64
result_end:
.align 6
seal:
  .zero 2048
seal_end:
.align 6
storebuf:
  .zero 1024
storebuf_end:
  TEST_DATA
RVTEST_DATA_END
