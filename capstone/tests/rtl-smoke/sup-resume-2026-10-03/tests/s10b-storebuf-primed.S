# See LICENSE for license details.

#*****************************************************************************
# s10b-storebuf-primed.S   (primed variant -- see the RESID macro)
#-----------------------------------------------------------------------------
#
# THE SECOND ROUTE TO S-10, and the one the write-buffer fix cannot reach.
#
# S-10 proper: a plain store sits in the WRITE BUFFER at word 1 of a granule, a granule-aligned
# LDC compares word 0, misses it, and falls through to the stale cap_tag_q. Fixed in
# wt_dcache_mem.sv by looking the tag up per GRANULE.
#
# S-10b: the store never reaches the write buffer at all. It is still in the store buffer's
# commit or speculative queue, where the ONLY interlock is page_offset_matches_o -- and that
# compared at WORD granularity too (store_buffer.sv:279/287/293, [11:3]). Same mistake, one
# pipeline stage earlier. The load is not stalled, the write-buffer fix has nothing to see
# because the store is not there yet, and the capability survives its own scrub.
#
# HOW THIS TEST CREATES THE CONDITION, which is the whole difficulty, and it is NOT what was
# tried first. In the natural sequence
#     sd x0, G+8 ; ldc G
# the store HAS reached the write buffer by the time the load's tag lookup happens -- address
# generation and translation take long enough. That is why s07-wbuf-forward-residual exercises
# the write-buffer route and went 8/16 -> 16/16 on the S-10 fix alone.
#
# FILLING THE WRITE BUFFER FIRST DOES NOT WORK EITHER, and it was measured rather than assumed:
# eight stores to eight distinct granules ahead of the scrub gave 8 traps of 8 on the PRE-FIX
# RTL, i.e. no defect at all, because the buffer drains continuously and had space again by the
# time the scrub was issued.
#
# *** AND NEITHER DOES THE SHAPE THIS TEST CURRENTLY HAS. READ THIS BEFORE CITING THE RESULT. ***
#
# THIS TEST DOES NOT CURRENTLY CREATE ITS CONDITION. Measured on PRE-FIX RTL (c3ca1b270, which
# carries S-07 and S-10 but not S-10b): 9 exceptions = 1 positive control + 8 legs ALL trapping,
# i.e. the tag was correctly cleared every time and the defect did not appear. A SUCCESS row for
# this test in a sweep therefore means "the condition was never created", NOT "the route is
# closed". It is committed because the shape is worth keeping and the next attempt should start
# from it, not because it covers anything.
#
# THE IDEA THAT WAS TRIED, AND WHY IT FAILS. Leave the seeding STC RESIDENT -- do not drain -- so
# that the scrubbing store to the granule's other word is refused allocation by gran_hazard
# (wt_dcache_wbuffer.sv), which is what the S-07 fix exists to do. The scrub would then have
# nowhere to go and would sit in the store buffer's commit queue, where page_offset_matches_o is
# the only thing between it and a load. In practice the STC DRAINS BEFORE THE SCRUB ARRIVES, the
# refusal never fires, the scrub allocates normally, and S-10's granule tag lookup catches it --
# which is why all eight legs trap.
#
# So the note below stands as a hypothesis about the mechanism and NOT as a description of what
# this binary does: if the window were opened, THE S-07 FIX IS WHAT WOULD MAKE THIS ROUTE
# REACHABLE, because its stall is the thing that parks a store in the store buffer. Closing one
# hole opens the path to another. Unobserved.
#
# WHAT A FUTURE ATTEMPT NEEDS: sustained pressure that keeps the STC's entry resident across the
# scrub, rather than a single STC that drains in the gap. Nothing here has achieved that yet.
#
# THE TAG ROUTE OF S-10b IS THEREFORE A CODE-READING ARGUMENT WITH NO REPRODUCTION. What IS
# demonstrated is the DATA route, and not by this test: untagged-ldc-stc-fixup changes its RVFI
# trace under the S-10b fix, where a load of an STC's high word returned 0x0 before and
# 0xfedcba9876543210 after.
#
# POLARITY IS INVERTED, as in s07-wbuf-forward-residual. A trap is the CORRECT outcome -- the
# granule's tag was cleared, the LDC returns a non-capability, the readback faults. Its ABSENCE
# is the defect.
#
#   8 traps / 8 legs   every leg saw the tag correctly cleared. THIS IS THE CURRENT RESULT ON
#                      BOTH FIXED AND PRE-FIX RTL, so it means the condition was not created --
#                      it does NOT mean the route is closed
#   0 traps            every leg was handed a live capability over memory it had scrubbed
#   n traps            the window is real and closed n times out of 8
#
# THE POSITIVE CONTROL RUNS FIRST AND IS MANDATORY, because the finding here is a ZERO and a
# detector that cannot fire would manufacture it.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc
.section .text.init
.globl _start
_start:

  li gp, 1
  csrwi mie, 0
  la t0, trap_handler
  csrw mtvec, t0
  la t0, trap_count
  sw x0, 0(t0)

#define MKCAP(CREG, LO, HI)   \
  CAPCREATE(CREG);            \
  li  a0, CAP_TYPE_NONLIN;    \
  lla a3, LO;                 \
  lla a4, HI;                 \
  li  a5, CAP_PERM_RWX;       \
  CAPTYPE(CREG, a0);          \
  CAPBOUND(CREG, a3, a4);     \
  CAPPERM(CREG, a5);

  # s1 = authority over the granule array; s2 = the capability VALUE stored into it.
  # NONLIN for both: a LINEAR store would fire the LDC linear-clear path and bring a second
  # tag-clearing writer into the experiment.
  MKCAP(s1, buf_start, buf_end)
  MKCAP(s2, val_start, val_end)
  MKCAP(s5, fill_start, fill_end)

# --------------------------------------------------------- positive control, FIRST
  li   t3, 0x1234
  LDC(t4, t3, 0)
  la   t0, trap_count
  lw   t1, 0(t0)
  li   t2, 1
  bne  t1, t2, fail_control
  sw   x0, 0(t0)                   /* reset the counter for the measurement */

# --------------------------------------------------------- the S-10b sequence
# NO SEEDING PASS AND NO DRAIN, deliberately. Each leg stores its capability and is immediately
# followed by the scrub, so the STC is still RESIDENT in the write buffer when the scrub asks to
# allocate -- which is the only state in which gran_hazard refuses it.

# THE ONE INSTRUCTION THAT IS DIFFERENT FROM s10b-storebuf-residual, and the whole point:
# `ld x0, OFF(s1)` PRIMES the line into L1 before the STC.
#
# Without it the LDC below is a first touch and MISSES, which routes it through the miss unit --
# and wt_dcache_missunit.sv:242 compares an incoming miss against in-flight write TXs at
# [PLEN-1:DCACHE_OFFSET_WIDTH] = [PLEN-1:4], i.e. GRANULE granularity, stalling the read
# outright (:495). By the time it is released the scrubbing store has allocated into the write
# buffer, S-10's granule tag lookup sees it, and the leg traps. That is why the unprimed test
# gets 8 traps of 8 and reproduces nothing.
#
# With the line primed the LDC HITS L1 and never enters the miss unit at all, so no collision
# check can fire. It then forwards the STC's ctag=1 out of the write buffer while the scrub sits
# invisible in the STORE buffer one stage upstream -- which is the S-10b tag route.
#define RESID(OFF)           \
  ld  x0, OFF(s1);           \
  STC(s1, s2, OFF);          \
  sd  x0, (OFF+8)(s1);       \
  LDC(t3, s1, OFF);          \
  LDC(t4, t3, 0);

  RESID(0)   RESID(16)  RESID(32)  RESID(48)
  RESID(64)  RESID(80)  RESID(96)  RESID(112)

# --------------------------------------------------------- report
# gp is written for readability only -- RVTEST_PASS overwrites TESTNUM, so the live measurement
# is the UNEXPECTED_OPERAND count in the .iss log, which is what the sweep records.
  la   t0, trap_count
  lw   t1, 0(t0)
  li   t2, 8
  beq  t1, t2, all_cleared
  sub  t2, t2, t1
  addi gp, t2, 200                 /* 200 + number of legs handed a live capability */
  j    done

all_cleared:
  li   gp, 1
  j    done

fail_control:
  li   gp, 999                     /* detector never fired -- run is VOID */

done:
  /* BOARD PORT (board lane, 2026-10-05): the only change from capstone-ariane 776d9d859. RVTEST_PASS writes
     tohost, which the board has no reader for; report the verdict through the board harness instead.
     #1 = trap_count after the measurement (the positive control's trap was reset): 8 = every leg's tag was
     cleared (the fix), 0 = every leg was handed a live capability (the resident defect). #2 = gp as above. */
  la   t0, trap_count
  lw   t1, 0(t0)
  CAPPRINT(t1)
  CAPPRINT(gp)
  CAP_PASS(s11)

  .align 2
trap_handler:
  la   t2, trap_count
  lw   t5, 0(t2)
  addi t5, t5, 1
  sw   t5, 0(t2)
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

  .data
RVTEST_DATA_BEGIN
.align 4
trap_count: .word 0
.align 6
buf_start:
  .zero 128
buf_end:
.align 6
fill_start:
  .zero 128
fill_end:
.align 6
val_start:
  .zero 64
val_end:
RVTEST_DATA_END
