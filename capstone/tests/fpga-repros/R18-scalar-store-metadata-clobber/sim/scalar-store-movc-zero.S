# See LICENSE for license details.

#*****************************************************************************
# scalar-store-movc-zero.S
#-----------------------------------------------------------------------------
#
# Does a plain scalar store whose DATA REGISTER WAS PRODUCED BY `movc rd, zero`
# corrupt a DIFFERENT scalar in the same 16-byte cache row?
#
# WHY THIS TEST EXISTS, AND WHY IT IS NOT A DUPLICATE OF
# scalar-store-cap-operand.S. That test puts a REAL, VALID capability in the
# store's data register and PASSES -- no misclassification, no dual-bank write.
# On the FPGA, however, a store whose data register came from `movc rd, zero`
# DOES corrupt a row neighbour, established 2026-08-08 by a four-way control:
#
#   reset store                              outer loop   victim
#   movc a0, zero ; sw                       short        9    DAMAGED
#   movc a0, zero ; sw  + 2 nops             padded       9    DAMAGED
#   ldc/lw ; sw    (stores the VALUE ZERO)   padded       576  clean
#   lui/addi ; sw  (stores a nonzero)        padded       576  clean
#
# so the trigger is the capability METADATA on the data register, not the stored
# value, not the store count, and not the loop length. The two directed tests
# therefore differ in exactly the way that matters: a real capability's shadow
# versus a null capability's, whose compressed form is 0x08000000
# (compress_cap cursorless branch, ariane_pkg.sv:754-772).
#
# That difference is also an UNRESOLVED TENSION worth stating plainly, because it
# is the reason this test is interesting rather than routine. The dcache
# classifies a store as a capability store by the VALUE of the metadata bits
# (wt_dcache_mem.sv:138, st_wr_cap = |wr_user_i), so a real capability ought to be
# misclassified at least as readily as a null one -- yet the real-capability test
# passes and the board fails. Either the real-capability test never gets metadata
# onto the write-user sideband, or the two take different paths. This test is the
# cheap way to find out, because it runs in simulation where every load's returned
# value is visible in the RVFI trace.
#
# THE GEOMETRY IS THE BOARD'S, not a guess. One 16-byte row at buf_start+0x10:
#
#     +0x10  bank 0, lanes 0-3     witness A
#     +0x14  bank 0, lanes 4-7     witness B
#     +0x18  bank 1, lanes 0-3     ROW-MATE, reset by the store under test
#     +0x1c  bank 1, lanes 4-7     VICTIM, only ever incremented
#
# The failing board arm has exactly this shape: the row-mate is reset once per
# outer pass by a movc-zero-sourced store and incremented once per inner
# iteration, and the victim is incremented once per inner iteration and never
# reset. The loop below reproduces it at the board's trip counts, 64 x 9.
#
# EXPECTED, if the board effect is an RTL property reachable from bare metal:
#     the victim reads back LESS than 576 -- it is silently zeroed part-way.
# EXPECTED, if it is not reachable this way:
#     everything reads back exactly, and the test PASSES. That is a real result
#     too: it says the domain context (capenter, a monitor-carved stack) is part
#     of the trigger, which narrows the search considerably.
#
# Do NOT read a pass here as exonerating the hardware. Six directed tests have
# already failed to reproduce R-18 and the effect is nonetheless reproducible on
# silicon on demand.
#
# The exit code names what happened:
#   1  the victim lost increments -- THE DEFECT, reproduced in simulation
#   2  the row-mate lost increments instead
#   3  witness A or B was modified
#

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SENT_A 0x0A0A0A0A
#define SENT_B 0x0B0B0B0B
#define OUTER  64
#define INNER  9

.section .text.init
.globl _start
_start:

        # a1: linear RWX capability over the buffer -- the base authority
        CAPCREATE(a1)
        li    a5, CAP_TYPE_LIN
        lla   t2, buf_start
        lla   t3, buf_end
        CAPTYPE(a1, a5)
        CAPBOUND(a1, t2, t3)
        li    a5, CAP_PERM_RWX
        CAPPERM(a1, a5)

        # a3: cursor at buf_start, base for all plain accesses
        lla   t2, buf_start
        SCC(a3, a1, t2)

        # seed the two witnesses; zero the row-mate and the victim
        li    t0, SENT_A
        sw    t0, 0x10(a3)
        li    t0, SENT_B
        sw    t0, 0x14(a3)
        sw    x0, 0x18(a3)
        sw    x0, 0x1c(a3)

        CHK_START

        li    a6, OUTER                 # outer counter
outer_loop:
        # ---- THE STORE UNDER TEST ----
        # a4's metadata shadow is written by a capstone FLU op with a null
        # capability result, so it holds the compressed NULL form. The store
        # itself is an ordinary `sw` into bank 1, lanes 0-3.
        MOVC(a4, x0)
        sw    a4, 0x18(a3)

        li    a7, INNER                 # inner counter
inner_loop:
        lw    t1, 0x18(a3)              # row-mate++
        addiw t1, t1, 1
        sw    t1, 0x18(a3)

        lw    t2, 0x1c(a3)              # victim++  -- never reset
        addiw t2, t2, 1
        sw    t2, 0x1c(a3)

        addi  a7, a7, -1
        bnez  a7, inner_loop

        addi  a6, a6, -1
        bnez  a6, outer_loop

        CHK_END

        # ---- read everything back with plain loads ----
        lw    t1, 0x1c(a3)              # victim   -- expect OUTER*INNER = 576
        lw    t2, 0x18(a3)              # row-mate -- expect INNER = 9
        lw    t3, 0x10(a3)              # witness A
        lw    t4, 0x14(a3)              # witness B

        li    t0, OUTER*INNER
        bne   t1, t0, fail1

        li    t0, INNER
        bne   t2, t0, fail2

        li    t0, SENT_A
        bne   t3, t0, fail3
        li    t0, SENT_B
        bne   t4, t0, fail3

        RVTEST_PASS

fail1:  li TESTNUM, 1
        j fail
fail2:  li TESTNUM, 2
        j fail
fail3:  li TESTNUM, 3
fail:
        RVTEST_FAIL

        .data
RVTEST_DATA_BEGIN

        .align 6
buf_start:
        .dword 0, 0, 0, 0, 0, 0, 0, 0
buf_end:
        .dword 0

RVTEST_DATA_END
