# See LICENSE for license details.

#*****************************************************************************
# movc-zero-self-clobber.S
#-----------------------------------------------------------------------------
#
# Does a `movc rd, zero`-sourced store into a BANK 1 slot write the capability
# METADATA into THAT SLOT, instead of the store data?
#
# WHY THIS IS A DIFFERENT TEST FROM scalar-store-movc-zero.S, WHICH ALREADY PASSES
# THE OTHER WAY. That test showed a NEIGHBOUR being zeroed while the store's own
# slot read back CORRECT. The board shows the opposite: on 2026-08-08 an fdreg
# stage-4 rung returned 0x08000A31 = 0x08000000 + 2609, i.e. its accumulator -- the
# store's OWN slot, at row offset 8 -- held compress_cap(NULL) plus the correct
# count, on three boots at two base VAs, with a green control each time and QEMU
# computing 2609 for the same binary. Rebuilding it at -O1, which keeps the
# accumulator in a register, is clean; emitting `addi` instead of `movc` for the
# initialiser is clean.
#
# The proposed mechanism is wt_dcache_mem.sv:158 --
#     bank_wdata[k][j] = ((st_wr_cap && k==1) ? wr_user_i : wr_data_i)
# -- i.e. bank 1 receives wr_user_i INSTEAD of wr_data_i when the store is
# classified as a capability store by VALUE (st_wr_cap = |wr_user_i, :138), which a
# `movc rd, zero` result triggers because compress_cap of a null capability is
# 0x08000000 (ariane_pkg.sv:754-772).
#
# THAT PREDICTION IS WHAT THIS TEST CHECKS, AND IT IS NOT YET CONFIRMED ANYWHERE.
# The board result is consistent with it, but the value's provenance is the only
# evidence that hardware wrote it; the path has been READ, never traced. If the
# simulation does not reproduce this, the mechanism is NOT established and must not
# be handed over as one -- the honest statement would be that the board produces a
# hardware-only encoding in a slot the program only writes zero to, cause unknown.
#
# GEOMETRY, one 16-byte row at buf_start+0x10, mirroring the failing rung:
#     +0x10  bank 0, lanes 0-3     witness A
#     +0x14  bank 0, lanes 4-7     witness B
#     +0x18  bank 1, lanes 0-3     ACCUMULATOR   <- movc-zero init, then +1 x N
#     +0x1c  bank 1, lanes 4-7     row-mate, RMW'd so the row is not quiet
#
# The accumulator is initialised by a `movc`-sourced store and then incremented N
# times with ordinary lw/addiw/sw. It is read back RAW.
#
#   accumulator == N            -> NO self-clobber. The simulation does not
#                                  reproduce the board, and the mechanism is
#                                  unconfirmed.
#   accumulator == 0x08000000+N -> self-clobber REPRODUCED; bank 1 took wr_user_i.
#
# Exit codes:
#   1  accumulator holds 0x08000000 + N   -- THE DEFECT, reproduced in simulation
#   2  accumulator holds something else entirely
#   3  a witness was modified (the splash form, i.e. the OTHER manifestation)
#

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SENT_A 0x0A0A0A0A
#define SENT_B 0x0B0B0B0B
#define NITER  64

.section .text.init
.globl _start
_start:

        CAPCREATE(a1)
        li    a5, CAP_TYPE_LIN
        lla   t2, buf_start
        lla   t3, buf_end
        CAPTYPE(a1, a5)
        CAPBOUND(a1, t2, t3)
        li    a5, CAP_PERM_RWX
        CAPPERM(a1, a5)

        lla   t2, buf_start
        SCC(a3, a1, t2)

        li    t0, SENT_A
        sw    t0, 0x10(a3)
        li    t0, SENT_B
        sw    t0, 0x14(a3)
        sw    x0, 0x1c(a3)

        CHK_START

        # ---- THE STORE UNDER TEST: init the bank-1 accumulator from movc-zero ----
        MOVC(a4, x0)
        sw    a4, 0x18(a3)

        # ---- ordinary RMW traffic, exactly as the rung does ----
        li    a6, NITER
1:      lw    t1, 0x18(a3)          # accumulator++
        addiw t1, t1, 1
        sw    t1, 0x18(a3)
        lw    t2, 0x1c(a3)          # row-mate++, keeps the row active
        addiw t2, t2, 1
        sw    t2, 0x1c(a3)
        addi  a6, a6, -1
        bnez  a6, 1b

        CHK_END

        lw    t1, 0x18(a3)          # accumulator, RAW
        lw    t3, 0x10(a3)          # witness A
        lw    t4, 0x14(a3)          # witness B

        # witnesses first: a splash would otherwise be reported as a self-clobber
        li    t0, SENT_A
        bne   t3, t0, fail3
        li    t0, SENT_B
        bne   t4, t0, fail3

        # the metadata form: 0x08000000 + NITER
        li    t0, 0x08000000
        addi  t0, t0, NITER
        beq   t1, t0, fail1

        # the clean form
        li    t0, NITER
        bne   t1, t0, fail2

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
