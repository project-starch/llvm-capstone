# See LICENSE for license details.

#*****************************************************************************
# lsu-mmode-gate.S
#-----------------------------------------------------------------------------
#
# IS THE LSU'S CAPABILITY CHECK ON PLAIN LOADS LIVE WHEN ITS GATE IS SATISFIED? Measured in M-mode
# with capmode set -- the one privilege where core/load_store_unit.sv's cap_violation_detection
# block is reachable at all (its gate is `capmode_i && ld_st_priv_lvl_i == PRIV_LVL_M`; a domain
# runs in S-mode and can never satisfy the second half, see
# docs/history/15-09-2026_lsu-capmode-gate-why-domains-cannot-satisfy-it.md in the parent repo).
#
# WHY THIS EXISTS. Every in-domain reading of that block is a NEGATIVE (E1's stale loads retire),
# and a negative cannot separate "the block is unreachable from a domain" from "the block is
# broken". The block has five clauses; this file drives three of them from M-mode with capmode set
# and reports what each delivers. Two facts fix the pre-registration BEFORE the run:
#
#   * The block's first clause emits mcause 24 for an untagged base (NOT_CAP). On this core 24 is
#     riscv_pkg.sv:348 DEBUG_REQUEST, and csr_regfile.sv:2008/:2197 route that value into DEBUG
#     MODE, never to mtvec (R-24's collision, seen on the execute path). So a LIVE first clause
#     cannot show as a trap: it shows as the core halting in the debug ROM, i.e. a hang at the
#     simulation timeout. That arm therefore runs LAST, after every other reading is printed.
#   * The passing test cpmp-if-check.S already does `la t0, trap_count; sw x0, 0(t0)` in M-mode
#     after its CAPENTER at line 37 -- lines 65-66 -- and terminates in 542 cycles, and the board monitor's
#     rdtime emulation stores through an `add`-derived (untagged) base at every clock read. Both
#     say the first clause does NOT fire when its gate reads satisfied. This file measures that
#     directly and, with the bounds and permission clauses beside it, says whether ANY clause does.
#
# ARMS, IN ORDER (every arm CAPPRINTs its loaded value or the cause it observed; the trap handler
# keeps its state in REGISTERS ONLY -- a handler that stored through an integer base would itself
# be the thing under test):
#
#   arm 0  BEFORE CAPENTER: plain ld through an integer base       -> the sentinel, no trap
#          (instrument control: the buffer reads back; capmode is 0 so no clause can fire)
#          CAPENTER
#   arm 1  illegal instruction (.word 0)                          -> cause 2, handler resumes
#          (vector control: proves the temporary vector catches and resumes)
#   arm 2  plain ld, in bounds, through a tagged NONLIN RW cap     -> the sentinel, no trap
#          (block reached and passed: a valid access through a capability base)
#   arm 3  plain ld through a tagged NONLIN cap with perm WO       -> cause 27 if the permission
#          clause is live, else the sentinel
#   arm 4  plain ld at bound_end through the RW cap                -> cause 28 if the bounds clause
#          is live, else the word past the buffer (0x0BADB0B0)
#   arm 5  AFTER CAPENTER: plain ld through an integer base (la)   -> the sentinel if the NOT_CAP
#          clause is dead; a HANG at the timeout (debug-mode entry) if it is live
#
# READ THE PAIRS. arm 0 and arm 2 must both read the sentinel or the instrument is broken and no
# other arm carries a verdict. arm 1 must read cause 2 or the handler cannot be trusted to report
# a capability cause either. Then:
#   arms 3/4 trap, arm 5 returns   -> the block is live and only its NOT_CAP clause is inert
#                                     (or routed to debug mode and swallowed -- the sim cannot
#                                     tell those apart, and neither can the board)
#   arms 3/4 trap, arm 5 hangs     -> the whole block is live; cpmp-if-check's pass then needs a
#                                     different explanation and must be re-read
#   nothing traps, arm 5 returns   -> the block is dead when reached: a defect independent of
#                                     the gate, and "satisfy the gate" would not have enforced
#                                     anything either
#
# The handler counts traps in s11 and keeps the last cause in s10; each arm moves those into its
# own registers (s2..s9) before the next arm so the CAPPRINTs at the end name each arm's reading.
# CAPPRINT prints a register as a capability; for a scalar the cursor field IS the value.
#
# This file goes through CPP: no comment line may begin with a hash followed by a directive word,
# and a MACRO(...) form inside a comment would expand.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1                    # TESTNUM must be non-zero or RVTEST_FAIL spins instead of reporting
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                   # last observed cause (handler-owned)
  li   s11, 0                   # trap count (handler-owned)

# ---------------------------------------------------------------- arm 0: control before CAPENTER
  la   t1, buf
  ld   s2, 0(t1)                # sentinel expected
  mv   s3, s11                  # traps so far (expect 0)

# ---------------------------------------------------------------- CAPENTER -- capmode sticky from here
  la   a0, _start
  la   a1, _end_of_text
  CAPENTER(a0, a1)

# ---------------------------------------------------------------- gate witnesses
# capmode: a capability CSR write is gated on capmode_q (csr_regfile.sv:2374); a CSCRATCH round trip
# that returns the capability proves capmode is set. Privilege: mstatus is printed so MPRV (bit 17)
# and MPP (bits 12:11) are on record -- ld_st_priv_lvl is mprv ? mpp : priv (csr_regfile.sv:2273).
  CAPCREATE(a6)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a6, a0)
  la   a2, buf
  la   a3, buf_end
  CAPBOUND(a6, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(a6, a4)
  CCSRRW(x0, CCSR_CSCRATCH, a6)
  CCSRRW(a7, CCSR_CSCRATCH, x0)
  CAPPRINT(a7)                  # capmode witness: type 2, bounds of buf if capmode is set
  csrr t5, mstatus
  CAPPRINT(t5)                  # privilege witness: MPRV bit 17, MPP bits 12:11

# ---------------------------------------------------------------- arm 1: the vector catches
  .word 0                       # illegal instruction -> cause 2; the handler steps 4 bytes
  mv   s4, s10                  # cause seen (expect 2)
  li   s10, 0

# ---------------------------------------------------------------- arm 1b: a misaligned plain load
# The ordinary misaligned exception travels the same LSU -> MMU -> load-unit path the capability
# causes would (load_store_unit.sv:951 merges cap_exception INTO misaligned_exception). If a
# misaligned lw traps (cause 4) while the capability clauses do not, the loss is specific to the
# capability path's timing; if it does NOT trap either, the whole exception path is broken.
  la   t1, buf
  addi t1, t1, 1
  li   t4, 0
  lw   t4, 0(t1)                # cause 4 expected; t4 stays 0 if it traps
  mv   s1, s10                  # cause seen (expect 4)
  li   s10, 0

# ---------------------------------------------------------------- arm 2: in bounds, RW cap
  CAPCREATE(a1)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a1, a0)
  la   a2, buf
  la   a3, buf_end
  CAPBOUND(a1, a2, a3)
  li   a4, CAP_PERM_RW
  CAPPERM(a1, a4)
  CAPPRINT(a1)                  # the capability the next three arms use (type, perm, bounds)
  ld   s5, 0(a1)                # sentinel expected, no trap
  mv   s6, s10                  # (expect 0)
  li   s10, 0

# ---------------------------------------------------------------- arm 3: permission clause (WO)
  CAPCREATE(a5)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a5, a0)
  CAPBOUND(a5, a2, a3)
  li   a4, CAP_PERM_WO
  CAPPERM(a5, a4)
  CAPPRINT(a5)
  li   s7, 0
  ld   s7, 0(a5)                # cause 27 if the permission clause is live (s7 stays 0), else the sentinel
  mv   s8, s10                  # cause seen (expect 27 if live)
  li   s10, 0

# ---------------------------------------------------------------- arm 4: bounds clause
  li   s9, 0
  ld   s9, 64(a1)               # ea = bound_end: cause 28 if the bounds clause is live, else 0x0BADB0B0
  mv   t6, s10                  # cause seen (expect 28 if live)
  li   s10, 0

# ---------------------------------------------------------------- report everything so far
  CAPPRINT(s2)                  # arm 0 value
  CAPPRINT(s3)                  # arm 0 traps
  CAPPRINT(s4)                  # arm 1 cause
  CAPPRINT(s1)                  # arm 1b cause (misaligned lw)
  CAPPRINT(t4)                  # arm 1b value (0 if it trapped)
  CAPPRINT(s5)                  # arm 2 value
  CAPPRINT(s6)                  # arm 2 cause
  CAPPRINT(s7)                  # arm 3 value
  CAPPRINT(s8)                  # arm 3 cause
  CAPPRINT(s9)                  # arm 4 value
  CAPPRINT(t6)                  # arm 4 cause
  CAPPRINT(s11)                 # total traps before arm 5

# ---------------------------------------------------------------- arm 4s: STORE clauses
# The store unit takes the same mmu_exception (load_store_unit.sv:656). A plain sd through a
# READ-ONLY tagged cap (perm clause for stores, 27 if live) into buf[1], then a misaligned sw at
# buf+1 (cause 6 if live); buf[0..1] are read back through an aligned ld so a silently completed
# store shows as changed bytes.
  CAPCREATE(a6)
  li   a0, CAP_TYPE_NONLIN
  CAPTYPE(a6, a0)
  CAPBOUND(a6, a2, a3)
  li   a4, CAP_PERM_RO
  CAPPERM(a6, a4)
  li   t3, 0x5EC0DE
  sd   t3, 8(a6)                # store through a read-only cap: cause 27 if live
  mv   s3, s10                  # cause seen (expect 27 if live)   [s3 reused: arm 0 traps were 0]
  li   s10, 0
  la   t1, buf
  addi t1, t1, 1
  li   t3, 0x11
  sw   t3, 0(t1)                # misaligned store: cause 6 if live
  mv   t6, s10                  # cause seen (expect 6 if live)    [t6 reused: printed above already]
  li   s10, 0

# arm 4b: the STORE side of the bounds clause. arm 4 covers the LOAD at bound_end; without this
# the entry says nothing about a store PAST the top, which is the direction that corrupts rather
# than merely leaks. Same capability, same address, opposite direction -- so a difference between
# arm 4 and arm 4b is about load-vs-store and nothing else.
  li   t3, 0x5B0
  sd   t3, 64(a1)               # ea = bound_end through the RW cap: cause 28 if the clause is live
  mv   s0, s10                  # cause seen (expect 28 if live)   [s0: unused until here]
  li   s10, 0

  la   t1, buf
  ld   s4, 0(t1)                # buf[0] after the stores (sentinel if neither store landed)
  ld   s6, 8(t1)                # buf[1] after the stores (0 if the RO-cap store was refused)
  la   t1, buf_end
  ld   t4, 0(t1)                # arm 4b guard word: 0x0BADB0B0 if refused, 0x5B0 if it LANDED
                                #   [t4 reused: its arm-1b value was printed at the first report]
  CAPPRINT(s3)                  # arm 4s cause (sd via RO cap)
  CAPPRINT(t6)                  # arm 4s cause (misaligned sw)
  CAPPRINT(s4)                  # buf[0] after
  CAPPRINT(s6)                  # buf[1] after
  CAPPRINT(s0)                  # arm 4b cause (sd at bound_end)
  CAPPRINT(t4)                  # arm 4b guard word: LANDED means the store went outside the region

# ---------------------------------------------------------------- arm 5: NOT_CAP clause, LAST
  la   t1, buf
  ld   t2, 0(t1)                # sentinel if the clause is dead; debug-mode entry (hang) if live
  CAPPRINT(t2)                  # arm 5 value
  CAPPRINT(s10)                 # arm 5 cause (0 unless it trapped to mtvec, which cause 24 cannot)

  RVTEST_PASS

# ---------------------------------------------------------------- trap handler (registers only)
# Direct-mode mtvec. Records the cause in s10, counts in s11, steps over the 4-byte faulting
# instruction and returns. No memory access: a store through an integer base here would be the
# very thing under test. An interrupt fails loudly.
.align 4
trap_handler:
  csrr t0, mcause
  bltz t0, unexpected_trap
  mv   s10, t0
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

unexpected_trap:
  li   TESTNUM, 9
  RVTEST_FAIL

.align 4
_end_of_text:

.data
RVTEST_DATA_BEGIN

.align 4
buf:
  .dword 0x4C535550             # "LSUP"
  .dword 0, 0, 0, 0, 0, 0, 0
buf_end:
  .dword 0x0BADB0B0             # the word past the buffer: what an unenforced bounds arm reads
  .dword 0

  TEST_DATA

RVTEST_DATA_END
