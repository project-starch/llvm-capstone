# See LICENSE for license details.

#*****************************************************************************
# r35-rotate-stale.S
#-----------------------------------------------------------------------------
#
# R-35: on the board RTL a REVOKED capability still reads AND writes the storage
# its object gave up, and the access does not trap. Root cause at 054cea69b in
# load_store_unit.sv: a SINGLE core-wide tracked revnode id whose adopt arm
# re-adopts any access presenting a DIFFERENT id as VALID, without asking the
# rev-node unit. Cause 25 is the LAST arm of the priority chain and tests only
# that tracker bit, so once the tracker is displaced the clause cannot fire,
# while bounds 28 and permissions 27 survive by reading the capability's own
# metadata.
#
# WHY THIS TEST EXISTS. Every board measurement of R-35 was taken on an
# integration bitstream whose timing does not close, WNS -8.307 with 90,379
# failing endpoints, and the repro folder says plainly that nothing there
# separates a design property from an artefact of that build. Simulation has no
# timing artefact by construction, so a two-sided run here is STRONGER evidence
# than the board reading, and it is the acceptance test a fix has to flip.
#
# THREE ARMS. The first two differ in exactly ONE thing -- whether an access to
# a DIFFERENT revnode intervenes between the revoke and the stale access:
#
#   ARM A  no rotation . revoke A, then load through A. The presented id EQUALS
#          the tracked id, so the adopt arm does not run, the broadcast has
#          already cleared validity -> cause 25 FIRES.
#   ARM B  rotation    . revoke A, load through B, then load through A. The
#          presented id DIFFERS, the adopt arm runs and asserts validity
#          -> cause 25 suppressed -> NO TRAP and the data is returned.
#   ARM C  the write half, same rotation -> the stale STORE lands.
#
# ARM A IS THE POSITIVE CONTROL for the whole instrument: it can only trap if
# capmode is set, the block is live, the renumbered cause 25 is reachable and
# delivery works. IF ARM A DOES NOT TRAP, ARMS B AND C MEAN NOTHING and the run
# is VOID rather than negative.
#
# AND THE REVOKE ITSELF IS WITNESSED, because a non-trap is equally what a
# revoke that never happened looks like. LCC with a zero immediate returns the
# rev-node unit's own validity answer in a register -- 1 live, 0 dead -- and
# never traps, so it cannot be confused with the thing under test. It is read
# on BOTH aliases BEFORE and AFTER the revoke. Proven pattern:
# verif/tests/custom/capstone/r12-delin-dead.S, whose header records reading 0.
#
# A BARRIER FOLLOWS THE REVOKE, and it is load-bearing. Nothing stalls a
# younger load behind the revoke walk: the rev-node unit exports a busy flag but
# at 054cea69b its only consumer is a debug LED, and REVOKE writes x0 so there
# is no register dependency to serialise on. A load placed immediately after the
# revoke can therefore be capability-checked BEFORE the walk's invalidating
# write exists, which reads exactly like the defect and is not.
#
# PRE-REGISTERED before this version ran:
#   arm 0 readback      00a5a5a5a5a5a5a5     memory is backed
#   grant a0            type 1, node 2, 80800000..82000000
#   grant a1            type 3, node 1
#   after the split     a0 node 2 lower, a6 node 3 upper -- DISTINCT
#   after delin         both type 2 NONLIN, ids unchanged
#   pre-revoke  LCC     A 1, B 1
#   post-revoke LCC     A 0, B 1             A dead, B a sibling and untouched
#   ARM A  cause        25                   INVALID_CAPABILITY
#   ARM B  cause        0, value = sentinel A
#   ARM C  cause        0, readback 00000000000000a5
#   traps               1, arm A's
# On repaired RTL arms B and C must also read 25 and neither return nor store,
# with arm A unchanged. That two-sidedness is what a fix has to show.
#
# CAUSE 28 CANNOT PRE-EMPT: every access under test is IN BOUNDS and PERMITTED.
# The chain is NOT_CAP, type, permissions, bounds, then revnode, so an
# out-of-bounds stale access would report 28 and a correct fix would look broken.
#
# TYPE AND TAG ARE NOT WITNESSES. Revocation writes rev-node memory only and
# never touches the register file, so on this RTL a revoked capability stays
# tagged and keeps both its type and its revnode id. That is why the witness has
# to be LCC and not a type read.
#
# SEMANTICS READ OUT OF THE RTL AT 054cea69b, each of which cost a run or would:
#   - CAPENTER with two x0 operands REPLACES a0 and a1: a0 becomes a LINEAR
#     capability over 80800000..82000000 with revnode 2, a1 a REVOKE capability
#     over all memory with revnode 1. Capabilities under test must be derived
#     AFTER entry, and nothing may be carried across it in a0 or a1.
#   - CAPNODE returns 0 here and is useless as a node readout; the id comes from
#     the metadata CAPPRINT emits.
#   - A scalar prints in a shorter form with no Cursor field, so grepping for
#     Cursor silently hides every scalar reading.
#   - SPLIT keeps the lower half in rs1 and puts the upper half, cut byte
#     included, in rd.
#   - DELIN rewrites the existing record's linear field in place. It does not
#     create a node, so the alias keeps the same revnode id and dies exactly
#     when that record dies.
#   - The access under test is a PLAIN ld or sd through a capability base, which
#     is what cap_violation_detection checks; not LDC, which would load a
#     128-bit capability out of untagged storage.
#
# AFTER CAPENTER THERE ARE NO PLAIN INTEGER-BASED DATA ACCESSES. With delivery
# repaired such an access raises the NOT_CAP clause, the shape that breaks the
# monitor. Sentinels are seeded BEFORE entry, every reading is emitted with
# CAPPRINT rather than stored, and the closing host write is EXPECTED to trap
# with cause 24: the run prints its readings and then spins to the timeout, and
# that timeout is the healthy shape rather than a lost run.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define SENT_A 0x00A5A5A5A5A5A5A5
#define SENT_B 0x005B5B5B5B5B5B5B
#define BARRIER .rept 64; nop; .endr

.option norvc
.section .text.init
.globl _start
_start:

  li   gp, 1
  csrwi mie, 0
  la   t0, trap_handler
  csrw mtvec, t0
  li   s10, 0                    # trap count
  li   s4, 0                     # last cause, cleared before each arm

# ---- seed the buffer while capmode is OFF ----
  li   t1, 0x80800000
  li   t2, SENT_A
  sd   t2, 0(t1)
  li   t5, SENT_B
  sd   t5, 32(t1)
  ld   s2, 0(t1)
  CAPPRINT(s2)                   # arm 0: memory is backed

# ---- enter. a0 = LINEAR grant node 2; a1 = REVOKE cap node 1. ----
  CAPENTER(x0, x0)
  CAPPRINT(a0)
  CAPPRINT(a1)

# ---- SPLIT the grant into two genuinely distinct revnodes ----
  li   a5, 0x80800020
  SPLIT(a6, a0, a5)              # a0 = lower, a6 = upper
  CAPPRINT(a0)                   # node A
  CAPPRINT(a6)                   # node B -- MUST differ

# ---- a handle over each half; each half handed out as a NONLIN alias ----
  MREV(s1, a0)
  DELIN(a0)                      # a0 = alias A
  MREV(s5, a6)
  DELIN(a6)                      # a6 = alias B
  CAPPRINT(a0)
  CAPPRINT(a6)

# ---- the revoke witness, live side ----
  LCC(t4, a0, x0)
  CAPPRINT(t4)                   # alias A validity -- expect 1
  LCC(t5, a6, x0)
  CAPPRINT(t5)                   # alias B validity -- expect 1

# ---- warm the tracker to node A through a legitimate access ----
  ld   t3, 0(a0)
  CAPPRINT(t3)                   # expect sentinel A

# ---- revoke the handle over A, then let the walk finish ----
  REVOKE(s1)
  BARRIER

# ---- the revoke witness, dead side. Without this the arms are ambiguous. ----
  LCC(t6, a0, x0)
  CAPPRINT(t6)                   # alias A validity -- expect 0, node is dead
  LCC(t2, a6, x0)
  CAPPRINT(t2)                   # alias B validity -- expect 1, a sibling

# ---- ARM A: no rotation. Presented id equals the tracked id. Expect 25. ----
  li   s4, 0
  ld   s6, 0(a0)
  CAPPRINT(s6)                   # value if it wrongly retired
  CAPPRINT(s4)                   # ARM A cause -- expect 25

# ---- ARM B: rotate the tracker through B, then repeat the stale read ----
  li   s4, 0
  ld   s7, 0(a6)                 # displaces the tracker to node B
  CAPPRINT(s7)                   # expect sentinel B
  ld   s8, 0(a0)                 # the stale read: in bounds, permitted
  CAPPRINT(s8)                   # expect sentinel A
  CAPPRINT(s4)                   # ARM B cause -- expect 0

# ---- ARM C: the write half. Same rotation, then a stale store. ----
  li   s4, 0
  ld   s9, 0(a6)                 # re-displace the tracker to node B
  li   t1, 0xA5
  sd   t1, 0(a0)                 # the stale WRITE through the revoked alias
  CAPPRINT(s4)                   # ARM C store cause -- expect 0
  ld   s11, 0(a6)                # displace again so the readback is not the store's own adopt
  ld   s11, 0(a0)
  CAPPRINT(s11)                  # expect 00000000000000a5: the store LANDED

  CAPPRINT(s10)                  # traps so far -- expect 1

  RVTEST_PASS                    # expected to trap with cause 24, then spin

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

.data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
