# See LICENSE for license details.
# This file goes through CPP: no comment line may begin with a hash followed by a directive word, and a MACRO name
# followed by a parenthesis anywhere in a comment, trailing ones included, would expand.

#*****************************************************************************
# s23-cpmp-rearm.S -- S-23's data face: can an id-preserving CPMP rewrite re-arm a cleared validity bit, and which
#                     workaround can? (2026-10-19)
#-----------------------------------------------------------------------------
#
# WHY. S-23's data face has now been captured four times on two bitstreams and three modules, every instance with
# slot 7 holding a live capability whose bounds cover the refused physical address and whose tag is set, so of the
# six terms in the CPMP allow (pmp_data_if.sv:145-151) only the per-entry validity bit is left. Every instance is
# also preceded in the same step by the monitor's fetch-retry handler (capstone-sbi 007d81d,
# sbi_capstone.c:3490-3533), which for EVERY slot reads the entry, writes a null, writes the entry back, and then
# returns to the faulting pc.
#
# WHAT THE SOURCE ALREADY SAYS, enumerated rather than inferred. There are exactly TWO setters of the validity bit:
# the ADOPT at pmp_data_if.sv:88-93, whose third term requires the entry's revnode id to DIFFER from the tracked
# one, and the NAMESPACE FLUSH at :109-114. One clearer, :96-108, a broadcast whose index matches. Reset leaves all
# sixteen INVALID (:120). So an id-PRESERVING rewrite -- which is exactly what the monitor's handler does -- cannot
# re-arm a bit that a broadcast has cleared, and the handler's write-back gives every appearance of refreshing a
# slot it leaves denied. The handler is therefore not the trigger; it is what hands execution back to the access
# that gets refused, and what fails to repair it.
#
# PRIOR ART, and what it already settles. r12-recl-cpmp.S's P3 is this exact step -- "reinstall the same stale e
# into CPMP0, expect 1: same id, not re-adopted" -- so the key move is already written down. But that arm is
# registered in NO testlist, is not in the 95-test sweep, and has no run log in either of this lane's trees, so its
# expectations are a claim with no reading behind them here. This arm re-measures it as its own control and goes on
# to the part that is new: which repair can re-arm the bit.
#
# THE PROBES, in order, each a load from S through one CPMP entry with the M-mode trap counter read across it.
# "ALLOWED" means the counter did not move; "DENIED" means it moved by one. Marker 0x5a23 opens the run.
#
#   Q1  slot 0 installed, probe                      ALLOWED   INSTRUMENT CONTROL: the entry works at all
#   Q2  the FRTY sweep, probe slot 0                 ALLOWED   the handler alone is harmless
#   Q3  REVOKE h0 (broadcasts index of c0), probe     DENIED   the broadcast clears the validity bit
#   Q4  the FRTY sweep again, probe slot 0            DENIED   THE KEY READING: an id-preserving rewrite of the
#                                                              very entry cannot re-arm it
#   Q5  csrsi csupctl, 4 (the strobe), probe          ALLOWED  THE WORKAROUND: the only other setter, M-mode
#   Q6  probe slot 1 after REVOKE h2                 ALLOWED   INDEX CONTROL: a broadcast naming a DIFFERENT
#                                                              index must not deny this entry
#   Q7  probe slot 2 (whose index h2 named)            DENIED   the same broadcast, the matching index
#   Q8  srevroot 0 then back, SPACED, probe slot 2    ALLOWED  the S-reachable repair
#   Q9  slot 3 after REVOKE h3                         DENIED   a fresh clear for the last variant
#   Q10 srevroot 0 then back, BACK-TO-BACK, slot 3    ALLOWED or DENIED -- see below
#
# Q10 IS THE ONE READING THAT IS NOT A PREDICTION. The namespace edge detector only samples the live root when the
# rev-node unit is idle (ex_stage.sv:1403-1415, gated on rev_busy = owed responses, a handshake, or memory traffic),
# so two writes close together can both land while it is busy, the intermediate value is never sampled, and NO
# pulse is produced. Software cannot see that happen: a read of srevroot returns the CSR (csr_regfile.sv:1078), not
# the held copy. So Q10 reading DENIED where Q8 reads ALLOWED is the collapse, measured; Q10 reading ALLOWED means
# the window is narrower than two back-to-back CSR writes and the S-mode repair needs no spacing.
#
# WHAT EACH OUTCOME MEANS, written before the run.
#   * Q2 ALLOWED and Q4 DENIED -> the mechanism as stated: the handler is harmless on a live entry and powerless on
#     a cleared one. This is what makes the monitor's one-line strobe the right fix and not a guess.
#   * Q2 DENIED -> the handler's read-null-write-back clears the bit BY ITSELF, the mechanism is the handler, and
#     the workaround must change the handler's shape rather than append to it.
#   * Q4 ALLOWED -> the rewrite does re-arm, this whole account is wrong, and S-23's data face needs another
#     mechanism entirely.
#   * Q5 ALLOWED -> `csrsi csupctl, 4` at the end of fetch_retry_covered unblocks the corpus with no bitstream.
#     Note csupctl bit 2 READS BACK AS ZERO (csr_regfile.sv:1077), so csrsi sets it for exactly one cycle and
#     preserves csupctl_resume, csupctl_virt and csupctl_table; the strobe itself is one cycle by construction,
#     since ns_flush_strobe_d defaults to 0 every cycle (:1183).
#   * Q6 DENIED -> the broadcast is not index-scoped and the whole model of the tracker is wrong.
#   * Q1 DENIED -> the instrument never worked and NOTHING else in the run is evidence.
#
# WHAT IT READ, run s23 on bf3db3175: 32 readings, 6 denials, and every probe that was soundly constructed read
# exactly as pre-registered. The harness printed FAILED at the 4,000,013-cycle ceiling for a reason unrelated to
# the arm (see the exit path at the bottom), so read the readings and not the verdict.
#   0x5a23  the run started
#   Q1  0   ALLOWED   the instrument control
#   Q2  0   ALLOWED   the handler's sweep over all sixteen entries is harmless on a live entry
#   Q3  1   DENIED    the broadcast clears the validity bit
#   Q4  1   DENIED    THE KEY READING, CONFIRMED: the handler's id-preserving rewrite of the very entry cannot
#                     re-arm it, because the adopt's third term needs a CHANGED id
#   Q5  0   ALLOWED   the strobe re-arms the bit. READ THE CORRECTION BELOW BEFORE CALLING THIS A WORKAROUND.
#   Q6  0   ALLOWED   the index control: a broadcast naming a different index does not deny this entry
#   Q7  1   DENIED    the matching index
#   Q9  1   DENIED    a fresh clear
#   last cause 5, total denials 6.
# THE DENIAL SURFACES AS CAUSE 5, a load access fault, not as a capability refusal. That is the same cause and
# shape the board's own panic reports ("Oops - load access fault", cause 5, badaddr the physical address), which
# is a second point of contact between this arm and the captures.
#
# Q8 AND Q10 WERE VOID BY CONSTRUCTION, mine. The arm read the live srevroot to restore it and wrote x0 as the
# intermediate -- but the live root WAS 0x0 (the trace's csrr reads x21 = 0), so both writes wrote 0 over 0, the
# held copy never saw a change, and no namespace pulse could have been produced. Both probes therefore read
# DENIED and say NOTHING about the S-mode repair: the arm never created the condition under test. The intermediate
# is now a nonzero aligned constant. This is the same failure shape as the FAULT_ESC arm's integer base and the
# fetch arm's setup proof, and the check that catches it is the same one: read back what the setup actually
# installed, which here means printing the root the arm saved.
#
# CORRECTION, 2026-10-19, to this arm's own framing and to what was reported from it. EVERY DENY IN THIS ARM IS
# A TRUE DENY, Q5's included. Each clear is a REVOKE of the handle above that slot's own child, so the capability
# left in the slot is genuinely dead afterwards -- r12-recl-cpmp's header calls it "the same stale e" for the same
# reason. So:
#   * Q4 stands unchanged: an id-preserving rewrite cannot re-arm a cleared bit, whatever cleared it.
#   * Q5 stands as a READING -- the strobe does re-arm the bit -- but the bit it re-armed was protecting a REVOKED
#     capability, so the access it allowed was to memory whose authority had been withdrawn. Q5 is therefore an
#     instance of a TRUE denial being masked, and it was reported as the workaround working. It is both.
#   * This arm contains NO false deny, so it cannot grade the workaround's safety in either direction.
# A false deny needs two LIVE capabilities sharing a revocation INDEX, which cannot happen within one namespace
# (ids are unique per live node) and can happen ACROSS namespaces, because under the virtual flag ids carry no
# generation: a virtual (0, i) and the physical (0, i) are one tag, which is what the namespace flush exists for.
# So the construction that would decide whether S-23's field deny is false is a CSRUNV arm with its own lifetime
# table, colliding a virtual index with a physical one held in a CPMP entry. That is a different arm and it is
# owed.
# The cheapest thing that would settle the field instance is not an arm at all: the monitor already dumps slot 7's
# CPID at the refusal, so reporting region_live and region_cpmp for that slot in the same handler says whether the
# monitor still vouches for the region -- live and mapped means the deny is false and the strobe is a repair; not
# vouched means the deny is correct and the strobe hands back retired memory.
#
# WHY M-MODE VALUES LIVE IN REGISTERS: with R-34 in place a store through an integer base under capmode faults, so
# a trap counter in memory would itself trap. s11 counts traps, s10 holds the last cause, and each probe's outcome
# is the change in s11 across it, printed immediately rather than accumulated.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

# CAPENTER grants a0 = [0x80800000, 0x82000000); every probe page and SPLIT point must lie inside it.
#define P0 0x80801000
#define P1 0x80802000
#define P2 0x80803000
#define P3 0x80804000
#define P4 0x80805000
#define P5 0x80806000
#define P6 0x80807000
#define CSUPCTL 0x7C4
#define SREVROOT 0x5C1

# One slot of the monitor's fetch-retry sweep: read the entry out (writing a null in its place), then write it
# back. This is read_cpmp/write_cpmp exactly, including the transient null.
#define FRTY_ONE(N) \
  CCSRRW(t4, CCSR_CPMP(N), x0); \
  CCSRRW(x0, CCSR_CPMP(N), t4)

# The whole sweep, all sixteen entries, as the handler does it. Unrolled because the ccsr number is an immediate.
#define FRTY_SWEEP \
  FRTY_ONE(0); FRTY_ONE(1); FRTY_ONE(2); FRTY_ONE(3); \
  FRTY_ONE(4); FRTY_ONE(5); FRTY_ONE(6); FRTY_ONE(7); \
  FRTY_ONE(8); FRTY_ONE(9); FRTY_ONE(10); FRTY_ONE(11); \
  FRTY_ONE(12); FRTY_ONE(13); FRTY_ONE(14); FRTY_ONE(15)

# MKENT(SLOT, BASE, CHILD): a NONLIN RW entry over one page, carrying CHILD's revocation node, installed in SLOT.
#define MKENT(SLOT, BASE, CHILD) \
  CAPCREATE(t2); \
  li t0, CAP_TYPE_NONLIN; \
  li t3, BASE; \
  li t4, BASE + 0x1000; \
  li t1, CAP_PERM_RW; \
  CAPTYPE(t2, t0); \
  CAPBOUND(t2, t3, t4); \
  CAPPERM(t2, t1); \
  CAPNODE(t2, CHILD); \
  CCSRRW(x0, CCSR_CPMP(SLOT), t2)

# PROBE(ADDR, RET): drop to S, one load at ADDR, ecall back, and print the trap delta.
#define PROBE(ADDR, RET) \
  li s7, ADDR; \
  mv s8, s11; \
  la s9, RET; \
  j go_s; \
RET: \
  sub t2, s11, s8; \
  CAPPRINT(t2)

.option norvc

.section .text.init
.globl _start
_start:

  li   gp, 1
  li   s10, 0
  li   s11, 0
  la   t0, m_trap_handler
  csrw mtvec, t0

  CAPENTER(x0, x0)              # a0 = linear data cap on node 2; capmode on
  CAPPRINT(a0)                  # what CAPENTER granted, so the probe pages can be checked against it

  li   t0, 0x001fffffffffffff   # PMP0: NAPOT over everything, RWX, so S-mode may touch memory at all
  csrw pmpaddr0, t0
  li   t0, 0x1f
  csrw pmpcfg0, t0

  # Under capmode an S-mode FETCH is checked against the CPMP entries too, so S needs a code entry or it traps at
  # the first instruction. NONLIN RWX over the text carrying the base's node, which stays live all run.
  CAPCREATE(a5)
  li   t2, CAP_TYPE_NONLIN
  la   t3, _start
  la   t4, _end_of_text
  li   t5, CAP_PERM_RWX
  CAPTYPE(a5, t2)
  CAPBOUND(a5, t3, t4)
  CAPPERM(a5, t5)
  CAPNODE(a5, a0)
  CCSRRW(x0, CCSR_CPMP(15), a5)

  li   t0, 0x5a23
  CAPPRINT(t0)                  # the run started

# ---------------------------------------------------------------- four children off the tail of a0, a handle above
# each. Each child owns one page, and its revocation node is the index a broadcast will name.
  li   t1, P0
  SPLIT(a2, a0, t1)             # a2 = [P0, end)
  li   t1, P1
  SPLIT(a3, a2, t1)             # a2 = [P0, P1), a3 = [P1, end)
  li   t1, P2
  SPLIT(a4, a3, t1)             # a3 = [P1, P2), a4 = [P2, end)
  li   t1, P3
  SPLIT(a6, a4, t1)             # a4 = [P2, P3), a6 = [P3, end)
  li   t1, P4
  SPLIT(a7, a6, t1)             # a6 = [P3, P4) = c3, a7 = [P4, end)
  li   t1, P5
  SPLIT(t6, a7, t1)             # a7 = [P4, P5) = c4, t6 = [P5, end) = c5
  li   t1, P6
  SPLIT(t5, t6, t1)             # t6 = [P5, P6) = c5, t5 = [P6, end) = c6
  # every split is done before any handle is minted, so no handle sits above a capability split afterwards
  MREV(s1, a2)                  # h0, above c0 = a2
  MREV(s2, a3)                  # h1, above c1 = a3
  MREV(s3, a4)                  # h2, above c2 = a4
  MREV(s4, a6)                  # h3, above c3 = a6

  MKENT(0, P0, a2)
  MKENT(1, P1, a3)
  MKENT(2, P2, a4)
  MKENT(3, P3, a6)

# ---------------------------------------------------------------- Q1: the instrument control
  PROBE(P0, q1_back)            # ALLOWED

# ---------------------------------------------------------------- Q2: the handler alone, on a live entry
  FRTY_SWEEP
  PROBE(P0, q2_back)            # ALLOWED

# ---------------------------------------------------------------- Q3: the broadcast
  REVOKE(s1)
  PROBE(P0, q3_back)            # DENIED

# ---------------------------------------------------------------- Q4: the handler on a cleared entry -- THE KEY
  FRTY_SWEEP
  PROBE(P0, q4_back)            # DENIED: the rewrite preserves the id, so the adopt cannot fire

# ---------------------------------------------------------------- Q5: the M-mode strobe -- THE WORKAROUND
  csrsi CSUPCTL, 4              # bit 2 reads back as 0, so this sets it for one cycle and preserves the rest
  PROBE(P0, q5_back)            # ALLOWED

# ---------------------------------------------------------------- Q6 and Q7: one broadcast, two entries
  REVOKE(s3)                    # names c2's index
  PROBE(P1, q6_back)            # ALLOWED: slot 1 tracks a different index
  PROBE(P2, q7_back)            # DENIED: slot 2 tracks the named one

# ---------------------------------------------------------------- Q8: the S-reachable repair, SPACED
  csrr s5, SREVROOT             # the live root, restored exactly below
  li   s6, 0x81000000           # a 4 KiB-aligned value DIFFERENT from the live root, which read 0x0 on the first
                                # run. Writing x0 there wrote 0 over 0 twice and changed nothing, so the first
                                # run's Q8 and Q10 were void by construction: the edge detector needs a CHANGE
                                # (csr_regfile.sv:2020 forces the low 12 bits to zero, so the value must differ
                                # above bit 11). Nothing walks the table between the two writes, so an
                                # intermediate root that points at no table is safe here.
  csrw SREVROOT, s6
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  csrw SREVROOT, s5
  PROBE(P2, q8_back)            # ALLOWED if either edge was sampled

# ---------------------------------------------------------------- Q9 and Q10: the same repair BACK-TO-BACK
  REVOKE(s4)                    # names c3's index
  PROBE(P3, q9_back)            # DENIED
  csrw SREVROOT, s6
  csrw SREVROOT, s5
  PROBE(P3, q10_back)           # ALLOWED, or DENIED if both writes landed while the unit was busy

# ---------------------------------------------------------------- Q11 and Q12: THE STROBE'S REACH IS GLOBAL, and
# make_hole's omission is what makes that matter. Slots 4 and 5 each get a vouched entry and a handle, then both
# are revoked, which leaves each slot LOADED with a dead capability -- exactly the state make_hole (:781-793) and
# the region-pop loop (:3035-3047) leave behind, since neither writes the slot. ONE strobe then follows and BOTH
# slots are probed. The reading that matters is the second: a strobe issued to repair one entry re-arming an
# unrelated stale one is why a vouching guard has to be global rather than per slot.
# PRE-REGISTERED: ALLOWED, ALLOWED, DENIED, DENIED, then ALLOWED, ALLOWED.
  MKENT(4, P4, a7)
  MREV(s5, a7)                  # h4, above c4 -- s5's saved root is spent by now
  MKENT(5, P5, t6)
  MREV(s6, t6)                  # h5, above c5
  PROBE(P4, q11a_back)          # ALLOWED: slot 4's own control
  PROBE(P5, q11b_back)          # ALLOWED: slot 5's own control
  REVOKE(s5)
  REVOKE(s6)
  PROBE(P4, q11c_back)          # DENIED
  PROBE(P5, q11d_back)          # DENIED
  FRTY_SWEEP                    # the guard's own inspection, itself id-preserving
  csrsi CSUPCTL, 4              # ONE strobe
  PROBE(P4, q12a_back)          # ALLOWED
  PROBE(P5, q12b_back)          # ALLOWED, from the SAME strobe: the reach is global
# Both of these are TRUE denials, as every deny in this arm is, so Q12 is also a second instance of the masking
# the correction above describes -- one strobe masking two of them at once.
#
# WHAT THEY READ, run s23c on bf3db3175: harness SUCCESS, 15,824 cycles, 806 retired, 6 denials, 21 readings, and
# EVERY pre-registered value holds including the new pair:
#   q11a 0 ALLOWED   slot 4's control        q11b 0 ALLOWED   slot 5's control
#   q11c 1 DENIED    slot 4 after its revoke q11d 1 DENIED    slot 5 after its revoke
#   q12a 0 ALLOWED   slot 4 after ONE strobe q12b 0 ALLOWED   slot 5 after the SAME strobe
# So ONE csupctl bit-2 strobe re-armed TWO unrelated slots, each holding a dead capability left loaded exactly as
# make_hole leaves it. The reach is global, measured rather than inferred from the signal's fan-out, which is why
# a vouching guard before the strobe has to inspect every loaded slot and not only the one being repaired.

# ---------------------------------------------------------------- Q13: THE CONTROL Q5 LACKED. `csrsi csupctl, 4`
# does TWO things -- it sets ns_flush_strobe_d AND raises flush_o, a pipeline flush (csr_regfile.sv:2019). So Q5
# does not separate "the strobe re-armed the tracker" from "a pipeline flush cleared a stale deny". This pair
# separates them on one slot with one clear: a csupctl write with bit 2 CLEAR raises flush_o and NOT the strobe,
# then the same write with bit 2 set raises both. csrsi is used rather than csrw because CSR_SET ORs with the read
# value (`:2797`) and csupctl bit 3 reads back as csupctl_table_q, so a SET cannot clear the table namespace --
# which matters here, because changing csupctl_table would move vctx_table_o and fire the namespace flush by
# itself, i.e. the control would contain the thing it is controlling for. Bit 0 is inert in this arm: it arms a
# resume that no supervised CALL consumes.
# PRE-REGISTERED, both branches, because this arm can only make the earlier reading weaker:
#   q13a ALLOWED  slot 6's control
#   q13b DENIED   after its revoke
#   q13c DENIED   after `csrsi csupctl, 1` -- the pipeline flush alone does not re-arm the tracker
#   q13d ALLOWED  after `csrsi csupctl, 4` -- the strobe does
# If q13c reads ALLOWED it does NOT void the monitor's stage 1, which is the semantics owner's refinement and
# better than this lane's first wording: it would narrow a stage-1 reading to "a deny present at the FRTY is
# removed by a pipeline flush", itself a finding, since a plain flush would then suffice and the global re-arm
# would not be needed. What it would void is only this lane's claim that the RE-ARM is the repairing mechanism.
#
# WHAT IT READ, run s23d on bf3db3175: harness SUCCESS, 17,745 cycles, 960 retired, 8 denials, 25 readings, and
# THE CONTROL FIRES in the direction that preserves the earlier reading:
#   q13a 0 ALLOWED   q13b 1 DENIED   q13c 1 DENIED   q13d 0 ALLOWED
# So `csrsi csupctl, 1`, which raises flush_o and not the strobe, leaves the deny standing, and the same write
# with bit 2 set removes it. Q5's reading is therefore attributable to the namespace re-arm and NOT to the
# pipeline flush that instruction also raises. Q1 through q12b are unchanged, so the addition perturbs nothing.
  MKENT(6, P6, t5)
  MREV(s5, t5)                  # h6, above c6; s5 is free again after q11's revoke
  PROBE(P6, q13a_back)          # ALLOWED
  REVOKE(s5)
  PROBE(P6, q13b_back)          # DENIED
  csrsi CSUPCTL, 1              # flush_o without the strobe
  PROBE(P6, q13c_back)          # DENIED, predicted
  csrsi CSUPCTL, 4              # flush_o WITH the strobe
  PROBE(P6, q13d_back)          # ALLOWED

# ---------------------------------------------------------------- report
  CAPPRINT(s10)                 # the last trap cause seen, expect 24 if every deny was a capability refusal
  CAPPRINT(s11)                 # the total trap count
  li   t0, 0x55
  CAPPRINT(t0)
  j    pass

# ---------------------------------------------------------------- drop to S, one load at s7, ecall back to *s9
go_s:
  li   t0, MSTATUS_MPP
  csrc mstatus, t0
  li   t0, (1 << 11)
  csrs mstatus, t0
  la   t0, s_probe
  csrw mepc, t0
  mret
s_probe:
  ld   t1, 0(s7)
  ecall

.align 4
m_trap_handler:
  csrr t0, mcause
  li   t1, 9
  beq  t0, t1, ecall_back
  mv   s10, t0                  # any other trap: count it and step over it
  addi s11, s11, 1
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
ecall_back:
  li   t0, MSTATUS_MPP
  csrs mstatus, t0              # MPP = M
  csrw mepc, s9
  mret

fail:
  RVTEST_FAIL
pass:
  CAP_PASS(a1)                  # NOT RVTEST_PASS: capmode refuses its integer-base store to tohost with cause 24
                                # (asm_insn.h:108-109), so the first run reached the 4,000,013-cycle ceiling and
                                # printed FAILED with every reading correct. A harness verdict that is wrong for a
                                # reason unrelated to the arm is worse than no verdict.

RVTEST_CODE_END
_end_of_text:

  .data
RVTEST_DATA_BEGIN
.align 4
test_data:
  .dword 0x1111111111111111
  .dword 0x2222222222222222
test_data_end:
RVTEST_DATA_END
