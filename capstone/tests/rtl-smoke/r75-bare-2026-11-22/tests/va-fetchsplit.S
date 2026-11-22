# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-fetchsplit.S -- the cross-namespace arm, with the caller's and the context's CODE COVERAGE SPLIT so the
#                    measurement is the caller's own fetch (2026-10-20)
#-----------------------------------------------------------------------------
#
# WHY THIS EXISTS. It is the dependency the next RTL revision's item 1 names: the namespace-aware invalidation
# broadcast cannot be graded without an arm that reads NO FAULT after the fix where today's mechanism predicts a
# refusal. `cap-fetchdeny` measures the refusal with the CALLER's own broadcast in the physical namespace; this
# arm makes the broadcast a CONTEXT's, in the context's namespace, which is the cross-namespace step the index-only
# compare carries and the fix is meant to stop.
#
# WHAT va-fetchctx GOT WRONG, and this fixes. There the caller's single CPMP grant covered the whole image, so it
# covered the CONTEXT's code too: the collision cleared the entry the context itself was fetching through, the
# fault was the context's rather than the caller's, and the handler could not repair it because a CCSRRW to
# 0x010..0x01F is a privilege violation while `sup_active_q` (`csr_regfile.sv:3018-3021`). The arm wedged with
# byte-identical counts across a fix to the repair's operands, which is how the cause was found.
#
# THE SPLIT. CPMP0 covers [_start, ctx) -- the caller's S code and the M handler's text -- and carries the
# TRACKED node, a physical child of x26. CPMP1 covers [ctx, pt_end): the context's code and the whole of .data,
# with CAPCREATE's default node 2, which a context table starting at 3 can never name. So the context's fetches
# and every data access on both sides survive the collision, and **the only thing a cleared tracker can refuse is
# a fetch of the caller's own code** -- which is the measurement. `ctx` is page-aligned so the two ranges meet at
# a clean boundary.
#
# AND THE REPAIR CAN NOW RUN, for a reason worth stating because it is the other half of the wedge: the context
# ecalls, so its escape is terminal (kind 2, cause 11), supervision is cleared, and `sup_active_q` is 0 by the
# time the caller's fetch faults. If this arm wedges anyway, supervision was NOT cleared at the escape and that
# is itself the reading.
#
# PRE-REGISTERED, on top of va-srevroot's twenty baseline readings unchanged in both arms:
#   firing (the code entry tracks the FIRST physical child, index 3, inside the 3..6 window the context sweeps)
#     faults 1; mtval the physical address of CSRUNV+4; mepc - s_code 4; children 0x100 0x100 0x100 0x100
#   control (the code entry tracks the EIGHTH, index 10, outside that window)
#     faults 0; mtval 0; mepc - s_code reads minus s_code; children 0x100 0x100 0x100 0x100
#
# CORRECTION to one pre-registered VALUE, made before the arm returned a clean reading and derivable from this
# file alone rather than from the run: `mepc - s_code` is **0x20, not 4**. The prediction was carried over from
# va-fetchctx's header and assumed CSRUNV was s_code's first instruction; count them -- `lla t0, tbl` (2),
# `csrw SREVROOT`, `csrr s3`, `lla a0, frame` (2), `li a1, 0`, CSRUNV -- and CSRUNV sits at s_code+0x1c, so the
# instruction after it is at s_code+0x20. The SUBSTANTIVE predictions (one fault, cause 1, mtval the fetch
# address, the four children at 0x100) are untouched. The control's reading of minus s_code is unaffected.
#
# FIRST RUN OF THE SPLIT ARM: **the fault FIRED, and the arm's only defect was its own repair.** The control read
# all 27 readings exactly as pre-registered (SUCCESS in 134,625 cycles, four children at 0x100, zero faults). The
# firing arm wedged -- but in the instruction trace the sequence is unambiguous, and it is the measurement:
#   the context's last store (`sd t6, 40(s10)`, the fourth 0x100 child record) at 0x8000306c, then
#   `INSTR_ACCESS_FAULT exception @ 0x0000000080001020`, with mtval = mepc = 0x80001020 = s_code+0x20, the
#   `csrr s4, SREVROOT` the caller resumes on after the escape; CSRUNV itself is at 0x8000101c (0x48b5165b).
# So a CONTEXT's kill broadcast, in the context's own namespace, cleared a tracker holding a PHYSICAL index and
# the caller's FIRST FETCH after the escape was refused with cause 1 -- the cross-namespace step the index-only
# compare carries, end to end, against a control naming an index outside the swept window. The handler then
# installed an untagged CPMP0 (see the repair) and the fetch repeated forever: 82,277 cause-1 entries. The
# wedge also MEASURES what va-fetchctx could only infer: the CCSRRW at 0x80002034 executed without a privilege
# fault, so `sup_active_q` really is 0 after a terminal ecall escape, and the repair can run.
# The children must read 0x100 in BOTH arms: that is what says the broadcasts went out, and it separates a real
# negative in the control from a context that never allocated. Both arms at zero faults with the children at
# 0x100 would mean the cross-namespace collision does not reach a caller slot at all -- which would REFUTE the
# mechanism the registry now records for both S-23 sites, and is the outcome this arm is built to be able to
# produce. Both firing would mean the index inference is wrong and the arm says nothing about namespaces.
#
# THE CSRUNV+4 ANOMALY, three variants (2026-10-20, at the semantics owner's request after the board read the
# discriminator). The field result that forced this: on pb10/tree B, one boot each, 4/4 cells both arms, the
# srz module reads FRTY 0 while THE SAME BUILD with the post-CSRUNV `csrw 0x5c1, x0` replaced by a `nop` --
# same layout, same instruction count -- reads FRTY 2, at the nop itself. So it is NOT timing and NOT
# allocation: **the instruction occupying CSRUNV+4 decides whether its own fetch is refused.**
#
# I ARGUED AGAINST THIS RUN EARLIER AND WAS WRONG TO. My reasoning was that both arms must read 1 by
# construction, since a write at CSRUNV+4 cannot execute before its own fetch, so the reading would only
# confirm what the source already gives. The field has refuted that premise: the slot's contents change the
# outcome, so something other than the instruction's own execution is involved, and the arm is informative.
# Recorded here because the go/no-go question is supposed to bind the decision, and it bound it the wrong way.
#
# CS4_TWO: TWO ENTRIES AND ESCAPES BACK TO BACK, which is what makes the triple discriminate (2026-10-20).
# The field triple is now complete and it RULES OUT the commit flush: on pb10/tree B, one boot each and 4/4
# cells, a nop reads FRTY 2, `csrrc x0, 0x5c1, rA` -- a write and a flush with NO value change -- also reads
# FRTY 2, and only the srz `csrw 0x5c1, x0` reads 0. So in the field the VALUE CHANGE is what matters.
#
# THE HYPOTHESIS being tested, the semantics owner's and labelled as theirs: a write at CSRUNV+4 cannot change
# the fetch of its own instruction, so the effect must reach a LATER fetch at that address. Every escape
# returns to CSRUNV+4 and the field makes many entries per session. With srz the caller sits at srevroot = 0
# between calls, so each NEXT entry is a 0 -> table-namespace edge whose `revnode_ns_flush` re-arms the
# trackers the previous context's broadcasts cleared. Without the value change srevroot stays at the context's
# table (R-58) and no entry or escape is ever an edge. Under that hypothesis CS4_ZERO's 0 is the already
# recorded mechanism seen ONE ITERATION LATER, and nothing defies it.
#
# THE FIRST VERSION OF THIS ARM WAS CONFOUNDED, and the confound sat in the variable under test. It discarded
# context 1 and took an ecall to M to re-store the PCC slot before a second ENTRY. But `vctx_table_o` is
# privilege-dependent (`csr_regfile.sv:2885-2887`): at S it is `srevroot != 0`, and at M it is
# `csupctl.table`, which this arm leaves clear. So that M round trip is a 1 -> 0 -> 1 namespace change in the
# NOP and SAME arms, where srevroot is still the context's table -- two `revnode_ns_flush` edges before the
# second iteration -- and NO change at all in the ZERO arm, where srevroot is already 0 at both privileges.
# It injected the edge into exactly the two arms that must not have one and withheld it from the arm whose
# edge is the subject. Found by reading the source before the run rather than by a confusing reading after it.
#
# THE SECOND DESIGN REMOVES M FROM BETWEEN THE ESCAPES ENTIRELY. `runv_mode` is
# `sup_runv_used ? 3 : 1` (`commit_stage.sv:628-629`), so a second CSRUNV on the SAME frame RESUMES rather
# than entering, with no discard and no re-mint; options bit 0 keeps the context's ecall escape non-terminal so
# it survives to be resumed; and the context block already ends `ecall; j ctx_ecall`, so it ecalls again with
# NO change to the context's code. Between escape 1 and escape 2 the caller executes the slot, a CSR read and
# one CSRUNV, all at S, and nothing runs at M.
#
# WHAT THE ARM DOES, as built: entry 1, the slot, the escape's reading, then a RESUME on the SAME frame, the
# slot again, and its own reading in s5. The caller executes only the slot, a CSR read and one CSRUNV between
# the two escapes, all at S. The handler records the first fault's mtval and mepc and the SECOND fault's mepc
# separately, so which escape was refused is read off the address rather than inferred from a count.
#
# THE TWO ADDRESSES, recomputed from the preprocessed source for the RESUME design and IDENTICAL across the
# three variants -- which is what keeps them a matched triple: s_code holds three CSRUNVs, at +0x24 (entry 1),
# +0x34 (the resume) and +0x58 (the final discard). So escape 1 returns to **s_code+0x28** and escape 2 to
# **s_code+0x38**, and with s_code at 0x80001000 those are 0x80001028 and 0x80001038. (The withdrawn
# discard-and-re-enter design put them at +0x28 and +0x50; those numbers are dead.)
#
# PRE-REGISTERED. If the hypothesis is TRUE:
#   -DCS4_TWO -DCS4_NOP    faults 2, mepc1 = s_code+0x28, mepc2 = s_code+0x38
#   -DCS4_TWO -DCS4_SAME   faults 2, the same two addresses
#   -DCS4_TWO -DCS4_ZERO   faults 1, mepc1 = s_code+0x28, and the second record UNWRITTEN, so the printed
#                          "mepc2 minus s_code" reads minus s_code (0xffffffff7ffff000)
# A BUILT-IN CONSTRUCTION CHECK, because options bit 0 is the one thing this design depends on and a silent
# failure of it would look like a result: the frame's +16 and +24 are written from registers (ids 70 and 71,
# `capstone_dom_switcher.anvil:468-470`), not from constants, so the kind is READ here and not predicted -- it
# is scaffolding, and the MEASUREMENT is the fault count and the two mepc values. But if bit 0 did not take,
# the context's ecall escape is terminal, the context is gone, and the RESUME is REFUSED -- which the refuse
# path marks as +16 = 2 with **+24 = 29** (`:429-431`). So a reading of +24 = 29 means the arm is VOID rather
# than that the hypothesis failed, and it is distinguishable from every outcome below.
#
# Escape 1 is refused in ALL THREE because no prior value change exists yet, and the single-iteration firing
# arm -- which already refuses at escape 1 -- is the control for that half.
# OUTCOMES THAT REFUTE IT, each of which this arm can produce:
#   CS4_ZERO faults 0            the effect is WITHIN one iteration and the hypothesis is wrong; a different
#                                model is needed, and the value change acts before its own slot's fetch
#   CS4_ZERO faults 1 with mepc1 = s_code+0x38   escape 1 clean and escape 2 refused: the ordering is the
#                                opposite of the hypothesis
#   CS4_ZERO faults 2            the field effect does not reproduce in simulation at all, which would make
#                                this arm the wrong model rather than the hypothesis wrong
#   CS4_NOP or CS4_SAME faults 1 one of the two escapes is clean without any value change, which the
#                                hypothesis forbids
#
# THE RESUME DESIGN CANNOT REFUSE ESCAPE 2 AT ALL, derived from source before spending a build on it, and it
# supersedes BOTH the 2/2/2 worry below and the design above. Three facts compose:
#   1. a namespace flush sets EVERY slot's valid bit to 1 (`pmp_data_if.sv:111-114`), so it re-arms all 16;
#   2. `vctx_table_o` is 1 at S whenever srevroot != 0 and 0 at M while csupctl.table is clear
#      (`csr_regfile.sv:2885-2887`); this arm never writes CSUPCTL and the bit resets to 0 (`:3272`);
#   3. therefore EVERY trap from S to M is a 1 -> 0 vctx edge, hence a flush, hence a re-arm of CPMP0.
# Escape 1's refusal traps to M. That trap re-arms the caller's code slot in ALL THREE arms. After the mret
# nothing clears it again: the resumed context makes NO kills, because the resume returns to the PCC slot's pc
# and the context's code there is `ecall; j ctx_ecall`. So escape 2 is clean in all three and the triple reads
# **1/1/1**, which says nothing about the hypothesis. (And had the context looped back to its kill code
# instead, its fresh virtual indices would be 7..10, which cannot collide with the tracked physical 3 -- so
# that variant would read 1/1/1 too. The 2/2/2 worry below was wrong in both directions.)
#
# WHAT WOULD DISCRIMINATE: iteration 2 must clear the caller's slot AGAIN, which needs its kills to broadcast
# the tracked index again, which needs a FRESH context table with next = 3 per iteration -- so a second ENTRY,
# not a resume. A second entry needs its PCC slot armed, and arming it from M would reintroduce the very
# asymmetric edge that killed the first design. The way out is to arm BOTH frames at setup: two frames, two
# copies of the context code (the PCC is LINEAR, so it cannot cover the same region twice), two tables each
# with next = 3, and a single x26 in both (it is NONLIN, so STC copies it rather than moving it). S then runs
# entry1 -> slot -> discard1 -> entry2 -> slot with NOTHING at M in between. That is the correct experiment and
# it is a substantial piece of scaffolding, so it is proposed rather than built.
#
# A LIMITATION OF THIS ARM, written down before the run rather than used to explain a reading afterwards.
# Iteration 2's context broadcasts its OWN four kills before escape 2, so whatever the entry-2 namespace edge
# re-armed is cleared again by the time escape 2's fetch happens. If that holds, escape 2 is refused in all
# three arms and the triple reads 2/2/2 -- which is the fourth outcome above (the arm is the wrong model), NOT
# a verdict on the hypothesis. The field's module may differ from this arm in where its write sits relative to
# the context's broadcasts, and that difference rather than the namespace edge may be what carries the effect.
# 2/2/2 is still worth having, because the ns-flush trace will show when each edge fired relative to the
# broadcasts, which is what would tell us whether the field's ordering is reproducible here at all.
#
# MEASURED 2026-10-20 ON THE ITEM-2 MODEL, AND THE FIELD EFFECT DOES NOT REPRODUCE IN SIMULATION.
# All four single-iteration arms SUCCESS with EXACTLY ONE cause-1 fault:
#   -s   (no slot)            mtval 0x80001020, mepc - s_code 0x20
#   -sn  (nop)                mtval 0x80001028, 0x28
#   -sz  (csrw 0x5c1, x0)     mtval 0x80001028, 0x28, and s4 reads 0x0 -- so the slot's write DID execute
#                             after the repair, which rules out "the instruction never ran"
#   -sq  (csrw 0x5c1, t5)     mtval 0x80001028, 0x28
# (The CS4 arms sit at +0x28 rather than +0x20 because all three carry the `lla t5, tbl3` before the entry,
# which is what makes them a matched triple; the offset was computed before the run and matches.)
# So nop, a zeroing write and a same-value write are INDISTINGUISHABLE here, while the field separates them
# 2 / 0 / 2. The slot's content changes nothing in simulation.
#
# AND THE TRACE SAYS WHY, which is the first time either side of this collision has been printed anywhere on
# this project. For -s (and -sz identically in the decisive part):
#   97846   cpmp[0] 0 -> 1 by ADOPT, tracked id 0x3            the code slot adopts the physical child
#   100527  ns-flush RISE, vctx 0 -> 1, srevroot 0 -> 0x80009000   the caller's OWN pre-entry write
#   117605  cpmp[0] 1 -> 0 by CLEAR, broadcast id 0x3 against tracked 0x3   THE COLLISION, read
#   133006  ns-flush RISE, vctx 1 -> 0   the refusal's own trap into M
#   133007  cpmp[0] 0 -> 1 by NSFLUSH    the trap re-arms the slot
# Both candidate mechanisms are REFUTED by that sequence:
#   (a) a DEFERRED flush landing after the clear: the only flush before the clear is the caller's pre-entry
#       write, and the next is the trap AFTER the fetch. Nothing lands in between, and rev_busy is 0 at every
#       rise, so nothing was held.
#   (b) a namespace-GATED clear: the clear plainly happened, with `clear=1` and the broadcast id printed.
# The remaining difference is therefore field-side -- the timer and SBI traps between pb10's calls, which the
# single-iteration arm has none of.
#
# THE TWO-ITERATION ARMS ARE VOID, as their own guard was written to detect, and their readings stop after
# three values. Their second fault is at s_code+0x34, the RESUME's own CSRUNV (0x48b5165b), and it happened
# with cpmp[0]'s valid bit reading 1. **WITHDRAWN the same day, on the semantics owner's prediction and a
# reading that confirms it: the handler read mcause = 0x1d = 29 there, not 1.** That is the switcher refusing
# to resume a context the terminal escape already ended -- the void condition this arm's own guard exists for --
# and the cpmp valid bit is irrelevant to it. Nothing unexplained remains.
#
# AND AN INSTRUMENT CAVEAT THE WITHDRAWAL EXPOSED, which matters beyond this arm: **the testharness prints
# "INSTR_ACCESS_FAULT exception @" for an event whose mcause reads 29**, so the trace LABEL is not a cause and
# counting those lines is not a count of cause-1 faults. In this arm the label count is 2 and the handler's
# cause-1 reads are 1. The four single-iteration arms read 1 and 1, so their results above are unaffected --
# checked rather than assumed. The runner now reports the handler's own cause-1 reads with the label count
# beside it, so a divergence shows up instead of passing as a fault count.
#
# RUN B, ON THE FIELD-EQUIVALENT RTL (cs4-fieldrtl fc3b353b6 = c7e6f604f's synthesizable RTL): 6 of 6 arms
# as pre-registered, and it closes the entry-edge question from a third direction.
#   -s   faults 1, mtval 0x80001020, 0x20     } BYTE-IDENTICAL readings to the item-2 model, so **item 2 is
#   -sn  faults 1, mtval 0x80001028, 0x28     } INERT on the same-table arm -- measured, not argued. (The
#   -sz  faults 1, 0x80001028, 0x28, s4 = 0x0 } cycle counts differ, 134882 vs 135053 for -s, which is item
#   -sq  faults 1, mtval 0x80001028, 0x28     } 2's extra walk step and is the expected difference.)
#   -e   faults 1, mtval 0x80001018, 0x18, s3 = 0x0, s4 = 0x80009000
#   -ec  faults 0, the matched control
# -e's offset 0x18 is computed, not a surprise: SREV_ZERO's `csrw SREVROOT, x0` is one instruction where the
# default's `lla` + `csrw` is three, so CSRUNV sits at +0x14 and the slot at +0x18.
#
# SO ALL THREE ENTRY TRANSITIONS REFUSE, which is the plain re-arm model confirmed from every direction this
# family can reach:
#   tbl3 -> tbl3   no edge in either term          faults 1
#   tbl  -> tbl3   an SREVROOT-term edge           faults 1
#   0    -> tbl3   a VCTX edge, the field's srz shape   faults 1   <- -e, the case that was open
# Nothing about an entry edge protects the escape's fetch, so (a) a deferred flush and (b) a namespace-gated
# clear are refuted for the ENTRY case as well as the in-iteration one. s3 = 0x0 in -e confirms the caller
# really did enter from srevroot 0, and s4 = 0x80009000 is R-58 on a tree without item 2.
#
# WHY THE PRE-ITEM-2 READING IS THE FIELD-EQUIVALENT ONE, tied to the field by evidence rather than by the
# word "pre-item-2". The field anomaly is on tree B, whose RTL is 8de3c48cf, and
# `git diff --stat 8de3c48cf c7e6f604f -- core/` is **core/ex_stage.sv, +42 lines and nothing else** -- all 42
# inside `ifdef SUP_TRACE`, checked by walking the ifdef nesting rather than by eye. So c7e6f604f's
# SYNTHESIZABLE RTL is identical to the field bitstream's, and a reading taken on it answers the field's
# question. (Note the bitstream NAME carries 8522d72fe, which is tree A and two commits older; 8de3c48cf is
# tree B, which is what pb10 ran. Naming the tree by the image's filename would have been wrong here.)
#
# THE VARIANTS ride FS_SAMETBL, not the default arm: item 2 (id 67 in the caller's snapshot) is now in this
# tree, and it removes the default arm's refusal by changing srevroot at the escape. Only the same-table arm
# still refuses, so only it can carry this measurement.
#   -DFS_SAMETBL -DCS4_NOP    a nop in the slot          -- the field's refused shape
#   -DFS_SAMETBL -DCS4_ZERO   `csrw 0x5c1, x0`           -- the field's NOT-refused shape; note this also
#                             CHANGES srevroot (tbl3 -> 0), so it conflates "a CSR write" with "a value change"
#   -DFS_SAMETBL -DCS4_SAME   `csrw 0x5c1, tbl3`         -- a CSR write that leaves srevroot UNCHANGED. THE
#                             DISCRIMINATOR, and the reason this is three arms and not two
# PRE-REGISTERED, with what each outcome would mean:
# All three carry the same `lla t5, tbl3` before the entry, so CSRUNV sits at the same address in all three
# and `mepc - s_code` is directly comparable; only the slot's single instruction differs.
#   CS4_NOP faults 1. If it reads 0, the baseline `csrr s4, SREVROOT` in that slot is itself the special case
#     and the field's nop result does not reproduce here at all -- which would make this arm the wrong model.
#   CS4_ZERO faults 0 if the field effect reproduces, 1 if it does not. A 0 is the reproduction.
#   CS4_SAME is the question: **0 means a CSR WRITE in that slot is what matters** (its flush or
#     serialisation changing when the fetch is issued or replayed), and the srevroot VALUE is irrelevant;
#     **1 means the VALUE CHANGE is what matters**, i.e. CS4_ZERO's 0 comes from the namespace flush that
#     tbl3 -> 0 raises (`ex_stage.sv:1462`), which re-arms every tracker, and the anomaly collapses into the
#     mechanism already recorded rather than defying it.
# Candidate mechanisms to separate in the trace, since the fetch precedes decode and so cannot depend on the
# instruction: the slot was PREFETCHED past CSRUNV while the tracker was still valid and is replayed rather
# than re-fetched; or a CSR write in that slot forces a flush and re-fetch at a point where the tracker has
# been re-armed. Either way the question the trace must answer is WHEN that fetch was issued relative to the
# switch, and which tracker state it saw.
#
# FS_SAMETBL, AND WHY ITEM 1'S ACCEPTANCE READING NEEDED IT (2026-10-20, found while writing item 2's RTL).
# The default arm's caller sets srevroot = tbl and the context's frame carries tbl3, so R-58's fix -- id 67 in
# the caller's snapshot -- makes the escape RESTORE a DIFFERENT value, which raises a `revnode_ns_flush` edge
# (`ex_stage.sv:1462` compares srevroot_rev_q against its own previous sample) and re-arms every tracker. So
# item 2 removes this refusal as well, by a different mechanism, and in a bitstream carrying BOTH items a
# reading of 0 here would not say which one did it. Two changes, one reading, confounded.
#   FS_SAMETBL removes the confound by making the caller's own table BE the context's: `lla t0, tbl3`. R-58's
#   restore then writes the value srevroot already holds, srevroot_rev_q does not change, no flush fires, and
#   only item 1 can remove the refusal. The arm is otherwise byte-identical, and its readings differ from the
#   default's in exactly one position: s3 reads 0x80009000 (tbl3) instead of 0x80007000 (tbl), matching s4.
#   This is the variant item 1's acceptance must use. The default variant keeps its own job: it is the
#   acceptance reading for ITEM 2, which must take it from faults 1 to faults 0.
#
#   MEASURED on the item-2 commit, 2026-10-20, and the attribution HOLDS -- all three readings as pre-registered:
#     va-fetchsplit     faults 1 -> 0, mtval 0, s3 = s4 = 0x80007000 (restored), children 0x100 x4 (134,700).
#                       So R-58's fix removes the CSRUNV+4 refusal, by the flush, exactly as predicted.
#     va-fetchsplit-s   faults 1, mtval 0x80001020, mepc - s_code 0x20, s3 = s4 = 0x80009000 (135,053).
#                       **The refusal SURVIVES item 2** when srevroot does not change at the escape. This is
#                       the reading that keeps item 1's acceptance independent, and a 0 here would have
#                       refuted the flush attribution.
#     va-fetchsplit-sc  faults 0, the matched control one CAPNODE operand away (134,720).
#   Note the pair's flush counts: 23 / 27 / 23. The firing SAMETBL arm's four extra pipeline flushes are the
#   trap it takes, and the default arm now reads 23 like the controls, which is the fix showing up a second way.
#
# AFTER ITEM 1'S FIX this arm's firing variant must read ZERO faults with the children still 0x100, while
# `cap-fetchdeny`'s firing arm must still read ONE: the same-namespace collision is a legitimate revocation and
# must keep denying. That pair is item 1's acceptance test and is written here before the fix exists.
#
# SECOND RUN, after the repair fix (ec0d8a06b), and EVERY PRE-REGISTERED VALUE HOLDS. Both arms SUCCESS:
#   firing  (134,882 cycles, FLUSH SEEN 27)  faults 1, mtval 0x80001020, mepc - s_code 0x20, children
#           0x100 0x100 0x100 0x100, and the twenty baseline readings unchanged
#   control (134,549 cycles, FLUSH SEEN 23)  faults 0, mtval 0, the mepc record minus s_code 0xffffffff7ffff000,
#           same four children -- all 27 readings byte-identical to the previous run's control
# The control's unchanged minus-s_code reading is also the check that the inserted MINT did not move s_code
# across its `.align 12`, which would have invalidated the firing arm's absolute mtval for a reason that is not
# a defect. The repair worked, which resolves what the wedge left under-determined: a cause-1 loop is equally
# consistent with an untagged entry and with a CCSRRW that takes no effect, and the clean return separates them.
#
# THE INDICES, read rather than inferred (`-DFS_IDX`, the third variant, firing). The verdict above does not
# need them -- the pair bounds them, since child 8 is child 1 plus seven derives -- but the registry entry for
# S-23 discloses that the colliding index has never been PRINTED, and this costs no RTL: CAPPRINT on a TAGGED
# register prints the full metadata line including `Revnode_id`, which every arm here has so far only ever
# invoked on integers. The variant adds four prints and nothing else, so both measurement arms compile
# byte-identically and their readings above stand without a re-run.
#   PRE-REGISTERED: child 1's Revnode_id 3; the code capability's id after CAPNODE 3 (the same number, which is
#   what says the re-node took and the entry tracks it); child 8's id 10; tbl3's next-unused 7, so the context
#   handed out virtual 3, 4, 5, 6 -- the window containing 3 and not 10. Any other set of four values still
#   reads the collision directly; a next-unused of 3 would mean the context allocated nothing and would void
#   the children's 0x100, which it contradicts.
#   The virtual side is read from the TABLE, from the caller, after the escape -- not with a print inside the
#   context -- so the context's instruction stream is unchanged and nothing can fault in there.
#   READ 2026-10-20, every pre-registered value exact (SUCCESS 135,077 cycles; faults 1, mtval 0x80001020,
#   mepc - s_code 0x20, children 0x100 x4, so the measurement is unchanged by the three added instructions):
#     Reg[16] a6, physical child 1   Revnode_id 3   0x80006000..0x80006040  perm 14
#     Reg[11] a1, the code cap after CAPNODE  Revnode_id 3   0x80000000..0x80003000 = [_start, ctx)  perm 7
#     Reg[16] a6, physical child 8   Revnode_id 10  same bounds
#     tbl3's next-unused               7, so the context handed out virtual 3, 4, 5, 6
#   So the firing arm's CPMP0 tracks PHYSICAL 3 and the context broadcast VIRTUAL 3, 4, 5, 6: 3 collides under
#   an index-only compare and the fetch is refused, while the control's PHYSICAL 10 lies outside the window and
#   is not. The 2-versus-3 difference is now a reading. Note the two numbers that MATCH are a1's and child 1's,
#   which is the separate fact that the re-node took; the collision itself is 3 against 3 ACROSS namespaces.
#
# Below this line, va-fetchctx's header, whose context table and fault-record design this arm keeps:
#-----------------------------------------------------------------------------
# va-fetchctx.S -- R-58's FETCH face END TO END: a CONTEXT's own kill broadcast, no namespace flush at the
#                  escape, and the caller's FIRST FETCH at CSRUNV+4 (2026-10-20)
#-----------------------------------------------------------------------------
#
# WHAT THIS ADDS. `cap-fetchdeny` measured that a broadcast clearing a CPMP tracker refuses the next fetch
# through that entry (cause 1 at the following instruction, matched control naming a different index silent), and
# `va-srevroot` measured that the caller keeps the context's srevroot so no namespace flush fires at the escape.
# Each step of the semantics owner's fetch-face candidate is therefore sourced or measured -- but by a caller's
# own broadcast in the physical namespace, not a context's, so no single arm has run the field's chain. This arm
# runs it: the broadcast comes from inside a virtual context, in the context's namespace, and the measurement is
# the caller's first fetch after the escape.
#
# ALSO AN ACCEPTANCE TEST for the R-58 fix, named before the fix exists. With id 67 in the caller's snapshot and
# restored at the escape, `vctx_table_o` falls at the escape, the flush re-arms every tracker, and the firing arm
# must read ZERO faults with the context's children still revoked.
#
# THE TABLE, which is what the first attempt got wrong. `va-fetchkill`'s context CDERIVE refused with status 2,
# cause 30 and a kind-5 PAUSED escape, because `vp_alloc_slot`'s node-pressure check reserves 256 identities
# inside a context: a 4 KiB table page holds at most 256 records, so `capacity - next + free <= 256` always
# holds and a 4 KiB page can NEVER satisfy a context allocation. tbl3 is 16 KiB, capacity 1024, next-unused 3,
# so 1021 available. (An uncreated parent would have given status 1 and cause 25, not 30 -- that was checked at
# the source after the semantics owner offered it as the explanation, and refuted.)
#
# THE ARMS, differing in ONE CAPNODE operand:
#   firing   the code entry (CPMP0) is re-noded onto the caller's FIRST physical child -- the physical pool's
#            bump pointer starts at 3, so index 3, inside the 3..6 window the context sweeps
#   control  re-noded onto the EIGHTH physical child, index 10, outside that window
# The context derives four children from the MANAGE|RW capability it inherits and revokes each, broadcasting its
# own virtual indices 3..6. Both arms run identical context code and carry identical broadcast traffic; only the
# caller slot's tracked index differs. The CPMP tracker compare is index-only (`pmp_data_if.sv:107`), so a
# virtual index collides with a physical one, which is the cross-namespace step the field's chain needs.
#
# THE FAULT RECORD LIVES IN THE FRAME, at +960/+968/+976 through s9 -- above the hidden words at +928..+944 and
# below the caller's snapshot at +1024. Not in registers: this scaffolding leaves exactly ONE register free (a6),
# and the two previous arms were voided by using registers it holds live (s1 the second table's capability, s2
# the caller's marker, s6 satp, s7 the page-table capability). The handler counts the fault, records its mtval
# and mepc once, repairs the entry with a different node -- the adopt fires only on an id CHANGE -- and returns,
# so the caller reaches its ecall and the baseline readings survive the fault.
#
# FIRST RUN: the CONTROL read exactly as pre-registered -- context ran, four children revoked (0x100 four times,
# so the broadcasts went out), the caller read back the context's table (0x80008000, R-58 again), ZERO fetch
# faults with the code slot's index outside the swept window. The FIRING arm WEDGED in a cause-1 trap loop
# (82,282 handler entries, 329,134 flush rises, one reading) because the repair's CAPNODE source was a0, which
# this scaffolding leaves holding an integer -- see the handler. No verdict was taken from it.
#
# SECOND RUN: the control reads IDENTICALLY again (the twenty baseline readings, four children at 0x100, zero
# faults) and the FIRING ARM WEDGED AGAIN WITH BYTE-IDENTICAL NUMBERS -- 329,134 flush rises, 4,000,013 cycles,
# one reading, the same to the digit as before the s9 fix. **That identity is the finding**: the s9 repair
# changed a line whose effect is discarded, because the repair cannot run at all.
#   `csr_regfile.sv:3018-3021` makes a CCSRRW to any of 0x010..0x01F a PRIVILEGE VIOLATION while `sup_active_q`
#   is 1, and `CCSR_CPMP(n) = 0x10 | n` is exactly that range. So the handler cannot repair a CPMP entry while a
#   context is live, the entry stays invalid, and the refused fetch repeats forever.
#
# AND THE COLLISION FIRES INSIDE THE CONTEXT, which is a face neither lane had considered. The caller's CPMP0
# grant covers the whole image, so it covers the CONTEXT's code too: the context's own kill broadcast clears the
# entry **the context itself is fetching through**, and the fault is the context's, not the caller's. In the
# field that would present as a context unable to make progress -- neither the caller's data face nor its fetch
# face. This arm therefore CANNOT measure the caller's-first-fetch case until the two coverages are split: the
# caller's S code under the tracked entry, the context's code under a different one, which needs ctx in its own
# range with its own CPMP slot. DEPRIORITISED rather than redesigned, because the surviving field route is now
# known to be retire()'s own select_root window, which cap-fetchdeny already models almost exactly (the field's
# broadcast there is the caller's own too, differing only in namespace).
#
# PRE-REGISTERED. va-srevroot's twenty readings unchanged in both arms, then seven more:
#   firing   faults 1; mtval the physical address of CSRUNV+4; mepc - s_code 4; children 0x100 0x100 0x100 0x100
#   control  faults 0; mtval 0; mepc - s_code reads minus s_code; children 0x100 0x100 0x100 0x100
# The children must read 0x100 in BOTH arms: that is what says the broadcasts went out, and it is the reading
# that distinguishes a real negative in the control from a context that never allocated. A child at 0x101 means
# its CREVOKE did not take; 0 means the block never ran, which is what the void first attempt read.
# If BOTH arms read zero faults with the children at 0x100, the cross-namespace collision does not reach the
# caller's trackers and the field's fetch face needs another explanation -- that is the outcome this arm is built
# to be able to produce. If both FIRE, the index inference is wrong and the arm says nothing about collisions.
#
# -DVA_RC_CTX (Section 5 phase 2, the record cache's CONTEXT-EDGE control, 2026-11-16; with -DFS_SAMETBL -DCS4_TWO).
# The cache holds records only inside a context and empties on that flag's edge, because outside one software may
# rewrite a record with a plain store. Here the caller keeps the context's own table in srevroot (FS_SAMETBL), so the
# escape and the resume move NO namespace term: only the context edge can empty the cache. The context derives a
# child of x26 (record 2 read, cached), ecalls (non-terminal), and the caller at S rewrites record 2's word 1 to depth
# 5 with a plain store (SUM set for the U page); after the RESUME the context derives again. Each child's word 1
# carries its parent's depth field, so the second child (slot 4) shows whether record 2 was read after the store:
#   correct, read on P1 d83d3a633 and on the fix   3; 0x1000000000000040; 0x1000000000000045 (slot 4 = slot 3 | 5)
#   a stale record 2 would give slot 4 = slot 3 = 0x1000000000000040
#   RC_SIM_NOCTXFLUSH (sim mutant)  the oracle's rc_stale_hit at slot 2, pa tbl3 + 0x20, before the reading
#   RC_SIM_NOINV (sim mutant)       the oracle's rc_hit_mismatch at slot 2 (value differs), before the reading
# (The first version of this note predicted depths 1 and 6 from a +1 rule; the measured field is the parent's depth
# ORed into the child's word 1. Corrected from the P1 reading, which has no cache.)
# -DVA_RC_TRAP keeps the arm running past the readings into S's ecall (runtime lane's matched pair with -DCS4_ZERO).
# -DVA_RC_BARE (R-75's discriminating arm, 2026-11-17): S sets srevroot (with FS_SAMETBL, to tbl3) and ecalls with
# a2 = 0x77. No context is ever entered (only the M-mode CSRUNV control, which is illegal there). A trap below M
# here means the defect needs no context at all.
# Printed at S after the resume's escape: its kind (3), then tbl3 slot 3 and slot 4 word 1 (depth in bits [16:0]).
#
# Below this line, va-srevroot.S's own header, which is this arm's scaffolding and whose readings are its control:
#-----------------------------------------------------------------------------
# va-srevroot.S -- R-58's arm: does the caller get its OWN lifetime-table base back after a virtual context
#                  escapes, or does it inherit the CONTEXT's? (2026-10-19)
#-----------------------------------------------------------------------------
#
# DERIVED FROM va-runv.S by six asserted edits, so everything below the added readings is that arm's known-good
# scaffolding and its readings are the control: if any of them moves, this arm's modification broke something
# rather than measuring it. The edits are a second table page, the frame's srevroot word pointing at it, the
# caller's own write and two read-backs of 0x5C1 at S, the CSNODE variant, five added CAPPRINTs, and the zeroing
# of s3, s4 and s5.
#
# THE CLAIM UNDER TEST (R-58, filed source-derived and NOT yet measured). The switcher's register map was
# extended for item D with id 67 (srevroot), but the SAVE/RESTORE window was not: every walk and the exchange
# cover 0..66 (`capstone_dom_switcher.anvil:527-538`). Id 67 is written in exactly three places, all
# context-side: the ENTRY load from the frame's second word (`:170-173`), the ESCAPE save to the hidden word
# (`:481`) and the RESUME restore from it (`:315`). The only other writer is the CSR instruction
# (`csr_regfile.sv:2014`; the switcher's is `:2051`), and there is no caller-side shadow analogous to
# `runv_caller_priv_q`. So the caller's value should never come back.
#
# THE ARM. The caller at S writes 0x5C1 = A (its own table, `tbl`), the frame carries B (`tbl2`), the context
# escapes, and the caller reads 0x5C1 back. B CONFIRMS R-58; A REFUTES it.
#   -DSREV_ZERO     A = 0, so the reading also says whether the inherited value is the frame's word verbatim
#   -DSREV_CSNODE   a CSNODE at S before the entry and again after the escape. CSNODE is gated at decode on the
#                   table-namespace flag, so with A = 0 it should be illegal before and legal after -- the
#                   namespace flip, which is R-58's consequence rather than the defect itself
#
# PRE-REGISTERED READINGS, the five added ones, printed after va-runv's own:
#   s3  = tbl's physical base      A, as the caller wrote it (0 under -DSREV_ZERO). Also the CONTROL that S can
#                                  write 0x5C1 at all: an illegal-instruction trap here voids the arm.
#   s4  = tbl2's physical base     THE READING, if R-58 holds. tbl (or 0 under -DSREV_ZERO) refutes it.
#   tbl                            printed so the comparison needs no external value
#   tbl2                           likewise
#   s5  = 0                        the CSNODE variant is not compiled in by default
# Every va-runv reading above them must be unchanged.
#
# WHAT IT READ, run srv on d78265520 at S12_MEM_DELAY=12, with va-runv itself in the same load as the baseline.
# **R-58 IS CONFIRMED.** s3 read 0x80006000 (tbl, the caller's own A, so S can write 0x5C1 and the control
# holds) and **s4 read 0x80007000 -- tbl2, B, the CONTEXT's table.** Under -DSREV_ZERO, with A = 0, s4 read
# 0x80007000 again, so the caller inherits the frame's word verbatim rather than keeping anything of its own.
# -DSREV_CSNODE demonstrated the consequence: the SAME CSNODE instruction read mcause 2 (illegal at decode)
# BEFORE the entry and executed AFTER the escape, which is the namespace flip.
#
# AND THE ASSUMPTION THIS HEADER NAMED DID FAIL on the first run, exactly as written: tbl2 was left all zeros, so
# the context's load through x26 (node 2) raised cause 25 in the new namespace instead of reaching its ecall --
# +24 read 0x19 against va-runv's 0xb, the pc and address readings moved, and data[8] stayed 0. The srevroot
# readings never depended on the context completing (the escape is terminal either way and the switcher's walk is
# identical) but the CONTROL did, and a control differing from its baseline in two variables is not one.
#
# RE-RUN WITH tbl2 INITIALISED BYTE-IDENTICALLY TO tbl, run srv2 on 57c277f4e, and the control is now EXACT:
#   va-runv        0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9
#   va-srevroot    0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb  0x80006000 0x80007000
#                  0x80006000 0x80007000 0x0  0x0 0x9
# -- the first thirteen and the last two byte-identical to the baseline, so this arm is va-runv plus exactly one
# variable, and then:
#   s3 = 0x80006000   tbl, the caller's own A, read straight back: S can write 0x5C1 and the control holds
#   s4 = 0x80007000   tbl2 -- THE CONTEXT'S TABLE. **R-58 CONFIRMED.**
#   -DSREV_ZERO       s3 = 0x0 and s4 = 0x80007000 again: the caller inherits the frame's word VERBATIM
#   -DSREV_CSNODE     three leading 0x2 instead of one -- the legacy-C control, then the PRE-ENTRY CSNODE
#                     illegal at decode -- and s5 = 0x4, the POST-ESCAPE CSNODE succeeding with a node id. The
#                     namespace flip, illegal before and successful after, with every other reading still the
#                     baseline's.
#
# Below this line, va-runv.S's own description, unchanged:
#
# The monitor (M, capability mode) builds a RECYCLE lifetime table (record 2 the root), identity page tables with USER
# leaves (the S-mode code's page without U, since S cannot fetch from a user page), CPMP0 over the whole image (S-mode
# code is confined by the capability PMP in capability mode), and a 4 KiB frame (the caller's FP state at +2048): +0 satp, +8 the
# table, +56 options 0, +64 the PCC (a tagged LINEAR RWX capability over `ctx`), +480 the x26 slot (a tagged NONLIN RW
# over `data`, node 2). It writes 0x77 into data[0], then mret's into scalar S code, which runs CSRUNV rd=a2, rs1=the
# frame's physical address, rs2=0. The context (C mode, translated at U) loads data[0] through x26, stores it at
# data[8], sets a0 = 0x55 and x11 = the loaded word, and executes ecall: options bit 0 is clear, so the escape is
# terminal (kind 2, cause 11). S resumes at CSRUNV+4 with a2 = 2 and ecalls into M, which reads the frame through the
# caller's restored s9, then S runs action 1 (discard, a2 = 0). First, in M after CAPENTER (legacy C), CSRUNV is
# illegal: the control reads mcause 2 from the trap handler.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   mcause 2 (the legacy-C control); then after the first escape: a2 (the kind) 2; the caller's s2 0x1234 (restored
#   from the frame's second kilobyte); the frame's +16 kind 2, +24 cause 11, +32 pc minus ctx_ecall 0, +40 addr 0,
#   +48 raw a0 0x55; the PCC slot's cursor at +64 minus ctx_ecall 0; the x11 slot at +240 0x77; data[8] 0x77;
#   csupstatus 5 and csupcause 11 (the escape's CSR record, written beside the frame); then after action 1: a2 0;
#   mcause (the handler's own, the second ecall) 9.
# -DVA_QUANTUM: csupquantum 3000 and the context loops: the escape is kind 1 (the quantum): a2 1; s2 0x1234; +16 1;
#   +24 0 (QEMU's quantum event; csupcause keeps the mark); +32 minus ctx_loop 0; +40 0; +48 0 (a0 untouched); PCC cursor minus
#   ctx_loop 0; +240 0; data[8] 0 (never reached); csupstatus 3; csupcause 0x8000000000000010; then action 1: 0; 9.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2

# The instruction occupying the slot at CSRUNV+4, as a macro so it can appear after BOTH entries of CS4_TWO
# without duplicating the selection. Empty when no CS4_* is defined, which is what leaves the five existing
# arms byte-identical. Macro names are kept out of comments: cpp expands them there too.
#if defined(CS4_NOP)
#define CS4_SLOT nop
#elif defined(CS4_ZERO)
#define CS4_SLOT csrw SREVROOT, x0
#elif defined(CS4_SAME)
#define CS4_SLOT csrw SREVROOT, t5
#else
#define CS4_SLOT
#endif

#define CSNODE(rd, rs1)       .insn r 0x5B, 0x1, 0x54, rd, rs1, x0
#define CSUPQUANTUM 0x7C3
#define CSUPCTL     0x7C4
#define CSUPSTATUS  0xFC0
#define CSUPCAUSE   0xFC1
#define SREVROOT    0x5C1

#define PTE_LEAF_RWX  0xDF
#define PTE_LEAF_SRWX 0xCF
#define PTE_PTR       0x01

.option norvc

.section .text.init
.globl _start
_start:

li gp, 1
la t0, trap_handler
csrw mtvec, t0
csrw mcause, x0

li s3, 0
li s4, 0
li s5, 0
lla a0, _start
lla a1, _end_of_text
CAPENTER(a0, a1)

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

# ---------------------------------------------------------------- the table: RECYCLE, next 3, record 2 the root
MINT(s8, CAP_TYPE_NONLIN, tbl, tbl_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 64
sd t0, 0(s8)
li t0, 3
sd t0, 8(s8)
sd x0, 16(s8)
sd x0, 24(s8)
li t0, 32
CINCOFFSET(t1, s8, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)

# ---------------------------------------------------------------- the CONTEXT's table, byte-identical to tbl
# The first run of this arm left tbl2 all zeros, and the context's load through x26 (node 2) then raised cause 25
# in the new namespace -- which the header had named as the assumption most likely to fail, and it did. The
# readings that matter did not depend on it, but the control did: va-runv's kind/cause/pc/data readings have to
# be reproduced EXACTLY for this arm to be that arm plus one variable. So tbl2 gets the same RECYCLE header and
# the same record 2 as the root, written through its own capability in s1.
MINT(s1, CAP_TYPE_NONLIN, tbl2, tbl2_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 64
sd t0, 0(s1)
li t0, 3
sd t0, 8(s1)
sd x0, 16(s1)
sd x0, 24(s1)
li t0, 32
CINCOFFSET(t1, s1, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)

# ---------------------------------------------------------------- data, and the context's capability over it (node 2)
MINT(x26, CAP_TYPE_NONLIN, data, data_end, 14)
li t0, 0x77
sd t0, 0(x26)
sd x0, 8(x26)

# ---------------------------------------------------------------- page tables as va-alloc.S; the S-mode page without U
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
# L0[s_code] -> no U
lla t0, s_code
li t1, 0x80000000
sub t0, t0, t1
srli t0, t0, 12
slli t0, t0, 3
add t0, t0, a7
CINCOFFSET(t2, s7, t0)
lla t1, s_code
srli t1, t1, 12
slli t1, t1, 10
ori t1, t1, PTE_LEAF_SRWX
sd t1, 0(t2)
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
# the capability PMP: S-mode code is confined by CPMP in capability mode; grant the whole image
# THE COVERAGE SPLIT, which is what va-fetchctx lacked. CPMP0 covers [_start, ctx) -- the caller's S code and
# the M handler's text -- and carries the TRACKED node. CPMP1 covers [ctx, pt_end): the context's code and the
# whole of .data, with CAPCREATE's default node 2, which a context table starting at 3 can never broadcast. So
# the context's fetches and every data access on both sides survive the collision, and the ONLY thing the
# cleared tracker can refuse is a fetch of the caller's own code -- which is the measurement.
MINT(a1, CAP_TYPE_NONLIN, _start, ctx, CAP_PERM_RWX)
MINT(a6, CAP_TYPE_NONLIN, ctx, pt_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(1), a6)
# The code slot must track a node whose INDEX the context's own allocations can collide with. x26 carries
# MANAGE|RW, so the caller derives physical children from it here; a6 is the one register this scaffolding leaves
# free, so it rolls. The firing arm re-nodes the code entry onto the FIRST physical child (the pool's bump
# pointer starts at 3, so index 3, inside the 3..6 window the context sweeps); the control re-nodes it onto the
# EIGHTH (index 10, outside that window). One CAPNODE operand is the whole difference between the arms.
li t0, 0x40
CDERIVE(a6, x26, t0)           # physical child 1
#ifndef FC_CTRL
CAPNODE(a1, a6)                # FIRING: the code entry tracks the index the context will broadcast
#ifdef FS_IDX
CAPPRINT(a6)                   # IDX: physical child 1's Revnode_id -- the index the CPMP0 entry will track
CAPPRINT(a1)                   # IDX: the code capability AFTER the re-node -- CPMP0's tracked id, must match
#endif
#endif
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)
li t0, 0x40
CDERIVE(a6, x26, t0)           # physical child 8
#ifdef FS_IDX
CAPPRINT(a6)                   # IDX: physical child 8's Revnode_id -- the CONTROL arm's tracked index
#endif
#ifdef FC_CTRL
CAPNODE(a1, a6)                # CONTROL: an index the context does NOT broadcast
#endif
CCSRRW(x0, CCSR_CPMP(0), a1)
# THE HANDLER'S REPAIR CAPABILITY, parked in a6 and in BOTH arms so they still differ in one operand. a6 is free
# from here on (the CDERIVE roll above is its last write, and nothing below this line touches it), it is restored
# by the switcher at the escape with the rest of the caller's registers, and its node is CAPCREATE's 2 -- which
# differs from the tracked index, so the adopt fires on the id change (pmp_data_if.sv:88-94).
MINT(a6, CAP_TYPE_NONLIN, _start, ctx, CAP_PERM_RWX)
lla t0, pt_root
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
mv s6, t0                      # the satp value, for the frame (s6 = x22; s10 is x26, the context's capability)

# ---------------------------------------------------------------- the frame
MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
sd s6, 0(s9)                   # satp
# tbl3 is the CONTEXT's table and must advertise a capacity above 256 plus its next-unused index, because
# vp_alloc_slot's in-context reserve is 256: a 4 KiB page holds 256 records and can never satisfy that, which is
# what refused va-fetchkill's context CDERIVE with status 2, cause 30 and a kind-5 paused escape. 16 KiB holds
# 1024 records, so 1024 - 3 = 1021 available against the reserve.
MINT(s0, CAP_TYPE_NONLIN, tbl3, tbl3_end, CAP_PERM_RW)
li t0, 1
slli t0, t0, 63
ori t0, t0, 1024               # the RECYCLE flag and capacity 1024 (slot 0's prev, read from bits [15:0])
sd t0, 0(s0)
li t0, 3
sd t0, 8(s0)                   # next-unused 3
sd x0, 16(s0)                  # slot 1: the free list, empty
sd x0, 24(s0)
li t0, 32
CINCOFFSET(t1, s0, t0)
sd x0, 0(t1)
li t0, 1
slli t0, t0, 60
sd t0, 8(t1)                   # record 2 valid at generation 0: x26's node, the context's MANAGE parent
lla t0, tbl3
sd t0, 8(s9)                   # srevroot := the CONTEXT's table, a real one it can allocate from
#ifdef CS4_TWO
li t0, 1
sd t0, 56(s9)                  # options bit 0: the ecall escape is NOT terminal, so the context survives for
                               # the resume, and its own `j ctx_ecall` makes it ecall again unchanged
#else
sd x0, 56(s9)                  # options 0
#endif
MINT(a2, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
STC(s9, a2, 64)                # the PCC slot
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, x26, 0)                # the x26 slot (+64 + 16 * 26)

#ifdef VA_QUANTUM
li t0, 3000
#else
li t0, 0x100000
#endif
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the control: CSRUNV in legacy C is illegal (2)
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # traps: the handler prints mcause (2) and steps over it

# ---------------------------------------------------------------- into scalar S
li s2, 0x1234
li t0, 0x1800                  # MPP := 01 (S)
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0
#ifdef VA_RC_CTX
li t0, 0x40000                 # SUM: the caller's plain store into tbl3 (a U page) between the two entries
csrs mstatus, t0
#endif
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
# R-58's arm. The caller sets its OWN lifetime-table base to A, enters a context whose frame carries B, and reads
# 0x5C1 back after the escape. B confirms the defect (nothing restores the caller's srevroot), A refutes it.
#ifdef SREV_ZERO
  csrw SREVROOT, x0            # A = 0: the namespace-flip variant
#elif defined(FS_SAMETBL)
  lla t0, tbl3
  csrw SREVROOT, t0            # A = the CONTEXT's table: see FS_SAMETBL in the header
#else
  lla t0, tbl
  csrw SREVROOT, t0            # A = the caller's own table
#endif
  csrr s3, SREVROOT            # A as the caller left it; also the control that S can write 0x5C1 at all
#ifdef VA_RC_BARE
  li a2, 0x77                  # R-75's bare arm: NO CSRUNV in the run. S holds srevroot = A and ecalls at once
  j s_bare_ecall
#endif
#ifdef SREV_CSNODE
  CSNODE(s5, s8)               # BEFORE the entry: illegal at decode while the caller is out of the namespace
#endif
#if defined(CS4_NOP) || defined(CS4_ZERO) || defined(CS4_SAME)
  lla t5, tbl3                 # CS4: materialised BEFORE the entry, in ALL THREE variants, so the three share
                               # one layout and differ ONLY in the single instruction at CSRUNV+4. t5 is x30,
                               # unused anywhere else in this arm, and restored with the caller's snapshot.
#endif
lla a0, frame
li a1, 0
CSRUNV(a2, a0, a1)             # the entry; S resumes here with a2 = the kind
  CS4_SLOT
  csrr s4, SREVROOT            # THE READING: B means the caller inherited the context's table
#ifdef VA_RC_CTX
  lla t1, tbl3
  li t0, 1
  slli t0, t0, 60
  ori t0, t0, 5
  sd t0, 40(t1)                # record 2 word 1 := valid, depth 5, behind the cache; srevroot stays tbl3
#endif
#ifdef SREV_CSNODE
  CSNODE(s5, s8)               # AFTER the escape: legal if the caller is now inside the table namespace
#endif
#ifdef CS4_TWO
#ifdef VA_RC_CTX
  li a1, 2                     # VA_RC_CTX: the ecall paused the context as kind 3 (a service request), which only
                               # action 2 resumes (commit_stage.sv:274-283); action 0 is refused with 29
#else
  li a1, 0
#endif
  CSRUNV(a2, a0, a1)           # RESUME, mode 3: sup_runv_used is set, so the same frame resumes rather than
                               # entering (commit_stage.sv:628-629). NOTHING runs at M between the two
                               # escapes, which is the whole point -- see the header.
  CS4_SLOT
  csrr s5, SREVROOT            # the SECOND escape's srevroot (this arm prints s5, which is 0 otherwise)
#ifdef VA_RC_CTX
  # The readings are taken HERE, at S, and the arm ends here: S's ecall after a service resume's second escape
  # traps below M on P1 and B5a alike (an illegal mcause read at the handler, in a loop), which is not this
  # arm's question.
  CAPPRINT(a2)                 # RC: the second escape's kind, 3
  lla t1, tbl3
  ld t2, 56(t1)
  CAPPRINT(t2)                 # RC: tbl3 slot 3 word 1, the first child: depth 1
  ld t2, 72(t1)
  CAPPRINT(t2)                 # RC: tbl3 slot 4 word 1, the second child: depth 6 (record 2 read fresh)
#ifndef VA_RC_TRAP
  lla t1, tohost
  li t2, 1
  fence
  sw t2, 0(t1)
1: j 1b
#endif
#endif
#endif
s_bare_ecall:
mv a3, a2
li a7, 1
ecall                          # to M: the readings
lla a0, frame
li a1, 1
CSRUNV(a2, a0, a1)             # action 1: discard
mv a3, a2
li a7, 2
ecall
1: j 1b

.align 12
trap_handler:
#ifdef R75_PRINT
  # R-75's discriminating reads (2026-11-20): the four S-level trap CSRs, read FIRST, at whatever privilege the handler
  # runs. They are legal at S, so the retirement trace's privilege digit and values say whether the trap was DELEGATED
  # (privilege 1, scause = the ecall's cause, sepc = s_bare_ecall's ecall, stvec = this handler) or taken to M with the
  # privilege then lowered (scause/sepc unrelated to the ecall). Clobbers t3-t6; only the first trap is read.
  # The li comes first: it retires at ANY privilege, so its retirement line carries the handler's privilege digit
  # even when every CSR read after it is refused (the supervision gate refuses [9:8] != 0 at S and M alike).
  li t3, 0x75
  csrr t3, scause
  csrr t4, sepc
  csrr t5, stvec
  csrr t6, sstatus
#endif
  csrr t0, mcause
  li t1, 1
  bne t0, t1, 9f
  # THE FETCH FACE. A CPMP fetch deny at S raises cause 1 with tval = the physical fetch address
  # (pmp_data_if.sv:238-247), and the fetch check reads the same cpmp_revnode_valid_q array as the data check
  # (:171 cached, :185 on the scan). The record goes in the FRAME through s9, at +960/+968/+976 -- above the
  # hidden words at +928..+944 and below the caller's snapshot at +1024 -- because this scaffolding leaves only
  # one register free and the two previous arms were voided by using registers it holds live. Then repair the
  # entry with a DIFFERENT node, since the adopt fires only on an id change, and return: the caller runs on to
  # its ecall and the baseline readings survive the fault.
  ld t2, 960(s9)
  addi t2, t2, 1
  sd t2, 960(s9)
  li t1, 1
  bne t2, t1, 8f
  csrr t1, mtval
  sd t1, 968(s9)
  csrr t1, mepc
  sd t1, 976(s9)
8:
#ifdef CS4_TWO
  li t1, 2
  bne t2, t1, 2f
  csrr t1, mepc
  sd t1, 984(s9)               # the SECOND fault's mepc: which escape was refused is read off the address
2:
#endif
  # The repair's source must be a capability that is PROVABLY LIVE AT M HERE, and a CODE capability: the entry it
  # replaces authorises the caller's fetches. Both earlier attempts failed on the same mistake, which is worth
  # naming because it is the third instance in this lane: they chose a register by what it was last SEEN holding
  # instead of enumerating every write to it.
  #   `CAPNODE(a1, s9)` -- a1 IS x11, and the S code writes `li a1, 0` as the CSRUNV ACTION argument two
  #   instructions before the faulting fetch (s_code+0x18, +0x1c, +0x20), so CAPNODE operated on an untagged register and the CCSRRW
  #   installed an untagged entry, which authorises no fetch at all: 82,277 handler entries and 329,126 flush
  #   rises in a cause-1 trap loop. Before that, a0, which the same block loads with `lla a0, frame`.
  # a6 is minted over [_start, ctx) at the CPMP0 install and never written again, so it is live by construction,
  # and its node 2 differs from the tracked index, which is what the adopt requires.
  CCSRRW(x0, CCSR_CPMP(0), a6)
  mret
9:
  li t1, 2
  bne t0, t1, 3f
  CAPPRINT(t0)                 # 2: the legacy-C control
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
3:
  li t1, 1
  bne a7, t1, 4f
  CAPPRINT(a3)                 # the kind: 2 (quantum variant: 1)
  CAPPRINT(s2)                 # 0x1234: the caller's registers came back
  ld t2, 16(s9)
  CAPPRINT(t2)                 # +16 kind
  ld t2, 24(s9)
  CAPPRINT(t2)                 # +24 cause
  ld t2, 32(s9)
#ifdef VA_QUANTUM
  lla t3, ctx_loop
#else
  lla t3, ctx_ecall
#endif
  sub t2, t2, t3
  CAPPRINT(t2)                 # +32 pc: 0
  ld t2, 40(s9)
  CAPPRINT(t2)                 # +40 addr: 0
  ld t2, 48(s9)
  CAPPRINT(t2)                 # +48 raw a0
  ld t2, 64(s9)
  sub t2, t2, t3
  CAPPRINT(t2)                 # the PCC slot's cursor: 0
  ld t2, 240(s9)
  CAPPRINT(t2)                 # the x11 slot
  ld t2, 8(x26)
  CAPPRINT(t2)                 # data[8]
  csrr t2, CSUPSTATUS
  CAPPRINT(t2)                 # 5 / 3
  csrr t2, CSUPCAUSE
  CAPPRINT(t2)                 # 11 / the quantum mark
  CAPPRINT(s3)                 # A, as the caller wrote it: tbl's physical base (0 under SREV_ZERO)
  CAPPRINT(s4)                 # THE READING: tbl2 confirms R-58, tbl (or 0) refutes it
  lla t2, tbl
  CAPPRINT(t2)                 # A's address, so the comparison needs no external value
  lla t2, tbl2
  CAPPRINT(t2)                 # B's address
  CAPPRINT(s5)                 # the CSNODE variant's node id, 0 when it is not compiled in
  ld t2, 960(s9)
  CAPPRINT(t2)                 # THE READING: the number of cause-1 fetch faults at the caller's first fetch
  ld t2, 968(s9)
  CAPPRINT(t2)                 # the first fault's mtval, the physical fetch address
  ld t2, 976(s9)
  lla t3, s_code
  sub t2, t2, t3
  CAPPRINT(t2)                 # the first fault's mepc minus s_code: 4 predicted, the instruction after CSRUNV
#ifdef CS4_TWO
  ld t2, 984(s9)
  lla t3, s_code
  sub t2, t2, t3
  CAPPRINT(t2)                 # the SECOND fault's mepc minus s_code, or minus s_code if there was none
#endif
  ld t2, 16(x26)
  CAPPRINT(t2)                 # the context's child 1 after its CREVOKE: 0x100 revoked, 0x101 live, 0 unwritten
  ld t2, 24(x26)
  CAPPRINT(t2)                 # child 2
  ld t2, 32(x26)
  CAPPRINT(t2)                 # child 3
  ld t2, 40(x26)
  CAPPRINT(t2)                 # child 4
#ifdef FS_IDX
  ld t2, 8(s0)
  CAPPRINT(t2)                 # IDX: tbl3's next-unused AFTER the context's four allocations. The virtual
                               # namespace does not recycle (capstone_rev_node.anvil:528-544: the kill's push
                               # is gated on vctx_q == 0), so next is monotone and the four indices it handed
                               # out are [3, this). s0 is the tbl3 capability, written once at the mint and
                               # never again, so it is live here.
#endif
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret
4:
  CAPPRINT(a3)                 # 0: the discard
  csrr t0, mcause
  CAPPRINT(t0)                 # 9
  CAP_PASS(s11)

# page-aligned so the two CPMP ranges meet at a clean boundary and the split is readable in a disassembly
.align 12
ctx:
#ifdef VA_QUANTUM
ctx_loop:
  j ctx_loop
#elif defined(VA_RC_CTX)
  li t0, 0x40
  CDERIVE(t3, x26, t0)         # entry 1: reads record 2 (cached), child slot 3 at depth 1
  ecall                        # non-terminal (CS4_TWO's options bit 0); the RESUME continues below
  li t0, 0x40
  CDERIVE(t3, x26, t0)         # after the resume: record 2 again, child slot 4 at the parent's depth + 1
ctx_ecall:
  ecall
  j ctx_ecall
#else
  ld t1, 0(x26)
  sd t1, 8(x26)
  li a0, 0x55
  mv x11, t1
  # Four children of x26 (MANAGE|RW), each revoked at once: the kills broadcast the context's own virtual
  # indices 3..6. CSRETIRE and CSNODE are illegal inside a running context, so the broadcast has to come from
  # CDERIVE plus CREVOKE, which carry no decode gate. CREVOKE has no rd, so each kill is recorded by querying the
  # child's validity afterwards and storing it with bit 8 set -- 0x100 revoked, 0x101 still live, 0 never written.
  li t0, 0x40
  CDERIVE(t3, x26, t0)
  CREVOKE(x26, t3)
  LCC(t6, t3, x0)
  ori t6, t6, 0x100
  sd t6, 16(x26)
  li t0, 0x40
  CDERIVE(t3, x26, t0)
  CREVOKE(x26, t3)
  LCC(t6, t3, x0)
  ori t6, t6, 0x100
  sd t6, 24(x26)
  li t0, 0x40
  CDERIVE(t3, x26, t0)
  CREVOKE(x26, t3)
  LCC(t6, t3, x0)
  ori t6, t6, 0x100
  sd t6, 32(x26)
  li t0, 0x40
  CDERIVE(t3, x26, t0)
  CREVOKE(x26, t3)
  LCC(t6, t3, x0)
  ori t6, t6, 0x100
  sd t6, 40(x26)
ctx_ecall:
  ecall
  j ctx_ecall
#endif
ctx_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
data:
  .zero 4096
data_end:
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
tbl2:
  .zero 4096
tbl2_end:
.align 12
tbl3:
  .zero 16384
tbl3_end:
.align 12
frame:
  .zero 4096
frame_end:
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
  TEST_DATA
RVTEST_DATA_END
