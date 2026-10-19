# See LICENSE for license details.
# Instruction names are lower case in comments because this file goes through cpp (a macro name followed by a parenthesis is a call).

#*****************************************************************************
# va-satpstale.S  (the switcher installs an address space with no TLB flush: the ESCAPE direction, 2026-10-18)
#-----------------------------------------------------------------------------
#
# ONE arm for two items, which is why it is worth its length.
#   * The satp item: the switcher writes satp on both of its walks and never flushes (csr_regfile has no flush_tlb
#     reference at all), so a translation installed under one address space can serve an access made under another.
#   * S-23: the caller's first instruction fetch after a return refused with a PHYSICAL mtval, three panics on the
#     board, mechanism open after two refuted candidates. A stale translation produces that exact shape, since the
#     fetch would translate through the context's entry to a physical page the caller's capability region does not
#     cover, and the capability check then refuses it with the physical address.
#
# WHAT THE DIRECTIONS MEAN. The first version of this paragraph claimed the escape direction was the defect because
# "an escape is asynchronous, so the caller resumes with no instruction of its own in which to have fenced", and that
# is WRONG. It was refuted by this arm's own control: va-satpstale-bf IS the caller fencing after the escape, at one
# static point, and it works. The escape returns to CSRUNV+4 whatever its kind, because the switcher's escape walk
# restores ids 0..66 from the save area with no kind-dependent branch and the saved id 0 is the request's pc_next, so
# a single sfence.vma placed there covers every escape. The sentence was written before the control existed and was
# not revisited once it did, which is the whole error: a claim about what software CANNOT do, sitting next to an arm
# that does it.
#
# What is true instead:
#   * For the caller's DATA accesses the hole is an ABI rule in BOTH directions, one fence at CSRUNV+4 and one before
#     the entry. That is what this arm measures, and an ABI rule is still worth closing in hardware, because every
#     caller must get it right and nothing enforces it.
#   * What a fence cannot cover is the caller's first instruction FETCH after the escape, since the fence at
#     CSRUNV+4 is itself fetched, under the caller's satp, through an instruction TLB that may hold the context's
#     entry. A context's fetches DO populate that TLB under its own satp: translation is forced on for fetch as well
#     as for load and store while a context runs, cva6.sv:1910 and :1912 OR vctx_eff into both enables. That is the
#     unfenceable surface and it is S-23's shape. The arms that read a LOAD cannot observe it, because the caller's
#     text page is mapped identically under both chains; -DFETCH_FACE is the build that can, and it is also the only
#     reading that covers the instruction-TLB half of the flush, which the other arms assert and do not measure.
#
# -DFETCH_FACE, PRE-REGISTERED before its first run. A page-aligned `tramp` page inside the text region, so the
#   caller's PCC covers it, holds `addi a3,x0,0x5a / addi a7,x0,3 / ecall`; a data page tb_page holds the same shape
#   reporting 0x5b, written physically before any satp exists. Chain A maps the tramp address to its OWN page and
#   S-only, since SUM covers loads and stores and not fetches, so an S fetch from a U page would be refused whatever
#   the TLB holds; chain B maps the SAME address to tb_page and with U, since a context's PTE permissions read as
#   USER. The context's entry PCC is the tramp address, so the context's own fetches install an instruction-TLB
#   entry for an address the caller has NEVER fetched from, which is what makes the reading unambiguous: there is no
#   correct entry for it to have displaced. After the escape the caller jumps there.
#   Readings in order: 0xaaaa, 3, 11, THE READING, 0x55.
#     0x5b = the caller fetched the CONTEXT's page. The instruction TLB carries the window too, S-23's candidate
#            stops being a shape match, and the flush's fetch half has something to fix.
#     0x5a = the caller fetched its own page. The window is DATA-only, and S-23 needs a different mechanism, which
#            is the direction the board lane's CPMP reading already points for their one capture.
#     a handler cause instead = also a reading, and the nearest yet to S-23's own signature, since a refused fetch
#            with a physical address is exactly what that issue reports. Trust it only if 0xaaaa printed first.
#   -DFENCE_CTRL pairs with it as for the load arms, the caller fencing before the jump, and must read 0x5a on
#     every tree. On the FIXED tree both builds must read 0x5a, and that is the flush's fetch half measured.
#   WHAT THE FIRST FETCH_FACE RUN READ, run satpfetch on bf3db3175, no exceptions, harness SUCCESS on both:
#     va-satpstale-bfec   0xaaaa, kind 3, cause 11, 0x5a, 0x55     29035 retired
#     va-satpstale-bfe    0xaaaa, kind 2, cause 12, 0x5a, 0x55     29032 retired
#   Cause 12 is an instruction page fault, taken by the CONTEXT, with kind 2, the terminal kind, so that build's
#   context died on its first fetch and the caller's own reading is VOID: nothing installed an entry for the
#   trampoline address, so there was nothing stale for the caller to hit. Same void-by-construction shape as the
#   load arms' first run, in a different place.
#   But the two builds have IDENTICAL page tables and differ only in the caller's instruction at CSRUNV+4. Note
#   that the gap build's instruction there is a plain nop and its context runs, so the sfence.vma's SEMANTICS are
#   not what rescues the context; bfeg is the nop control for bfec, which this header failed to say.
#   WITHDRAWN: "the only mechanism that can couple those is the caller's own frontend fetching the jump's target".
#   That was unearned. What is OBSERVED, from the .iss traces: the pre-entry trace content is bit-identical across
#   bfec, bfeg and bfegc and differs in bfe alone; in bfe the jump RESOLVES one cycle before the switch begins
#   (its writeback names the jump's slot pc with result pc+4, and the switcher's busy flag rises the next cycle);
#   and the frontend freezes its next-pc for the whole switch (frontend.sv:458) while continuing to request it
#   (:339), with privilege flipping to M only at the switcher's own register step and vctx forced off during the
#   walk. So the caller's frontend demonstrably requests the trampoline address at S under chain A across the
#   switch. That is consistent with a stale instruction-TLB entry AND with an in-flight page-table walk completing
#   after the switcher's satp write, and THIS RUN CANNOT SEPARATE THEM. The second alternative matters for the fix
#   rather than for the finding: a walk that completes after a single-cycle flush pulse would re-install an entry
#   the flush had just cleared. The four arms on the fixed tree do not show that happening, which bounds it but
#   does not exclude it at other timings. THE DECISIVE ARM IS BUILT AND RUN, and it settles it.
#   -DJUMP_ELSEWHERE points the caller's jump at a second page the context's entry PCC never names. Run satpj on
#   bf3db3175, no gap in either arm, both harness SUCCESS and 0 exceptions:
#     va-satpstale-bfeJ   0xaaaa, 3, 11, 0x5b, 0x5c, 0x55   29057 retired
#     va-satpstale-bfe    0xaaaa, 2, 12, 0,    0x5a, 0x55   29038 retired
#   The ONLY difference is WHICH page the caller jumps to. Pointed elsewhere, the context runs to its ecall,
#   kind 3 and cause 11, leaving its own marker 0x5b in the shared page; pointed at the context's own entry page,
#   the context dies on its first fetch with kind 2 and cause 12, having stored nothing. So the harm is
#   specifically THE CALLER'S FETCH OF THE CONTEXT'S ENTRY PAGE, which is branch (a) as pre-registered, and the
#   in-flight page-table walk is REFUTED for this timing: a walk already under way cannot care which page the
#   caller's jump names. That also removes the residual doubt recorded just above about the fix -- the mechanism
#   is a stale TLB entry, so a single-cycle flush pulse on the satp write is the right shape and there is no
#   post-flush refill to outrun.
#   Two things the run cannot separate, which the next build fixes rather than argues:
#     * WHICH page the context fetched in the passing build. Both pages end in ecall, so kind 3 does not say.
#       tb_page's code now stores its marker through x26, the context's shared-page capability, and the handler
#       prints that page: 0x5b means the context fetched tb_page, 0 means it fetched the trampoline's own page.
#       The trampoline's own code stays store-free, because the CALLER executes it too.
#     * Whether the caller's fetch stream is really the mechanism. -DFETCH_GAP puts 64 nops between CSRUNV and the
#       jump, so the target is outside any plausible fetch-ahead window.
#   PRE-REGISTERED for the next run, all four arms:
#     bfec    kind 3, cause 11, shared 0x5b, reading 0x5a
#     bfe     kind 2, cause 12, shared 0,    reading 0x5a
#     bfeg    kind 3, cause 11, shared 0x5b, reading 0x5b   <- the fetch face proper: with the context's own entry
#               installed, the caller's unfenced fetch after the escape resolves to the CONTEXT's page
#     bfegc   kind 3, cause 11, shared 0x5b, reading 0x5a   <- its control, the caller fencing first
#   A bfeg reading of 0x5a with shared 0x5b is a FINDING and not a failure: the instruction TLB would then not
#   carry the window although the data TLB does, which pushes S-23 towards a different mechanism.
#
#   CORRECTIONS to this header, two of them to text that was already pushed; an auditor found all three.
#   (i) The tohost claim below is FALSE and satpfetch2 was written off for nothing. Read from each run's own
#       command line: satpfetch2 had tohost at 0x80005000 with the trampoline at 0x80004000, and satpfetch3 has
#       tohost at 0x80006000 with the trampoline at 0x80005000. They never collided in either run. The filler page
#       moved TOHOST, not the trampoline off tohost. satpfetch2 is sound, and it is in fact the STRONGEST evidence
#       for the escape-side reading: its looping handler walked forward and the instruction words tracked tb_page's
#       contents at addresses no instruction ever retired, including zero past the sixteen bytes the setup wrote,
#       which kills the one real confound, that the trace's instruction field might be a stale latch of the
#       context's own last retired word at that pc.
#   (ii) "the only mechanism that can couple those" is withdrawn, see below.
#   (iii) the bound on S-23 is narrower than written, see below.
#
#   WHAT THE FOUR READ, run satpfetch3 on bf3db3175, with the handler made to END on an unexpected cause instead
#   of returning into the same fault:
#     va-satpstale-bfec    0xaaaa, 3, 11, 0x5b, 0x5a, 0x55                  29042 retired, SUCCESS, 0 exceptions
#     va-satpstale-bfegc   0xaaaa, 3, 11, 0x5b, 0x5a, 0x55                  29106 retired, SUCCESS, 0 exceptions
#     va-satpstale-bfe     0xaaaa, 2, 12, 0,    0x5a, 0x55                  29038 retired, SUCCESS, 0 exceptions
#     va-satpstale-bfeg    0xaaaa, cause 12, mepc 0x80005000, mtval 0x80005000, 0x55   29094, SUCCESS, 1 exception
#   THREE of the four are the pre-registered values exactly. The two controls read 0x5a with the context's own
#   marker 0x5b in the shared page, which proves at the same time that chain B's leaf works and that the context
#   fetched tb_page rather than the trampoline's own page. bfe is the pre-registered 2/12/0: the context died on
#   its first fetch and the caller's reading is void, with the shared 0 confirming it stored nothing.
#   bfeg MISSED its predicted 0x5b, and the reason makes the finding stronger rather than weaker. The caller's
#   fetch DID resolve through the context's translation -- the single exception is at the trampoline's virtual
#   address and the instruction word the trace shows there is 0x05b00693, tb_page's first instruction, which
#   exists nowhere else -- and it was then refused with cause 12, because that leaf carries U and the caller runs
#   at S. The prediction of 0x5b was impossible by construction: a leaf cannot be both U, as the context needs,
#   and S-executable, as the caller would need in order to run the code it fetched. So this arm's fetch-face
#   reading is a refused fetch and can never be an executed one.
#   THE BOUND, corrected. What holds: THIS ARM cannot reproduce S-23, because its refusal is cause 12 with a
#   VIRTUAL mtval, which is the PTE permission check speaking, while S-23 reports an instruction ACCESS fault with
#   a PHYSICAL address, which is the capability check speaking. Two supports for that, both read from source: the
#   capability check on a fetch is strictly subordinate to the page fault, since it is gated on
#   !fetch_exception.valid (pmp_data_if.sv:239-248); and it is the ONLY instruction-side fault carrying a physical
#   tval, the other two reporting the virtual address (pmp_data_if.sv:224-227, cva6_mmu.sv:477-480).
#   WITHDRAWN, as an overclaim: that a stale translation could not reach S-23's signature in this design at all,
#   and the reason given for it. The reason was wrong twice over. The fetch permission check never consults the
#   PTE's execute bit on a TLB hit; it tests the U bit against the effective privilege alone
#   (cva6_mmu.sv:376-379), so "S-executable" names nothing the hardware looks at. And the exclusivity it rested on
#   only holds for a caller at S: the instruction-side capability check fires for every privilege below M
#   (pmp_data_if.sv:241), so a caller at U hitting a context-installed U page PASSES the permission check and then
#   meets the capability check, which refuses it with a PHYSICAL address. That is S-23's exact signature, from a
#   stale translation alone, with no capability table in flux. This arm cannot produce it only because its caller
#   runs at S and its single capability region covers every page it can reach, and both of those are choices.
#   A -DCALLER_U build tested it, and S-23's SIGNATURE IS REPRODUCED. Run satpu2 on bf3db3175, both arms harness
#   SUCCESS, one exception each:
#     va-satpstale-bfeU    0xaaaa, cause 1,  mepc 0x80005000, mtval 0x80014000, 0x55    29099 retired
#     va-satpstale-bfeUc   0xaaaa, cause 12, mepc 0x80005000, mtval 0x80005000, 0x55    29100 retired
#   Both as pre-registered. The caller stays at S for the whole arm, because CSRUNV and sfence.vma both need it,
#   and drops to U for its FIRST FETCH ONLY, by clearing sstatus.SPP and returning with sret. The pair differs by
#   the one post-escape sfence.vma and differs in BOTH the cause and the KIND of address reported:
#     without the fence the fetch resolves through the CONTEXT's entry, whose page carries U, so the U fetch
#       PASSES the permission check and then meets the instruction-side capability check, which refuses it with
#       an INSTRUCTION ACCESS FAULT reporting the PHYSICAL address 0x80014000. That page is tb_page, and the
#       caller's single capability region is [_start, ptB_end) = [0x80000000, 0x80014000), so the refused page
#       begins exactly one byte past the region's end. The trace's instruction word there is tb_page's own first
#       instruction. That is S-23's signature: an instruction access fault with a physical mtval, from a stale
#       translation alone, with no capability table in flux.
#     with the fence the caller walks chain A, whose trampoline leaf is S-ONLY, and a U fetch of a U=0 page is
#       refused by the permission check with cause 12 and the VIRTUAL address, which is why the leaf was left
#       S-only rather than made U: it makes the two outcomes differ in the cause as well as the value.
#   WHAT THIS DOES NOT SHOW. It does not show that the board's S-23 panics are this. Their captured face is a
#   DATA access by the kernel at S, and the board lane has shown its refused address IS covered by a loaded
#   region, which this mechanism does not explain; that face points at the revocation-node validity term
#   instead, which is R-44's class. So S-23 may have two faces with two mechanisms, and what is established
#   here is that the FETCH face is reachable from a stale translation and needs no table in flux.
#   The first cut of this build put the WHOLE caller at U and died at once, which is kept because it is a fact
#   about the ISA rather than about this arm: sfence.vma is illegal at U, so the pre-entry fence trapped with
#   cause 2 before anything ran, and CSRUNV from U would have been refused next.
#   * Caveat kept deliberately: "returns to CSRUNV+4 whatever the kind" is static by design plus observed for the
#     ecall kind only. No quantum-expiry or page-fault escape arm exists yet.
#
# The construction. Two page-table chains are built EXPLICITLY, by one macro run twice over its own three pages at
# fixed offsets, never by copying one chain into the other: the first cut of this idea died three times on a copy
# loop and its offset arithmetic rather than on the hypothesis. Chain A is the caller's, chain B the context's, and
# they are identical but for the leaf of one page, `probe`, which maps to `pa` under A and to `pb` under B. The
# caller's own text page is S-only in both, as it must be for an S fetch; everything else carries U, and the caller
# runs with SUM so that a stale hit stays readable as DATA instead of turning into a permission fault.
#
# The arm proves its own setup before it tests anything, because a clean reading from a mis-built table says nothing.
#
# Pre-registered readings (CAPPRINT values in the retirement trace, in order):
#   0xaaaa  the caller's read of `probe` under chain A, before the entry: chain A proven
#   0xbbbb  the context's read of `probe` under chain B, reported through a page both chains map alike: chain B
#           proven, and the entry a stale window would serve
#   3, 11   the escape's kind and cause, so the escape is known to have happened
#   THE READING, the caller's read of `probe` as its first access after the escape returns:
#     0xaaaa = the satp write took effect for this access, no stale window; the satp flush stays a correctness
#              argument and S-23 keeps its mystery
#     0xbbbb = the caller read through the CONTEXT's translation; the hole is a reading and the flush fixes a
#              demonstrated defect. It is also S-23's strongest CANDIDATE and not its mechanism: S-23 is a FETCH
#              refused with a physical address, this arm demonstrates a stale DATA translation, and the caller's
#              text page is mapped identically under both chains here, so the instruction TLB is not observed at
#              all. Reproducing S-23 directly needs the caller's code VA pointed at a different physical page
#              under chain B, which is a separate arm.
#     a cause printed by the handler instead = ALSO positive, PROVIDED the two setup proofs printed first, since a
#              missing capability region makes every S fetch read cause 1, as the first run of this arm did: the stale entry was hit and then refused by a
#              permission or capability check rather than read, which is nearer still to S-23's own signature
#   0x55 and the pass marker.
#
# WHAT THE FIRST VALID RUN ACTUALLY READ, 2026-10-18 on bf3db3175 (run satpst3, both arms, no exceptions, harness
# SUCCESS): 0xaaaa, 0xaaaa, 3, 11, 0xaaaa, 0x55. The SECOND reading is the pre-registered 0xbbbb MISSED, and the
# reason is in this arm's own setup: the caller's read that proves chain A also installs a translation of the probe
# VA to `pa`, and nothing flushes between that read and the entry, so the context hit the caller's entry instead of
# walking chain B. That failure is informative but it is not the escape direction, and it makes the fifth reading
# VOID whatever its value, because a context that never walked chain B never installed the entry the escape
# direction is supposed to leave behind. Two hypotheses survive it and the arm as first built cannot separate them:
#   1. the context translated through the CALLER's surviving entry, which is the stale window on the entry side;
#   2. the frame's satp word never took effect at all, so the context ran under chain A throughout.
# -DFENCE_BEFORE separates exactly those two by one instruction: the caller executes sfence.vma immediately before
#   the entry, which drops the entry its own setup proof installed. Then a second reading of 0xbbbb proves chain B
#   AND proves that sfence.vma flushes, both from the single change, and the fifth reading becomes the escape
#   direction's real test; a second reading still 0xaaaa is hypothesis 2 and sends the question to the switcher's
#   entry walk rather than to the escape.
# -DFENCE_CTRL is the control: the caller executes sfence.vma immediately after the escape returns, before the read.
#   It must read 0xaaaa whatever the default build does, which is what makes a 0xbbbb above a staleness reading
#   rather than a mapping mistake. Build it WITH -DFENCE_BEFORE: on its own it is paired against a default arm whose
#   fifth reading is void, and the two were indeed indistinguishable in run satpst3, differing by one retired
#   instruction and nothing else.
# A trap anywhere else ends the arm with the handler's mcause as the last reading.
#
# WHAT THE PAIR READ, 2026-10-18 on bf3db3175 (run satpst4, no exceptions, harness SUCCESS on both):
#   va-satpstale-bf   0xaaaa, 0xbbbb, 3, 11, 0xaaaa, 0x55     28985 retired
#   va-satpstale-b    0xaaaa, 0xbbbb, 3, 11, 0xbbbb, 0x55     28984 retired
# The two arms differ by ONE retired instruction, the caller's sfence.vma after the escape returns, and the fifth
# reading differs with it. So:
#   * The SECOND reading flipping from 0xaaaa in satpst3 to 0xbbbb here, on the single pre-entry fence, settles
#     three things at once: chain B is built correctly, sfence.vma does flush, and the frame's satp word DOES take
#     effect, which refutes hypothesis 2 above. It also makes satpst3's own second reading a positive result rather
#     than a void one: with chain B proven and the fence proven, a context that read 0xaaaa read the CALLER's
#     surviving translation of that VA. The entry direction has the window too. That reading does not even need
#     satpst4 to stand: satpst3's OWN trace shows the live satp at the escape walk's reg-66 transaction as chain B's
#     root, and the print carries the live value under either reading of it, since a read shows the current value
#     and a write shows the pre-write one. So the context ran under chain B and was served chain A regardless.
#   * Two further arms, both PRE-REGISTERED here and both run on the BASE tree, since each is about something other
#     than the flush and must not depend on it:
#     -DFENCE_BEFORE -DNO_CTX_READ closes the last gap in the fifth reading. The fence clears the entry the setup
#       proof installed, and the context then never touches the probe VA, so nothing can install one under either
#       chain. Predicted 0xaaaa, 0xcccc, 3, 11, 0xaaaa, 0x55. The fifth reading is the point: against -b's 0xbbbb
#       on the same tree these two differ in exactly one thing, whether the context accessed that address, which
#       is what turns "the caller read a stale entry" from the pair's reading into a demonstration of its origin.
#     -DFENCE_BEFORE -DPLAIN_LD settles, inside a real context, what a plain load with an UNTAGGED base does, which
#       the registry carries as an ordering question only. lla gives an integer, the same form the caller uses at S
#       where such a load succeeds, because the capability check on a plain access is gated on priv M and a context
#       runs at M. Predicted REFUSED: 0xaaaa, 0, a fault kind, 24, 0xaaaa, 0x55, the second reading being 0 because
#       the context is refused before it stores anything. The other branch is a finding rather than a failure: if
#       the load RETURNS, the readings are 0xaaaa, 0xbbbb, 3, 11, 0xbbbb, 0x55, and either lla carries a tag in
#       capmode or the check does not reach a context. That build never reads the probe VA either, so a returning
#       load would make it a second copy of the control above.
#     WHAT THEY READ, run satpctl on bf3db3175, both exactly as pre-registered, no exceptions, harness SUCCESS:
#       va-satpstale-bn   0xaaaa, 0xcccc, 3, 11, 0xaaaa, 0x55   28985 retired
#       va-satpstale-bp   0xaaaa, 0, 2, 24, 0xaaaa, 0x55        28983 retired
#     The first settles the ORIGIN of the stale entry. Against va-satpstale-b on the same tree, which reads 0xbbbb
#       fifth, these two differ in exactly one thing, whether the context accessed that address, and the fifth
#       reading differs with it. So the caller read an entry the CONTEXT's own access installed; that is no longer
#       an inference from the pair.
#     The second settles the plain-load question inside a real context: the load is REFUSED with cause 24, so lla
#       yields an untagged integer and the capability check does reach a context, which runs at M. The escape kind
#       is 2, the terminal kind, so such a refusal ends the context rather than pausing it. The second reading of 0
#       is the independent confirmation that the refusal happened before the context stored anything.
#   * The FIFTH reading, 0xbbbb with no fence and 0xaaaa with one, is the escape direction: the switcher restores
#     the caller's satp and does not flush, so the caller's first access after the escape translates through the
#     CONTEXT's address space. The satp flush is now a fix for a demonstrated defect.
# What this does NOT show, and must not be written as though it did: it is a stale DATA translation, S-23 is a
# FETCH refused with a physical address, and the caller's text page is mapped identically under both chains here,
# so the instruction TLB is never observed. S-23's candidate is strengthened, its mechanism is not shown.
# Scope of the isolation claim: the physical pages both chains can reach lie inside the one capability region this
# arm installs, so what is demonstrated is that translation crosses the boundary, not that the capability regions
# failed to contain it. A tighter region is the backstop and this arm does not test it.
#
# PRE-REGISTERED READINGS FOR THE FIX, written before the fix was built (the switcher's satp write raises a
# TLB-flush pulse, csr_regfile reg id 66): all FOUR arms read 0xaaaa, 0xbbbb, 3, 11, 0xaaaa, 0x55, that is
#   va-satpstale     second reading FLIPS 0xaaaa -> 0xbbbb   (the entry direction closed)
#   va-satpstale-f   second reading FLIPS 0xaaaa -> 0xbbbb   (same; its own fence is after the escape)
#   va-satpstale-b   fifth  reading FLIPS 0xbbbb -> 0xaaaa   (the escape direction closed)
#   va-satpstale-bf  UNCHANGED, and that is the NEGATIVE CONTROL: an arm that already fenced on both sides must
#                    read exactly what it read before, so a change there would mean the flush altered something
#                    other than the stale window.
# Anything else, including a new exception or a retired count that moves by more than the arms' own fences, is a
# failure of the fix and not a reading about staleness.
#
# WHAT THE FIX READ. The FIRST implementation failed here and this arm is why, so the failure is kept:
#   run satpfix1, three of the four arms read exactly their PRE-fix values and only the fenceless one changed.
#   That is impossible as a reading about staleness, because the fenceless arm and the post-escape-fence arm are
#   instruction-identical up to the entry; their traces are line-identical through the context's load, same virtual
#   address, different values. The cause was not in the switcher at all: the flush had been OR'd into the MMU's
#   flush_tlb_i, which the instruction and data TLBs qualify by asid_to_be_flushed_i and vaddr_to_be_flushed_i,
#   and those two latch the EX stage's rs1/rs2 forwarding on EVERY cycle, so they are meaningful only for an actual
#   sfence.vma. The pulse therefore cleared one arbitrary entry. The shared TLB, which is unqualified, did clear,
#   which is what made the failure partial rather than total, and a partial failure is what a pass/fail gate would
#   have called a success. The pre-registered set above is what made it a contradiction instead.
#   run satpfix2, with an unqualified flush-all carried to the instruction and data TLBs, all four arms read
#     0xaaaa, 0xbbbb, 3, 11, 0xaaaa, 0x55, no exceptions, harness SUCCESS, retired 28983 / 28984 / 28984 / 28985.
#   The negative control holds exactly: va-satpstale-bf is unchanged in every reading, in its retired count and
#   even in its cycle count, 222875 before and after, which is what an arm that already fenced on both sides must
#   do when the flush it no longer needs is added.

#include "riscv_test.h"
#include "test_macros.h"
#include "asm_insn.h"

#define CSRUNV(rd, rs1, rs2)  .insn r 0x5B, 0x1, 0x24, rd, rs1, rs2
#define CSUPQUANTUM 0x7C3
#define PTE_LEAF_RWXU 0xDF
#define PTE_LEAF_SRWX 0xCF
# The caller's own pages stay S-only even under CALLER_U, because the caller IS at S there; only its final
# fetch is taken at U. That is what makes a stale hit (the context's U page, cause 1 with a physical address)
# and a clean walk (chain A's S-only page, cause 12 with a virtual one) differ in the cause as well as the value.
#define PTE_CALLER_X PTE_LEAF_SRWX
#define PTE_PTR       0x01

#define MINT(REG, TYPE, LO, HI, PERM) \
  CAPCREATE(REG); li a0, TYPE; lla a3, LO; lla a4, HI; li a5, PERM; \
  CAPTYPE(REG, a0); CAPBOUND(REG, a3, a4); CAPPERM(REG, a5)

# BUILD_CHAIN(CAP, PROBE_PA): CAP is a capability whose base is this chain's root, with l1 at +4096 and l0 at +8192.
# 512 four-kilobyte leaves over [0x80000000, 0x80200000) with U, megapages above them, root[2] and l1[0] as pointers,
# the caller's text page S-only, and `probe` pointed at PROBE_PA. Clobbers t0-t5.
#define BUILD_CHAIN(CAP, PROBE_PA) \
  li t1, 0x80000000; li t2, 0; \
901: \
  srli t3, t1, 12; slli t3, t3, 10; ori t3, t3, PTE_LEAF_RWXU; \
  slli t4, t2, 3; li t5, 8192; add t4, t4, t5; CINCOFFSET(t0, CAP, t4); sd t3, 0(t0); \
  li t4, 0x1000; add t1, t1, t4; addi t2, t2, 1; li t4, 512; bne t2, t4, 901b; \
  li t2, 1; \
902: \
  slli t3, t2, 21; li t4, 0x80000000; add t3, t3, t4; srli t3, t3, 12; slli t3, t3, 10; \
  ori t3, t3, PTE_LEAF_RWXU; \
  slli t4, t2, 3; li t5, 4096; add t4, t4, t5; CINCOFFSET(t0, CAP, t4); sd t3, 0(t0); \
  addi t2, t2, 1; li t4, 512; bne t2, t4, 902b; \
  li t4, 8192; CINCOFFSET(t0, CAP, t4); LCC(t1, t0, x2); srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_PTR; \
  li t4, 4096; CINCOFFSET(t0, CAP, t4); sd t1, 0(t0); \
  li t4, 4096; CINCOFFSET(t0, CAP, t4); LCC(t1, t0, x2); srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_PTR; \
  li t4, 16; CINCOFFSET(t0, CAP, t4); sd t1, 0(t0); \
  lla t0, s_code; li t1, 0x80000000; sub t0, t0, t1; srli t0, t0, 12; slli t0, t0, 3; li t5, 8192; add t0, t0, t5; \
  CINCOFFSET(t2, CAP, t0); lla t1, s_code; srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_CALLER_X; sd t1, 0(t2); \
  lla t0, probe; li t1, 0x80000000; sub t0, t0, t1; srli t0, t0, 12; slli t0, t0, 3; li t5, 8192; add t0, t0, t5; \
  CINCOFFSET(t2, CAP, t0); lla t1, PROBE_PA; srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_LEAF_RWXU; sd t1, 0(t2)

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

# ---------------------------------------------------------------- the two physical pages the probe VA can reach
MINT(s2, CAP_TYPE_NONLIN, pa, pb_end, CAP_PERM_RW)
li t0, 0xaaaa
sd t0, 0(s2)                   # pa, written physically before any satp exists
li t0, 4096
CINCOFFSET(t1, s2, t0)
li t0, 0xbbbb
sd t0, 0(t1)                   # pb
MINT(s3, CAP_TYPE_NONLIN, shared, shared_end, CAP_PERM_RW)
sd x0, 0(s3)

# ---------------------------------------------------------------- the two chains, built explicitly, A then B
MINT(s6, CAP_TYPE_NONLIN, ptA, ptA_end, CAP_PERM_RW)
BUILD_CHAIN(s6, pa)
MINT(s7, CAP_TYPE_NONLIN, ptB, ptB_end, CAP_PERM_RW)
BUILD_CHAIN(s7, pb)

#ifdef FETCH_FACE
# chain A: the trampoline address maps to its own page and S-ONLY. The identity loop left it U, and an S-mode
# FETCH from a U page is refused whatever the TLB holds, because SUM covers loads and stores and not fetches.
lla t0, tramp; li t1, 0x80000000; sub t0, t0, t1; srli t0, t0, 12; slli t0, t0, 3; li t5, 8192; add t0, t0, t5
CINCOFFSET(t2, s6, t0)
lla t1, tramp; srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_CALLER_X
sd t1, 0(t2)
#ifdef JUMP_ELSEWHERE
lla t0, tramp2; li t1, 0x80000000; sub t0, t0, t1; srli t0, t0, 12; slli t0, t0, 3; li t5, 8192; add t0, t0, t5
CINCOFFSET(t2, s6, t0)
lla t1, tramp2; srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_CALLER_X
sd t1, 0(t2)
#endif
# chain B: the SAME address maps to tb_page, and with U, because a context's PTE permissions read as USER
lla t0, tramp; li t1, 0x80000000; sub t0, t0, t1; srli t0, t0, 12; slli t0, t0, 3; li t5, 8192; add t0, t0, t5
CINCOFFSET(t2, s7, t0)
lla t1, tb_page; srli t1, t1, 12; slli t1, t1, 10; ori t1, t1, PTE_LEAF_RWXU
sd t1, 0(t2)
# tb_page's three instructions, written physically before any satp exists: addi a3,x0,0x5b / addi a7,x0,3 / ecall
MINT(s1, CAP_TYPE_NONLIN, tb_page, tb_page_end, CAP_PERM_RW)
li t0, 0x00dd3023              # sd a3, 0(x26): the context reports WHICH page it fetched
slli t0, t0, 32
li t1, 0x05b00693              # addi a3, x0, 0x5b
or t0, t0, t1
sd t0, 0(s1)
li t0, 0x00000073              # ecall
slli t0, t0, 32
li t1, 0x00300893              # addi a7, x0, 3
or t0, t0, t1
sd t0, 8(s1)
#endif

# ---------------------------------------------------------------- the lifetime table and the frame
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

MINT(s9, CAP_TYPE_NONLIN, frame, frame_end, CAP_PERM_RW)
lla t0, ptB                    # the CONTEXT runs under chain B
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
sd t0, 0(s9)                   # the frame's satp word
lla t0, tbl
sd t0, 8(s9)                   # srevroot
li t0, 1
sd t0, 56(s9)                  # options: a resumable ecall
#ifdef FETCH_FACE
MINT(s4, CAP_TYPE_LIN, tramp, tramp_end, CAP_PERM_RWX)
#else
MINT(s4, CAP_TYPE_LIN, ctx, ctx_end, CAP_PERM_RWX)
#endif
STC(s9, s4, 64)                # the PCC slot
li t0, 480
CINCOFFSET(t1, s9, t0)
STC(t1, s3, 0)                 # x26: the shared page, how the context reports
li t0, 496
CINCOFFSET(t1, s9, t0)
MINT(s5, CAP_TYPE_NONLIN, probe, probe_end, CAP_PERM_RW)
STC(t1, s5, 0)                 # x27: a capability over the probe VA, for the context to read through
li t0, 0x100000
csrw CSUPQUANTUM, t0

# ---------------------------------------------------------------- the machine-level windows the caller needs
# Without these every S fetch is refused with an instruction access fault, which the first run of this arm
# demonstrated at length: capmode is on after CAPENTER, so a fetch at S has its PHYSICAL address checked
# against the capability regions, and with no region installed nothing is fetchable. Worth keeping for S-23:
# that omission reproduced S-23's exact cause, an instruction access fault in S, which is independent evidence
# that a physical fault address there is the capability check speaking and not the walker.
li t0, -1
csrw pmpaddr0, t0
li t0, 0x1f
csrw pmpcfg0, t0
MINT(a1, CAP_TYPE_NONLIN, _start, ptB_end, CAP_PERM_RWX)
CCSRRW(x0, CCSR_CPMP(0), a1)

# ---------------------------------------------------------------- into S under chain A, with SUM set
lla t0, ptA
srli t0, t0, 12
li t1, 8
slli t1, t1, 60
or t0, t0, t1
csrw satp, t0
sfence.vma
li t0, 0x1800
csrc mstatus, t0
li t0, 0x0800
csrs mstatus, t0               # MPP = S
li t0, 0x40000
csrs mstatus, t0               # SUM: an S access to a U page is legal
lla t0, s_code
csrw mepc, t0
mret

.align 12
s_code:
lla s10, probe
ld s11, 0(s10)
mv a3, s11
li a7, 1
ecall                          # M prints it: 0xaaaa, chain A proven

lla a0, frame
li a1, 0
#ifdef FENCE_BEFORE
sfence.vma                     # drops the entry the setup proof above installed, so the context must WALK chain B
#endif
CSRUNV(a2, a0, a1)             # the entry; the context's ecall escapes it
#ifdef FENCE_CTRL
sfence.vma                     # the control: the caller fences for itself before its read
#endif
#ifdef CALLER_U
# SPP = 0 and sret: the caller drops to U for its FIRST FETCH ONLY, which is the access under test. Everything
# before this ran at S, because CSRUNV and sfence.vma both need it. The stale entry, if there is one, was
# installed by the context before the escape and is already in place.
#ifdef FETCH_GAP
.rept 64
nop
.endr
#endif
li t0, 0x100
csrc sstatus, t0               # SPP = 0, so sret returns to U
lla t0, tramp
csrw sepc, t0
sret                           # THE READING is which page this U fetch resolves to, and how it is refused
#endif
#ifdef FETCH_FACE
#ifdef FETCH_GAP
# 64 nops, so the jump target is not reachable by the frontend's fetch-ahead at the CSRUNV. Without this the
# caller's own speculative fetch installs the trampoline address under chain A, S-only, and the context faults 12.
.rept 64
nop
.endr
#endif
#ifdef JUMP_ELSEWHERE
j tramp2                       # a page the context's PCC never names: the discriminator is the ESCAPE's kind
#else
j tramp                        # THE READING is which page this fetch resolves to
#endif
#endif
ld s11, 0(s10)                 # THE READING, the caller's first access after the escape
mv a3, s11
li a7, 2
ecall                          # M prints the escape's words and this reading
1:
j 1b

.align 12
trap_handler:
  csrr t0, mcause
  li t1, 9
  beq t0, t1, 5f
  # An unexpected cause: a positive reading, see the header. It ENDS the arm rather than returning, because the
  # causes that reach here include a refused FETCH, and advancing mepc past one of those refaults for ever:
  # bfeg burned 4,000,013 cycles and 156,995 exceptions on exactly that before this was written.
  CAPPRINT(t0)                 # the cause
  csrr t1, mepc
  CAPPRINT(t1)                 # where it faulted
  csrr t1, mtval
  CAPPRINT(t1)                 # and the address the fault reports, physical or virtual
  li t2, 0x55
  CAPPRINT(t2)
  CAP_PASS(s11)
5:
  li t1, 1
  bne a7, t1, 6f
  CAPPRINT(a3)                 # 0xaaaa
  j 9f
6:
  li t1, 2
  bne a7, t1, 7f
  ld t2, 0(s3)
  CAPPRINT(t2)                 # 0xbbbb, the context's read through chain B
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 3, the escape kind
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 11, the cause
  CAPPRINT(a3)                 # THE READING
  li t2, 0x55
  CAPPRINT(t2)
  CAP_PASS(s11)
7:
#ifdef FETCH_FACE
  li t1, 3
  bne a7, t1, 8f
  ld t2, 16(s9)
  CAPPRINT(t2)                 # 3, the escape kind, so the escape is known to have happened
  ld t2, 24(s9)
  CAPPRINT(t2)                 # 11, the cause
  ld t2, 0(s3)
  CAPPRINT(t2)                 # which page the CONTEXT fetched: 0x5b tb_page, 0 the trampoline's own
  CAPPRINT(a3)                 # THE READING: 0x5a the caller's own page, 0x5b the CONTEXT's
  li t2, 0x55
  CAPPRINT(t2)
  CAP_PASS(s11)
8:
#endif
  CAPPRINT(a7)
9:
  csrr t0, mepc
  addi t0, t0, 4
  csrw mepc, t0
  mret

.align 12
ctx:
#ifdef PLAIN_LD
  # The tag question, settled inside a real context rather than by reading the condition: lla yields an INTEGER,
  # the same form the caller uses at S for its own probe read, which succeeds there because the capability check on
  # a plain access is gated on priv M. A context runs at M, so this load must be refused with cause 24. If instead
  # it RETURNS, then either lla carries a tag in capmode or the check does not apply here, and that is the finding.
  # Placed first, so this build never reads the probe VA and so also serves as the no-context-read control.
  lla t3, probe
  ld t4, 0(t3)
  sd t4, 0(x26)                # only reached if the load was NOT refused: reports the value it read
  li a0, 7
  ecall
  j ctx
#endif
#ifdef NO_CTX_READ
  # The control S-29's fifth reading still owed: the context never touches the probe VA, so no entry for it can be
  # installed under this chain, and the caller's own read after the escape must then see ITS chain's page whatever
  # the flush does. It reports a constant so the second reading still proves the context ran.
  li t1, 0xcccc
  sd t1, 0(x26)
  li a0, 7
  ecall
  j ctx
#endif
  ld t1, 0(x27)                # the context reads the probe VA under chain B
  sd t1, 0(x26)                # and reports through the shared page
  li a0, 7
  ecall                        # the escape
  j ctx
ctx_end:

# A page of filler, so the trampoline does not land on 0x80004000, which is also the harness's tohost address.
# The arm never writes there, but an address that the harness itself watches is one an alternative explanation
# can be hung on, and run satpfetch2's reading should not need that ruled out.
.align 12
tohost_filler:
  .rept 16
  nop
  .endr

# FETCH_FACE's trampoline. Page-aligned and inside the text region, so the caller's PCC covers it. Chain A maps
# this virtual address to THIS physical page, S-only; chain B maps it to tb_page, which carries the same shape
# reporting 0x5b. Whichever page the caller's fetch resolves to after the escape is the reading.
.align 12
tramp:
  li a3, 0x5a
  li a7, 3
  ecall
  j tramp
tramp_end:

# A second trampoline, for JUMP_ELSEWHERE: a page the CONTEXT's entry PCC never names, so the caller's frontend
# requests this one across the switch instead of the context's. Its marker is 0x5c so the two cannot be confused.
.align 12
tramp2:
  li a3, 0x5c
  li a7, 3
  ecall
  j tramp2
tramp2_end:

.align 4
_end_of_text:

RVTEST_CODE_END

  .data
RVTEST_DATA_BEGIN
.align 12
tbl:
  .zero 4096
tbl_end:
.align 12
frame:
  .zero 4096
frame_end:
.align 12
shared:
  .zero 4096
shared_end:
.align 12
pa:
  .zero 4096
.align 12
pb:
  .zero 4096
pb_end:
.align 12
probe:
  .zero 4096
probe_end:
.align 12
ptA:
  .zero 12288
ptA_end:
.align 12
ptB:
  .zero 12288
ptB_end:
# Past the caller's one capability region, which stops at ptB_end. A context's fetch is not capability-checked,
# because a context runs at M architecturally, so the context can still fetch here; a CALLER below M cannot.
.align 12
tb_page:
  .zero 4096
tb_page_end:
  TEST_DATA
RVTEST_DATA_END
