# Commit subjects lost in the 2026-09-04 consolidation — safety copy

**Why this file exists.** The `capstone-bootstrap` history was rewritten twice on 2026-09-04
(81 commits folded into 30). **77 commit subjects appear nowhere in the new history**,
of which **18 are retraction or correction records**. On this project a retraction is
evidence — the reasoning that ran one step past the data is exactly what the record is for — and
the consolidated messages narrate those retractions in prose but do not preserve them as
addressable commits.

**The objects are intact but exist in ONE place.** They are reachable only from the local tags
`pre-consolidate-backup` and `pre-consolidate2-backup` in this clone. `git ls-remote --tags
origin` returns **zero** matching tags: they have never been pushed, the push allowlist blocks
it, and that is being raised with the project lead.

**This file is a safety copy of the index, not a substitute for pushing the tags.** It preserves
what was lost and the hash to recover each item from *while the tags still exist*. If the tags
are pushed, or the objects are otherwise preserved, this file can be deleted.

**Recover an individual commit** (while the tags exist in this clone):

    git show <hash>
    git log --format=%B -1 <hash>      # full message, which is the part that was lost

## Retraction and correction records (18)

- `b8311b1003ac` RETRACT: the proposed S-12 fix breaks UNINIT, and the mechanism framing was wrong
- `7f0b6a390945` CORRECT the S-12 [33] localisation: bad pooling, a non-sequitur, and an unproven mechanism
- `f4dd37d8c2bb` S-12: a one-instruction arm to finish the attribution, and a correction to what --tight can show
- `f5e91f194196` RETRACT: the sentinel arm does not refute the S-12 stale-VALUE account
- `f46c13f85bb8` S-12: the stale-VALUE account is refuted on the board; the cure is in five instructions, unattributed
- `73a6e6d34697` S-12: retract "the signature appears with no reload" from the live repro folder
- `794aa2363ec1` S-12: retract the no-reload signature; fix four gates that had already passed bad cuts
- `ee13337aa215` Archive the superseded SQLite correctness proposal from its orphan branch
- `7085ee642468` Correct the root cause: the domain has NO trap vector at all, and the fix is not firmware-only
- `6c2cd1aebf81` Correct the tval finding: it is stronger than claimed in one place and weaker in three others
- `775cd402934e` Correct the S-12 tallies: three "wedges" never ran a domain, and the depth framing inverts
- `0a3fce44b582` RETRACT the register-match claim: the curing binary still contains the pairing, at the same address as last time
- `769c8ef20a51` RETRACT the register-pairing claim: the counterexample was in disassembly I had already printed
- `a2e21aed03a0` RETRACT the null-store mechanism and the layout refutation: slot is confounded with role
- `375d54043285` LAYOUT REFUTED: a nop at the identical point wedges 4/4 where a fence returns 0/7
- `ae50ac1c7fbc` RETRACT the S-12 "store-to-load drain" mechanism: the dose-response rung was a fabricated wedge
- `269b11a9ef2e` CORRECTED: S-12 needs PLAN DEPTH >= 2, not a join -- and dd3_subq was never a two-level test
- `4e03cab55e50` RETRACT "S-12 reproduced in simulation": the test stored the NOT_CAP it then loaded back

## All other lost subjects (59)

- `cb75315e1f0c` S-12: relocating the store cured it 4/4 but did not discriminate; the addresses narrow it instead
- `8a699877a7be` S-12: update the recorder go/no-go -- H3 strongly disfavoured, H1 weaker, recommend deferring
- `7e2d8697cddc` S-12: register the D3/D4/D5 arms and the shape rung's pressure and stack-slot knobs
- `a537fb2fb8c3` S-12: a written GO/NO-GO for the operand-mux recorder, before any synthesis is spent
- `6bf75f7b23d2` S-12: the narrowed-fence experiment cannot discriminate on this core -- decoder ignores pred/succ
- `e2f5ab9a7d62` S-12: a one-word fence removes the fault and SQLite COMPLETES on silicon
- `29540a880eb6` S-12: store-buffer pressure is not the missing trigger either
- `a029c357aa5c` S-12: the shape does not fault in a real capability domain either -- 12,288 executions, zero faults
- `291b247b196b` S-12: an in-domain rung for the faulting shape -- WIP, does not run yet
- `53c96a459182` S-12: the trap-enabled build reports the fault on all three draws
- `58f20385ddde` S-12 no longer kills the board: the fault returns a code naming its own cause and site
- `5397e0fb9ad5` S-12: all THREE operands must be the same register, and an RTL mechanism that predicts exactly that
- `2a75c44d2a2e` S-12 reduced to one byte: the store's source operand, with the four-image 2x2 and the repro
- `1f8f65a94603` S-12: two one-byte arms to break the three-way tie left by the [33] result
- `cde27f168ff0` S-12 localised to ONE instruction operand: [33] stc must read a4, the register [34] ldc writes
- `49746d49d22c` S-12: the cure narrows to TWO instructions -- [28] and [33]
- `d9fc242eca1b` S-12: separate "never ran" from "in progress" from "failed" in the arm classifier
- `a5f3792246aa` S-12: the arm classifier reported in-progress draws as failures
- `363a833b624b` S-12: the value account cannot be tested by patching this window -- value and stored null are the same register
- `a1502107b3a3` S-12: require the control slot to have passed before an arm draw counts, and the two arms so far
- `e055761c8836` S-12: a two-instruction control for the store-register question, because the five-instruction one cannot attribute
- `e18c3702a8b3` S-12: scope the arm tabulator to its own test stage -- it was reading the replayed previous boot
- `2f4724e061bf` S-12: one classifier for all three arms, and the sentinel arm's result
- `154e05b3e7c5` S-12: record the functional gate, the RTL's two accounts, and the matched pair
- `b3e35060afbf` S-12: add the sentinel arm's matched control -- the pair differs by exactly one word
- `55654fede587` S-12: guard the stale-operand verdict on mepc, and add the sentinel arm the gate cleared
- `62baabef44c3` Gate the minimisation tool: def-use closure, a pinned base, and the check order that matters
- `4f0d17c9f1bf` The S-12 signature appears with NO reload preceding it, on a different instruction
- `b71f77ab6dda` The truncation ladder cannot work: it crashes before reaching the reduced site
- `c10d9e39b35c` Acceptance test: the domain trap vector WORKS, and the failure moves one layer inward
- `da21d121a1d8` Client side of the workbench contract: epoch, gap detection and an integrity verdict
- `87e2c75e423d` Narrow S-12's fault to load-to-use forwarding, and exclude three of the four mcause-25 producers
- `c6d1d2d300c0` ROOT CAUSE: a domain runs with mtvec = 0, so any exception it takes jumps to address zero
- `3e5cffc587c9` S-12 is TWO defects, and the one that kills the board is not specific to S-12
- `f37a5abb6f96` The mtval instrument WORKS: tval = 0xBEEF, twice. The GDB halt was destroying the reading
- `8a02c8fd2ac2` Let a run proceed when the console does not KNOW the bitstream, loudly and only then
- `f9ce4c1fc36f` Start actually MINIMIZING S-12: exact source mapping, a truncation knob, and a gate for it
- `07cdd03cfe2f` The mepc precondition is PER-ARM, and as written it would have voided three real wedges
- `ea360b577ece` The whole S-12 arm series on one classifier: eight images, p = 0.018, and still not a finding
- `f4dbdac7a755` Close two holes in the boot-1 reading table before it fires
- `49b32111bfa5` State: the S-12 next step is instruments, not another arm — and the tallies it rested on moved
- `1d0afd1ee37f` SPEC VIOLATION: capability mcause from the data path is one too high, and 25 means two things
- `44e95abd2521` The firmware-freshness gate never checked the domain binary, and said it did
- `32f163dbffde` Name the prepare that wedges: a progress marker that survives a wedge and leaves the fault site alone
- `dfb64a523b2a` Preserve the S-12 register patcher as a script, with the anchor check it always needed
- `5dfd051730c9` A higher-power S-12 probe, because five 1-bit draws cannot settle a coin flip
- `1d3393a73e08` Add the S-12 movc/ldc rename pass: validated SAFE, DEFAULT OFF, efficacy not established
- `f2790b08c86f` State: S-12 has a causal trigger condition, and the next step is a COMPILER change
- `703276c39cc5` REGISTER MATCH IS CAUSAL: a three-byte patch cures S-12, p ~ 1e-5
- `4a4a4e53702f` The correlate is a REGISTER pairing, not the store: movc rD,zero immediately before ldc rD
- `590fb4d7cabd` RE-ESTABLISHED in slot 1: the null capability store's PRESENCE in the window is required
- `cea4d2dd2f58` The null capability store is implicated: moving it out of the window cures 0/5, a layout-matched control does not
- `42954a8a0c2f` MECHANISM: S-12 is a store-to-load drain hazard -- closing the window eliminates it, 0/4 against 4/4
- `89aaea18d1c3` A semantically neutral fence cuts the S-12 rate from 11/11 to 1/4: the drain window is a major contributor
- `4a8179aaadbc` S-12 cannot be reduced to an instruction sequence: the instructions are identical on every call
- `bd9acb5e854c` Clamping the 5th WhereCodeOneLoopStart call: suggestive, confounded, and recorded as weak
- `935ad17244de` SETTLED: S-12 is NESTING, not repetition -- two levels inside ONE prepare, measured as a matched pair
- `d01b8a213275` Measure what the ARGP counter never reported: calls = 3 + plan depth, exactly
- `2c00224da4fe` FIRST POSITIVE CHARACTERISATION OF S-12: it is a JOIN, and it happens at PREPARE time
