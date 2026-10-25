# S-29's arm made three errors in one evening, and each was caught by a different thing

*RTL lane, 2026-10-18. The finding itself is `ISSUES.md` S-29 and the readings live in the arm's own header
(`capstone-ariane`, `verif/tests/custom/capstone/va-satpstale.S`, on `virtual-profile-2-c`). This note is only the
process trail, because the three errors are different shapes and only one of them is in the existing rules.*

## What the arm was for

The switcher writes `satp` on both of its walks and nothing on that path flushes a TLB. The question was whether
that is a correctness argument or a reading. It is a reading, in both directions, and the fix is on the lane branch.

## Error 1: the setup proof consumed the condition under test

The arm proves its own page tables before testing anything, because a clean reading from a mis-built table says
nothing. The caller therefore reads the probe address under its own chain and reports the value — and that read
installs the very translation the context was then supposed to be denied. So the context hit the caller's entry,
the pre-registered second reading missed, and the fifth reading became void: a context that never walked its own
chain never left behind the entry the escape direction is meant to serve.

This is the mirror of the rule we already have. CLAUDE.md says a directed test must CREATE the triggering
condition rather than merely contain the shape. Here the test created it and then destroyed it, in the step whose
whole purpose was to make the result trustworthy. **What caught it:** the pre-registered value. The reading was
"wrong" rather than "clean", and a wrong value is bisectable where a clean one is not.

**What would have caught it earlier:** asking, before the run, which of the arm's own instructions touch the state
under test. The setup proof is an access to the exact address whose translation is the subject.

## Error 2: the fix fired a flush that was filtered by registers belonging to a different instruction

The first implementation raised a pulse on the switcher's satp write and OR'd it into the MMU's `flush_tlb_i`.
That input is not a flush-all: the instruction and data TLBs qualify it by `asid_to_be_flushed_i` and
`vaddr_to_be_flushed_i`, two registers that latch the EX stage's rs1/rs2 forwarding **every cycle**, so they carry
meaning only for an actual `sfence.vma`. A pulse from anywhere else flushes at most one arbitrary entry. The shared
TLB is unqualified and did clear, which is what made the failure partial rather than total.

**What caught it:** that the prediction named all four arms. Three read exactly their pre-fix values and only the
fenceless one changed, which is impossible, because the fenceless arm and the post-escape-fence arm are
instruction-identical up to the entry; their traces are line-identical through the context's load, same virtual
address, different values. A pass/fail gate, or a prediction naming only the arm expected to move, would have read
a partial success.

**The general shape, which is not in the rules:** a control input that *looks* like a command ("flush") can be
qualified by state that belongs to the instruction that normally raises it. Before reusing such a signal, read what
else gates it downstream, and check what drives that gate when your new source fires rather than the usual one.

## Error 3: a sentence about what software cannot do, left standing next to the arm that does it

The arm's header claimed the escape direction was the serious half because "an escape is asynchronous, so the caller
resumes with no instruction of its own in which to have fenced". The control arm *is* the caller fencing after the
escape, at a single static point, and it works: the escape returns to CSRUNV+4 whatever its kind. The sentence was
written before the control existed and was not revisited once it did. It had already been sent to two lanes.

What survives is narrower and more useful: for the caller's data accesses the window is an ABI rule in both
directions, and the surface a fence cannot cover is the caller's first instruction FETCH after the escape, since
that fence is itself fetched through an instruction TLB that may hold the context's entry. That is S-23's shape and
this arm does not observe it.

**What caught it:** an auditor, asked to attack the claim's weakest link, with the weak link named in the prompt.
**What would have caught it without one:** re-reading the framing paragraph when the control arm first returned. A
control that succeeds is evidence about the claim, not only about the instrument.

## The one rule worth taking from this

Two of the three errors were caught by the same thing, and it was not a gate: the prediction named **every** arm's
expected value, including the arms expected not to move. That is what turned a plausible partial result into a
contradiction. The existing rule asks for the predicted reading to be written down before the build and used as a
go/no-go input. The sharpening is that it must cover the whole set, because the arms expected not to move are the
ones that detect a fix which works for the wrong reason.
