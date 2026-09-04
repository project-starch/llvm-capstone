# A sim that produced the wanted answer, and was void: the control is what caught it

**Date:** 2026-08-26
**Status:** result discarded before it was acted on. Second attempt in flight.

## The result that looked right

`s12-postfault-capture.S` asked whether a load response arriving after a faulting
instruction still reaches the S-07/S-12 LDC recorder — the last question between that
recorder and a bitstream, and a race, so unanswerable by reading RTL.

It returned one capture:

    t=1123   valid=1   paddr=0x80003000   tag=0   src=0

`0x80003000` is the subject granule. Read straight, that says the flush killed the shadow
load, post-fault responses do not capture, and the frozen record reliably holds the
subject. Which is exactly the answer that made the bitstream worth building.

## The control

Identical file, the faulting `cincoffsetimm` replaced by a `nop`, `diff` confirming that
was the ONLY difference. It came back **bit-identical** — same single capture, same
granule, same cycle number.

So the shadow LDC never captured *even with no fault present*. The positive control did
not fire, and the test never created the condition it was built to test. The matching
number was not evidence of anything.

## Why

The RVFI trace ends at `CHK_START` (`0x8000017c`) in both arms. The subject LDC at
`0x80000180` never retires, so the shadow LDC at `0x80000186` never issues. Both arms run
to the 200013-cycle timeout. The single capture is the subject load's *response* — the
load issues and answers, it just never retires.

Root cause of the non-retirement: the test made the subject return UNTAGGED by clearing
its granule with a plain scalar store, and an LDC over a scalar-cleared granule does not
retire on this core. Unestablished whether that is the LDC faulting, an LSU stall, or the
test; not chased, because the fix routes around it.

## What this cost, and what it did not

One sim iteration. It did not cost a bitstream or a reflash, and it did not produce a
published claim — because the control ran. Without the control this would have been
reported as "the race is settled, post-fault responses do not capture", and the first
board session on the new bitstream would have been the thing that discovered otherwise.

## Already covered, and not a new rule

CLAUDE.md's "A CLEAN result is not evidence until the check is known to fire" names this
exact shape, twice over: *"directed tests that come back clean without ever creating the
triggering condition"* and *"Give every detector a positive control."* The `rtl-sim`
skill's "Reading a negative result" says the same for simulation specifically. **Nothing
here needs adding.** The rule worked; it was followed and it caught the error.

Worth recording anyway, because the failure had a feature the rule does not describe: the
void result was not blank or obviously broken. It was a specific, plausible, *desirable*
number that agreed with the leading hypothesis. That is the case where a control feels
most redundant and is most load-bearing.

## The second attempt

`s12-race-v2.S` removes the untagged load from the experiment entirely — the recorder now
captures every LDC regardless of tag, so both loads can be ordinary tagged loads that
retire normally, and the fault comes from an unrelated illegal instruction. Both arms are
generated from ONE source file via `-DRACE_CONTROL=1`, so they cannot drift apart the way
two copied files can.

Reading, written down before the run:

| fault arm | control arm | conclusion |
|---|---|---|
| granule A | granule B | flush kills the shadow; post-fault responses do NOT capture |
| granule B | granule B | the response wins the race; expect VOID often on the board |
| granule A | granule A | **VOID again** — control did not fire, read nothing |

## Related

- The instrument under test: `s12-ldc-rolling-min` @ `68bcd0860` in `capstone-ariane`.
- [[26-08-2026_11-05-00_fabricated-lut-figures-retracted]] — same session, different
  failure class: that one was a number with no source, this one a number with no meaning.
