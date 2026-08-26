# Three wrong comparisons in one session, all the same shape

**Date:** 2026-08-26
**Status:** evidence, not a rule. A rule is proposed at the bottom and is the project lead's call.

Each of these was caught, none reached a published claim, and each cost real time. They are
recorded together because separately they look like three unrelated slips and together they are
one habit.

## The shape

**Two quantities were matched on one axis, differed on another, and were compared as if
equivalent.** In every case the axis that WAS checked is real and relevant, which is exactly what
made the mismatch invisible: the check that ran came back clean and was mistaken for the check
that mattered.

## The three

**1. Placed vs synthesized.** `84.13% / 171,460` (base, PLACED report) was compared against
figures from post-SYNTHESIS reports, and when a grep failed to find the placed number I concluded
the whole set was fabricated and retracted it. Both numbers were real; they measure different
stages. Matched on "utilization report", differed on stage.
Cost: a false retraction, a false claim written into RTL source that would have shipped in a
bitstream, and the same false claim relayed to two peer lanes.

**2. Entire-flow vs synthesis-phase.** A run at 1h50m *in synthesis* was called off-trajectory
against a base that took 1h32m *for its entire flow* — synthesis, placement, routing and bitgen.
The correct comparison turned out worse, not better (base synthesis was 41-46 minutes), but the
alarm as originally framed compared a part against a whole. Matched on "elapsed time", differed
on scope.
Cost: an escalation raised on a figure that did not support it, though the underlying concern
proved real.

**3. Stage-matched vs condition-matched.** Arm A (172,962, retiming OFF) was compared against the
base (171,497, retiming ON) and labelled apples-to-apples because both were post-synthesis
reports. Retiming is a register-movement pass that directly moves LUT count, so the +1,465
attributes to neither the RTL change nor the setting. Matched on report stage, differed on build
conditions.
Cost: a cost-of-change figure that had to be withdrawn, and a routing-risk assessment briefly
built on it.

## Why the existing rules did not catch it

The standing rules are about DETECTORS -- give every check a positive control, ask what the
instrument cannot distinguish, make "no data" an error. All three failures here had working
instruments returning correct numbers. Nothing was broken and nothing was silent. The error was
in the pairing, downstream of every gate we have.

Note also that #1's positive control FIRED and still did not help: it proved grep could find a
string in the searched tree, while the live risks were another host and a compressed archive.
A control has to exercise the failure mode actually faced, and for a comparison the failure mode
is not "can I measure" but "are these two things the same kind of thing".

## Proposed rule -- NOT ADOPTED, needs the project lead

Smallest wording that would have caught all three, offered for the lead to accept, sharpen or
reject:

> **Before comparing two numbers, name every axis on which they could differ, not just the one
> you matched.** Stage (synthesized/placed/routed), scope (phase/whole flow), build conditions
> (flow settings, RTL revision, machine). A comparison that matches one axis and silently differs
> on another produces a confident wrong delta, and every gate we have is blind to it because both
> numbers are individually correct.

Considered and rejected: adding this to the "CLEAN result is not evidence" section, because that
section is about instruments that fail to fire and these instruments all fired correctly;
a per-artifact convention such as always labelling figures with their stage, because it is more
machinery than the problem needs and would not have covered #2 or #3.

## Related

- [[26-08-2026_11-05-00_fabricated-lut-figures-retracted]] -- incident #1 and its un-retraction.
- [[26-08-2026_12-40-00_waw-refutation-is-structural-not-measured]] -- same session, different
  shape: a measurement that could not fire at all.
