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

---

# ADDENDUM: the delta REVERSES SIGN between stages, measured

The strongest single reason to stage-qualify a LUT figure on this design is not an argument, it
is this measurement. Same two commits, same flow, two stages:

    post-SYNTHESIS   base 171,497  ->  arm A 172,962    +1,465
    post-PLACEMENT   base 171,460  ->  arm A 170,726      -734

**The synthesis-stage excess does not survive placement, and the sign flips.** A change that looks
like it costs 1,465 LUTs after synthesis is 734 LUTs *smaller* than its base after placement.

So an unqualified "this change costs N LUTs" is not merely imprecise here, it can carry the wrong
SIGN, and the routing-risk conclusion drawn from it inverts with it: +1,465 reads as "approaching
the congestion cliff that killed 1cb22e30a at 85.05%", while -734 reads as "placed less densely
than a build that routed in 92 minutes". Those are opposite decisions from the same pair of
commits.

Caveat kept attached, because it is the very error this note is about: arm A is retiming-OFF
against a retiming-ON base, so neither figure is the cost of the RTL change alone. Both are
your-RTL-plus-retiming-off. The clean measurement is arm B, retiming ON, RTL the only variable.

# ADDENDUM 2: a fourth instance, and it belongs to a second family

The milestone ladder tracking these builds pointed its 50/75/100 markers at `work-fpga/`. The
Makefile only populates that directory by a `cp` AFTER `run.tcl` returns, so all three markers
would have fired at once at the very end and reported nothing during the run. During a run the
reports live in `ariane.runs/impl_1/`. Found because a build was visibly routing while the
tracker still said 25%.

That is not a comparison error, so it does not extend the list above. It is the SECOND family
this session produced, and the two are worth naming apart:

* **Family A -- compared two things that are not the same kind of thing.** Incidents 1-3 above.
* **Family B -- looked for the right thing in the wrong place.** This ladder; and the grep in
  incident 1 that searched one machine and could not see inside `.tar.gz` archives, which is what
  produced the false retraction.

Family B is the more dangerous of the two, because its failure output is an ABSENCE — an empty
grep, a marker that never fires — and an absence reads as a clean result. Family A at least
produces a number somebody may sanity-check.

The proposed rule above addresses Family A only. If the lead wants Family B covered too, the
smallest wording that would have caught both instances is:

> **A check that finds nothing must state where it looked.** An empty result is only evidence if
> the search covered the places the subject could actually be — other machines, inside archives,
> directories not yet populated at that point in the flow.
