# RETRACTION: three LUT/occupancy figures were fabricated, and nearly reached synthesis

**Date:** 2026-08-26
**Status:** retracted before the numbers left this machine. Caught by an audit, not by me.

## What was claimed

A draft commit message and an accompanying RTL comment block in `core/load_unit.sv` stated, of
the granule-filter build `1cb22e30a` that failed to route:

- **"+1,840 LUTs"**
- **"occupancy 84.15% -> 85.05%"**
- and, as the base figure, **"84.15%"**

All three were written as measurements, in the tone of measurements, next to figures that *are*
real (`8h38m25s`, `46 node overlaps`, `76 unroutable signals`).

## What is actually on record

    $ grep -rn '1840\|1,840\|85\.05\|84\.15' capstone/agent-handoff/
    (no output, exit 1)

    $ grep -rn '84\.13' capstone/agent-handoff/
    26-08-2026_03-17-36_s12-filter-unroutable-instrument-too-big.md:25 ... **84.13% Slice LUTs** (171,460 of 203,800)

The positive control fires three times; the three figures do not exist anywhere. So:

- `84.15%` is a **corruption** of the recorded `84.13%` -- a digit changed in transcription.
- `+1,840 LUTs` and `85.05%` are **inventions**. Worse, the same history doc these were supposedly
  drawn from says in as many words that they *cannot* exist
  (`26-08-2026_03-17-36_...:45-47`): *"Vivado writes the placed utilization report only after
  routing succeeds, so there is no occupancy figure for `1cb22e30a` to compare against 84.13%."*

I had read that sentence in the same session in which I wrote the contradicting number.

## Why this class is dangerous here specifically

The numbers were not decoration. They were the **causal argument** for the design decision:
"the filter cost +1,840 LUTs, which pushed occupancy 0.9 points, which caused the routing
failure, therefore the replacement must add nothing." That story is unsupported, and the
design's own history contradicts it in magnitude -- `39b21639d` (169,386 Slice LUTs, verified in
the surviving `ariane_xilinx_utilization_placed.rpt:35`) to `84ed6eafb` (171,460) is
**+2,074 LUTs for a 32-line diff**, larger than the number I invented for the entire filter.
LUT count on this design is not a simple function of RTL size, so a fabricated delta does not
merely mislead about one build -- it teaches the next reader a false model.

They were also headed into an **RTL source comment**, which is the most durable and least
audited place a number can land. A commit message is read once; a comment next to a
synthesis-driving design decision is read by everyone who touches that line afterwards.

## The correction

Message and comment now carry only: the base placed at 84.13% Slice LUTs (171,460 of 203,800)
and routed in 92 minutes; the filter build's cost is **unmeasured and unmeasurable after the
fact**, and is stated as such. The +/-2,000 LUT noise band from the 39b21639d->84ed6eafb
datapoint is recorded so the next post-synthesis number is not over-read in either direction.

## What would have caught it

Nothing we had. The lint gate, the sim suite and `precommit-scan.sh` are all blind to a number's
provenance -- the change was *correct*, the gates were green, and the prose was wrong.

The check that did catch it is one line, and it is cheap enough to be routine:

    grep -rn '<each number about to be asserted>' capstone/agent-handoff/

with a **positive control** -- grep a figure you know is in the record, and confirm it returns
hits. Without the control, an empty result reads as "nothing to contradict it" instead of
"this has no source." That is the same instrument-blindness rule this project already applies
to detectors, applied to citations.

**Generalisation worth considering (NOT yet adopted -- CLAUDE.md is the project lead's file):**
before a number enters a commit message, an RTL comment, a repro README or the paper, it must
either be pasted from a command run in that session or carry a `file:line`. A number that is
neither is fabricated, whatever it felt like when written.

## Related

- `26-08-2026_03-17-36_s12-filter-unroutable-instrument-too-big.md` -- the source that was
  misquoted, and which states the figure cannot exist.
- The corrected change is `s12-ldc-rolling-min` @ `1fbf62bc1` in `capstone-ariane`.

---

# CORRECTION, 2026-08-26 later the same day: THE RETRACTION ABOVE WAS WRONG

**The three figures were real. This document, up to this line, is itself the error.**

`+1,840 LUTs`, `85.05%` and `84.15%` were not inventions. They came from **post-SYNTHESIS**
utilization reports, and the base figure `84.13% / 171,460` came from the base's **placed**
report. Both live in `synth-84ed6eafb-exit0.tar.gz` and the corresponding failed-build archive
**on the synthesis machine**, not on this one.

    84ed6eafb  synthesized  171,497  84.15%        1cb22e30a  synthesized  173,337  85.05%
    84ed6eafb  placed       171,460  84.13%        1cb22e30a  placed       MISSING

Every figure reconciles against the 203,800-LUT device, and the deltas are apples-to-apples:

    173,337 - 171,497 = 1,840   (synthesized vs synthesized -- the filter's cost)
    171,460 - 169,386 = 2,074   (placed vs placed -- the 32-line-diff move)

## Why the check failed, which is the part worth keeping

The grep that "proved" the numbers unsourced ran over the repo and the local scratchpad. It
searched **this machine only**, and it could not see **inside `.tar.gz` archives** — which is
where synthesis artifacts are kept. Both blind spots were invisible in the result: a `grep -rIl`
returning nothing looks identical whether the artifact is absent or merely unreadable by that
command.

The positive control I ran (`84.13` returns hits) did **not** protect against this. It fired on
prose in `agent-handoff/`, proving only that the grep worked on uncompressed text in the searched
tree. **A positive control has to exercise the same failure mode as the thing being tested.**
Mine tested "can grep find a string", when the live risks were "can grep see another machine" and
"can grep see inside an archive". It answered neither.

That is a sharper form of the rule this project already has. Not "give the detector a positive
control" — that was done — but: *the control must live where the subject lives.* A control in
plain text says nothing about a subject in an archive on another host.

## What this cost

Nothing shipped wrong, but real damage was done to the record and had to be undone:

- an RTL comment was rewritten to state the filter's cost was "UNMEASURED and unmeasurable" —
  **false**, and it would have shipped in the instrument bitstream's source. Corrected.
- the same false claim went to two peer lanes, one of which had used the correct figure
  correctly and was told it was unfounded;
- the "no base number exists, so the placed-only run gives us our first" justification for the
  synthesis run was wrong. The run is still worth making, for the reason the synthesis lane gave
  instead: **we have no placed figure for any MODIFIED build**, so an early placed report is the
  first thing comparable against the base's 84.13% — and it lands before routing decides
  anything, which matters on a run that may not route.

## What survives from the original retraction

One thing, and it is small: the claim at `26-08-2026_03-17-36:45-47` that *"Vivado writes the
placed utilization report only after routing succeeds, so there is no occupancy figure"* is still
wrong, just differently than I said. The **placed** report for `1cb22e30a` is genuinely missing.
The **synthesized** one is not, because `run.tcl:127` writes it before `launch_runs impl_1`. So
the failed build's cost was recoverable all along.

Also unchanged: **compare like with like.** Placed against placed, synthesized against
synthesized. The two deltas above are different measurements of different things, and mixing
them is how a real number starts looking like a fabricated one.
