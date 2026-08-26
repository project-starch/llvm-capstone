# A guard that "worked" and produced two confident wrong answers

**Date:** 2026-08-26/27
**Status:** fixed on the synthesis machine. Recorded because the FIX teaches the next person
nothing about why a working gate lied.

## What it did

`synth-guard.sh` selected the processes it owned by matching every `vivado` belonging to the user
**except those in a PID snapshot taken when that guard started**. Anything launched afterwards was
counted as this run's.

Two runs were in flight. Arm B's guard started at 06:56. Arm A's **collector** started around
11:00 — four hours later, therefore absent from the snapshot, therefore attributed to arm B.

## The two wrong answers, both delivered with confidence

**1. It killed arm B and reported a memory blowup that did not happen.** The ceiling fired at
11:17:48 at 51.43 GB against a 50 GB limit. The arithmetic:

    arm B, this attempt, same RTL and same setting, peak    21.15 GB
    arm A collector peak                                    33.35 GB
                                                     sum  = 54.5 GB, bracketing the 51.43 kill

That kill became the evidence for "retiming-ON does not complete with this RTL", which was stated
as measured fact to two peer lanes and to the project lead, and used to justify a synthesis-flow
deviation as *necessary*. On the rerun, retiming-ON completed synthesis in **213 minutes against
retiming-OFF's 254** — it is the FASTER setting. The claim was not merely unproven, it was
backwards, and it had already propagated into three conversations and a commit message.

**2. It destroyed arm A's timing enumeration.** The same kill took arm A's collector with it
(`exit=143`). Section 5 of the forensics and the worst-path reports never existed. They were
missing for an hour before anyone connected the absence to the kill.

## Why no check caught it

The guard was not broken in any way a test would find. It monitored, it accumulated, it fired at
its threshold, it logged. **Every component worked.** What was wrong was the SET of processes it
believed it owned — a scoping error, invisible in every number it printed, and one that only
manifests when two runs overlap, which had never happened before on this project.

The peak it reported was real. It was the peak of two runs added together.

## The fix, and the part the fix does not carry

Now scoped by **run directory** rather than by a start-time PID snapshot; the ceiling path emits an
`exit=` line so a kill is distinguishable from a run in progress; and collection runs inside the
ceiling with its peak reported.

What the diff does not say, and what this note exists to carry: **a gate can be fully functional
and still answer about the wrong population.** The guard's output was never wrong about what it
measured. It was wrong about what it was measuring. Nothing downstream — not the trace, not the
ceiling arithmetic, not the artifact — could have revealed that, because all of them were
consistent with each other.

Related: the collector was also found to be the true memory high-water mark (33.35 GB against
synthesis's 20.88), and it ran OUTSIDE the monitored block by construction. So on every successful
build this project has ever run, the largest consumer was the phase nobody watched. Also fixed.

## Second incident, same session, different class: prior art re-derived

Two lanes independently spent an afternoon re-deriving a chain that was **already written down in
the folder of the bug being investigated** — the `tval`-passthrough argument excluding tag-only
mechanisms, and the structural exclusion of the competing granule-reordering mechanism, both on
file with their premises verified.

That is the third time on this project that effort has gone into re-deriving a recorded result. The
rule already exists in CLAUDE.md ("Search prior art before investigating, and read PAST the root
cause"). **No new rule is proposed** — the rule was not absent, it was not followed. Recorded as an
instance so the count is visible, since three occurrences is what distinguishes a habit from an
accident.

## Related

- [[26-08-2026_14-30-00_three-wrong-comparisons-one-shape]] — five instances of the measured
  quantity not being the intended one. The guard is the same family: right instrument, wrong
  population.
- [[26-08-2026_16-00-00_s12-recorder-bitstream-built-and-collector-exposure]] — the collector peak
  and the build it came from.
