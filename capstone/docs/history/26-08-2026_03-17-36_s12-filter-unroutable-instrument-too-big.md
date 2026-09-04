# The S-12 filter did not route: 8h38m, no bitstream, and the warning was in my own audit

**Result: `1cb22e30a` FAILED to synthesise.** Exit 2, 8h38m25s wall against a 1h31m58s baseline for
the same flow on its own base — 5.6x — and **no bitstream at all**.

    ERROR [Route 35-2]    Design is not legally routed. 46 node overlaps.
    CRITICAL [Route 35-162]  76 signals failed to route due to routing congestion.
    ERROR [Common 17-39]  route_design failed due to earlier errors.

Overlaps descended 130,978 -> 69 -> 47 -> 44 across ~40 iterations and five hours, then stalled
above zero. `write_bitstream` **never ran**.

## What cannot be reported, and one zero that must not be misread

No routed design means **no post-route WNS, no failing-endpoint count, and no answer on whether
the new nets appear in failing paths.** The only timing figure produced is an intermediate
mid-route `WNS -15.188`, which is **not** comparable to the base's post-route `-13.516`.

**`DRC LUTLP-1` count is 0 and that zero is MEANINGLESS** — bitgen never ran, so the check that
would flag a combinational loop never executed. The matcher works (126 hits for DRC/critical-warning
text in the same log); the check simply did not happen. Do not read it as "no combinational loop".

## Cause: device occupancy, and the warning was in my own pre-synthesis audit

The base `84ed6eafb` placed at **84.13% Slice LUTs** (171,460 of 203,800) and routed in 92 minutes.
That is already the region where routing goes nonlinear on this part. The change added, on top of
that:

- a **56-bit bus** threaded through four module levels — `csr_regfile` -> `cva6` -> `ex_stage` ->
  `load_store_unit` -> `load_unit`;
- a **56-bit zero-detect** and a **52-bit equality comparator** in the LSU;
- `watchpoint_addr` fanout grown from one commit-stage comparator to a deep LSU consumer.

**The sufficiency audit flagged exactly this and I filed it under the wrong heading.** It said the
change was a fanin *increase* overall and "an honest synthesis/routing delta that the message
understates" — and I recorded that as **"not a correctness risk"** and moved on, because I was
auditing for correctness and silent failure. **Routing capacity is a resource question, and my
audit had no question for it.**

**And CLAUDE.md already says the thing I did not do:** *"Keep an observation-only change a strict
reduction where it can be... An instrument rich enough to be interesting is rich enough to be
unsynthesizable; the minimal version is the one that ships."* The filter is precisely an
instrument rich enough to be interesting.

**Attribution is not proven.** Vivado writes the placed utilization report only after routing
succeeds, so there is **no occupancy figure for `1cb22e30a`** to compare against 84.13%. What is
established: the base routed in 92 minutes at 84.13%, and this did not route in 8.6 hours. A
**placed-only run** would produce the utilization number in well under an hour and settle it.

## The redesign this implies

The change is two parts with very different costs:

- **Dropping `&& !s07_ldc0_valid_q`** is a **strict reduction** — one fewer term in an enable, no
  new ports, no new bus. It makes the design *smaller* than the base that routed.
- **The granule filter** is the entire cost: the wide bus, the wide comparators, the fanout.

**Rolling alone may now suffice, because the vehicle changed under us.** The filter existed because
SQLite issues thousands of legitimately-untagged loads. The repro is now **one iteration, ~20
instructions**. "The last untagged LDC before the fault" is plausibly the subject in a body that
small.

**The residual that stops it being free:** the monitor's trap path issues `LDC(gp, sp, -16)` on
every timer tick, and the recorder keeps rolling after the fault, so a post-wedge trap could
overwrite the record. Closing that needs **either** a freeze (one bit from `cva6`, not 56) **or** a
narrowed filter (compare `paddr[19:4]`, a 16-bit bus and comparator rather than 56/52). Both are a
fraction of what just failed to route.

## Two process points

**`LIMIT_GB=50` was inert**, exactly as the weaker justification predicted — peak 21.48 GB, the
guard fired nothing, and exit 2 is the flow's own return code. The collector never opened a routed
checkpoint because none exists, so its peak was 0.00 GB and the unguarded-collector exposure did
not materialise. **That exposure remains real for successful runs.**

**A falling convergence metric is not evidence that a router will close.** The run was monitored
throughout and reported as "converging, not stuck" on a monotonically decreasing overlap count. It
*was* converging — to a floor above zero. The pessimistic geometric extrapolation, repeatedly
flagged and repeatedly discounted with "the tail often collapses suddenly", was the closer read.
