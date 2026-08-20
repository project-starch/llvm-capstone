# The bitstream misses timing by 10.6 ns — and it is the domain switcher, not the S-07 fix

**Date:** 2026-08-20
**Build analysed:** `fpga-e1140aeea.tar.gz` (archived synthesis output, `/tmp/capstone/_bitstreams`)
**Bitstream:** `caplifive_s07fix.bit`, silicon-confirmed as the stripped tree
**Status:** the S-07 fix is EXONERATED as the cause. The timing failure is real, structural,
and pre-existing.

## The number

Post-route summary (`work-fpga/ariane_xilinx_timing_summary_routed.rpt` — the only timing
summary in the archive, so there is no post-place file to confuse it with):

```
WNS   -10.629 ns
TNS   -438671.250 ns
TNS failing endpoints   96727 / 246476      (39% of all endpoints)
WHS   +0.054 ns                             hold is FINE — this is setup only
WPWS  +0.062 ns
```

## It is one clock, and the constraint on it is correct

Every other clock closes. The failure is entirely on the core clock:

| clock | period | WNS | failing endpoints |
|---|---|---|---|
| `clk_out1_xlnx_clk_gen` | 40.000 ns (25 MHz) | **−10.629** | **96727 / 174481** |
| `clk_out2_xlnx_clk_gen` | 8.000 ns (125 MHz) | +0.694 | 0 / 427 |
| `sys_clk_p` | 5.000 ns (200 MHz) | +1.063 | 0 / 112 |
| `eth_rxck` | — | +4.140 | 0 / 612 |
| all MIG/DDR derived clocks | — | positive | 0 |

**This kills the "misconstrained clock domain" hypothesis.** `clk_out1` is constrained at
40 ns / 25 MHz, which is exactly what `xlnx_clk_gen` is generated for. The constraint is right;
the design genuinely does not make 25 MHz.

## Where the failure is

> ### RETRACTION, 2026-08-20 — the "0 of 100" figure was TRUE AND VACUOUS
>
> This section originally read *"100 of the 100 worst violated paths"* and *"violated paths
> touching `wbuffer` or `dcache`: 0 of 100, positive-controlled"*. **Both sentences are
> withdrawn.** Re-measured:
>
> | report | path blocks | distinct sources | distinct destinations | distinct (src,dst) |
> |---|---|---|---|---|
> | `ariane.timing_WORST_100.rpt` | 100 | **1** | **1** | **1** |
> | `ariane_xilinx_timing_summary_routed.rpt` | 766 | 193 | 568 | 569 |
> | `ariane.timing.rpt` | 30 | 27 | 30 | 30 |
>
> `report_timing -nworst 100` returns up to 100 paths **per endpoint**, not 100 endpoints.
> WORST_100 has a single (source, destination) pair and a single slack value, `-10.629`: it
> characterises **one endpoint pair by 100 different routes**. So "100 of 100" was one endpoint
> counted a hundred times, and the figure says nothing whatever about the other 96726 failing
> endpoints.
>
> **And the matcher could not have fired.** Across all 896 path blocks in the three reports —
> 760 distinct endpoint fields, violated and met alike — `wbuffer|dcache` appears in **zero**
> Source or Destination fields. These are worst-N samples of roughly ten per path group; the
> dcache is not among the worst, so it never appears in that field class at all. The check
> cannot separate *"the dcache is clean"* from *"the dcache was never sampled"*.
>
> **The lesson, and it is not the one recorded first.** The original control checked that
> `wbuffer`/`dcache` appear *somewhere in the file*. That was improved to checking the matcher
> fires on the same *line class* (`issue_stage|scoreboard` returns hits on Source/Destination
> lines). That improvement was real and **still insufficient**: both controls interrogate the
> matcher, and the defect was in the **sample**. The question neither control asked is *what is
> the denominator, and can the target appear in it at all* — here it provably cannot, because
> the target is absent from the entire field class by construction of the report.
>
> Settling dcache coverage properly needs the Vivado machine: re-open the routed `.dcp` and run
> `report_timing -nworst 1 -max_paths 100000 -slack_lesser_than 0 -sort_by slack`, then group
> sources by module. That enumerates instead of sampling. The archive has no checkpoint, so it
> cannot be done from here. **It is not required for the exoneration below**, which never
> depended on it.

**What survives, from the routed summary — a better report, and a genuine finding.** Its
violated set is 10 Setup paths from **one** source register bit to **ten distinct**
destinations:

```
Source:        i_ariane/i_cva6/dom_switcher/cur_idx_q_reg[1]/C        (1 distinct, all 10)
Destinations:  i_scoreboard/mem_q_reg[N][sbe][rs1][k]/CE              (8 of 10)
               i_issue_read_operands/fu_data_q_reg[0][cap_data][cap_metadata_a][20]/D  (2 of 10)
Slack range:   -10.629 .. -10.565 ns
```

This is still a **sample**, not an enumeration — it is the worst ten, not the failing 96727.
What it establishes is that the worst failures are domain-switch machinery. It does not
establish that nothing else fails.

The worst path in detail:

```
Slack (VIOLATED)   -10.629 ns
Requirement         40.000 ns
Data Path Delay     50.536 ns   (logic 8.984 ns / 17.8%,  route 41.552 ns / 82.2%)
Logic Levels        123         (CARRY4=30 LUT2=7 LUT3=10 LUT4=8 LUT5=19 LUT6=49)
Clock Path Skew     -0.069 ns
```

**123 logic levels in one 25 MHz cycle**, with 82% of the delay in routing, all fanning out
from a single register bit. That is a structural property of `capstone_dom_switcher`, not
something a change elsewhere can induce.

## Why it cannot be ours

Three independent facts, each verified from git rather than taken on report:

1. **The design delta is one module-internal file.** `git diff --stat 618f4ce36 e1140aeea` over
   `core/ corev_apu/src/ corev_apu/fpga/src/ corev_apu/tb/` returns exactly
   `wt_dcache_wbuffer.sv`, +146/−1. Of 147 changed lines, **zero** touch an `input`/`output`/
   `inout` declaration — positive-controlled, the same pattern finds 51 port declarations in the
   file body. The change cannot be observed from outside the module.
2. **The timing environment is byte-identical.** Empty diff across
   `corev_apu/fpga/constraints/`, `corev_apu/fpga/xilinx/`, `corev_apu/fpga/Makefile`,
   `fpga-env.sh` and the top-level `Makefile`. `run.tcl` gains 28 lines, **0 non-comment**
   (checked programmatically). `RETIMING true` and both `RuntimeOptimized` directives were
   already present at `618f4ce36`.
3. **The failing module has not been touched in this work at all.** `core/anvil_build/` is
   byte-identical across the range, and `capstone_dom_switcher.anvil` last changed at
   `25035c4c0` (2026-08-14) — an ancestor of `618f4ce36`, the healthy reference.

For the S-07 fix to have caused this, a change confined to the inside of one cache module would
have to create a 123-level, 50.5 ns path in the domain switcher — or move 96727 endpoints. There
is no mechanism.

**None of these three facts comes from the timing reports**, which is why the exoneration is
unaffected by the retraction above. They are properties of the git history and are checkable
without a Vivado machine. The retracted figure was corroboration that turned out to be vacuous;
it was never the argument.

## Contributing factor: the design is near capacity

```
Total LUTs   169415 / 203800 on xc7k325t   ~83%
  csr_regfile_i    33218 LUTs
  ex_stage_i       33484 LUTs
  dom_switcher      6144 LUTs
```

83% LUT occupancy is congested territory, which is consistent with 82% of the critical path
being routing rather than logic. Both `place_design` and `route_design` run with
`-directive RuntimeOptimized` — the fastest, least effort setting — in this build and in
`618f4ce36` alike.

## Why the board nonetheless works — hypothesis, NOT a conclusion

`cur_idx_q` is the current domain index. It changes only on a domain switch and is then stable
for many thousands of cycles. If the destinations only need a settled value — rather than
capturing it on the cycle immediately after it changes — the path is architecturally multicycle
while being constrained as single-cycle, and a 50.5 ns path resolves comfortably within two
40 ns cycles.

That would explain the whole picture at once: a hard setup violation coexisting with repeatable,
correct silicon behaviour. **It is a hypothesis.** Confirming it means reading the dom-switcher
handshake to establish how many cycles the consumers are actually given, and it is a question
for whoever owns that RTL. If it is right, the fix is a `set_multicycle_path` constraint rather
than an RTL change.

## What this does and does not do to the S-07 results

**Does:** removes the S-07 fix as a candidate cause. The exoneration is decisive.

**Does not:** make the results unconditional by itself. They were taken on a bitstream that
fails setup timing. What supports them is unchanged and independent:

- every result is **differential** — `wb1` vs `wb3`, `wb1` 1107→0, `wb2` 15193→16384, `wr6` vs
  `wr7` — arms differing by exactly one thing on one bitstream, and a setup violation has no
  reason to track an architectural mechanism arm-for-arm;
- the failing cone is **domain-switch machinery**, not the load/store path under measurement;
- the silicon agrees with cycle-level Verilator behaviour, which has no timing model at all.

**Repeatability is NOT part of that argument** and was withdrawn: a setup-violating path at
fixed voltage and temperature can fail deterministically, so four clean runs are consistent with
a timing failure rather than evidence against one.

## Consequence for every prior silicon result

The constraints, the flow, the directives and the dom-switcher RTL are all unchanged back to at
least `618f4ce36`, and the failing path is in a module untouched since `25035c4c0`. So on the
evidence available, **this build is not special** — every silicon measurement this project has
taken was very likely taken on a bitstream missing timing the same way. No timing report from an
earlier build survives to confirm it, and the acceptance-criterion block in `run.tcl` telling
anyone to read WNS post-route is itself part of this diff: it did not exist at `618f4ce36`, so
that build's WNS was in all likelihood never read either.

This is worth its own issue against the hardware side, separate from S-07.

## Open

- Confirm or refute the multicycle hypothesis by reading the dom-switcher consumer handshake.
- If refuted, `capstone_dom_switcher` needs pipelining — 123 logic levels will not close at
  25 MHz by placement effort alone, though moving off `RuntimeOptimized` is the cheap first try.
- Re-running implementation on `618f4ce36` unchanged (~2h, no board) would convert
  "very likely long-standing" into "measured". Only worth it if someone disputes it.
