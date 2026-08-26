# The S-12 recorder bitstream exists, and the collector has been running unguarded all along

**Date:** 2026-08-26
**Status:** bitstream built and verified. NOT flashed. Flashing is the project lead's call.

## The artifact

    bitstream  ariane_xilinx.bit   11,443,722 bytes
    sha256     6d29d3d14457df34cf4bca745093c1c936d7d84b0ca177ce83ad89b59660fef8
    commit     52fa06b9d  (s12-ldc-rolling-min, based on 84ed6eafb)
    artifact   synth-52fa06b9d-exit0.tar.gz, 392,443,910 bytes, on the synthesis machine's
               durable scratch (not /tmp, not auto-cleaned)

The copy inside the tarball hashes identically to the one on disk. **The hash is recorded because
this project has twice referred to a bitstream by a filename nobody produced.** Note the build
produces `ariane_xilinx.bit`; every `caplifive_*.bit` name exists only in the board console's
image store and is applied at upload time.

## Three builds, one variable at a time

                            base 84ed6eafb    ARM A (ret OFF)   ARM B (ret ON)
    guard exit                   0                  0            2 (ceiling)
    total wall               1h32m              6h27m15s        4h21m (killed)
    peak RSS                 21.00 GB           20.88 GB        51.43 GB
    synthesis                41-46 min          254 min         >261 min, never finished
    placement                UNKNOWN            21 min          UNKNOWN
    routing                  UNKNOWN            105 min         UNKNOWN
    post-synth LUTs      171,497 / 84.15%   172,962 / 84.87%    UNKNOWN
    post-place  LUTs     171,460 / 84.13%   170,726 / 83.77%    UNKNOWN
    WNS clk_out1            -13.516            -14.125          UNKNOWN
    failing endpoints        93,241            104,238          UNKNOWN

Arm B's UNKNOWNs are because it never left synthesis, not because anything was lost. The base's
phase splits are UNKNOWN because its artifact carries no top-level run logs.

## What is settled

**The bitstream is a usable debug vehicle.** A COMPLETE launch census — sums exact to 104,238 on
both the startpoint and endpoint axes — shows every failing endpoint launching from
`dom_switcher/cur_idx_q_reg[0]`, with zero from `s07_ldc0`, `load_unit` or `lsu_i` and
`dom_switcher` as the positive control in the same command. `cur_idx` toggles only during a
domain switch and the pipeline is flushed throughout one, so every failing path is inert while
the code under test runs. Same argument that made the base usable at -13.516.

**The 6x synthesis cost is the RTL, not the flow.** base and arm B differ in exactly the 14 lines
of RTL and nothing else; base finished synthesis in 41-46 minutes, arm B never finished in 261.

**Retiming-OFF is what made a bitstream possible.** The flow change that looked like the mistake
of the cycle is the reason an artifact exists: retiming-ON with the same RTL blew 51.4 GB and
produced nothing. The mistake was bundling the flow change with the RTL change in one build, not
making it.

**Eliminated as causes:** congestion (arm A placed BELOW the base at 83.77% and routed in 105
min); timing loops (exactly 100 in base and both arms); retiming (arm B holds it at the base's
setting and is the worse arm). Remaining suspect for the synthesis cost is
`cap_clear_addr_q -> cap_clear_addr_d`, the only part of the diff that adds arcs to an existing
timing-relevant net.

## The finding that generalises: the collector runs unguarded, and it is the high-water mark

    arm A synthesis peak   20.88 GB
    arm A collector peak   33.35 GB      <- 60% HIGHER

`synth-guard.sh`'s monitor loop ends at :216; `collect_artifacts` is called at :227, OUTSIDE it.
So on **every successful build this project has ever run**, the largest memory consumer was the
phase nobody was watching. This was flagged earlier as a theoretical exposure. It is not
theoretical: the unmonitored phase is the peak.

Now fixed on the synthesis machine — collection runs under a guarded wrapper inside the ceiling,
with its peak reported either way.

## Two other defects fixed in the same pass

- **`kill_job` scoped by a start-time PID snapshot**, so any process started after a guard began
  was in scope. Arm B's ceiling kill destroyed **arm A's timing enumeration** (exit=143), which
  is why section 5 and the worst-path files do not exist. Now scoped by run directory.
- **The ceiling path emitted no `exit=` line**, so a ceiling kill was indistinguishable from a run
  still in progress.

`synth-guard.sh` is tracked, so builds after this point carry `M synth-guard.sh` in PROVENANCE in
addition to the three env files. It does not affect the synthesised design, but the dirty state no
longer matches the reference build's exactly.

## Recoverable, not lost

The routed `.dcp` is retained inside the arm A tarball, so the killed enumeration can be re-run:
extract ~400 MB, open the routed checkpoint in Vivado, expect ~33 GB and 15-30 minutes. That is a
Vivado invocation, not a log re-read. Not run, and not proposed — the launch census already
answered the question that motivated it.

## Standing

**Not flashed.** WNS is -14.125 and `run.tcl`'s stated criterion says negative slack means do not
flash. Worth recording honestly beside that: the currently resident bitstream is `84ed6eafb` at
-13.516, so this design has never met that criterion and the project has knowingly used debug
instruments that fail it. Arm A is not categorically different, and the census is a stronger
argument for usability than WNS is against it. The decision is the project lead's either way.
