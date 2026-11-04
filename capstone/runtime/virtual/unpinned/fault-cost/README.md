# Unpinned page-fault experiment

These are lane N's `perf/linux-question-micro` tools at
`390a4043851a9995be74b726fc72ceec1cd6a7b8`, extended with `fault MIB WARM`.
Mapping, verification and unmapping occur outside the fault window. A cold
window stores one word per page; a warm window first populates the same pages
outside the window. Every fault-mode run verifies every stored word.
The original `touch` mode includes mapping and unmapping and is retained as
a separate whole-path measurement. No THP is used.

Build `lqmicro.c` with the shared virtual SDK's `capstone-cc -O2` and the plain
RISC-V musl compiler's `-O2`, naming the outputs `lqmicro-cap` and
`lqmicro-plain`. The runner also expects lane N's unchanged `lq-hog` binary,
even for these modes where it does not run it. The runner header documents
its environment; set all writable directories to the experiment's own lane.
It enables raw moves, disables test CSRs and forwards module parameters at
initial load. Use the matching patched kernel and model, with `unpinned=1`.

Run the following list once as `capstone` and once as `plain`:

```
-- fault 4 0
-- fault 64 0
-- fault 4 1
-- fault 64 1
-- touch 4
-- touch 64
```

Let C and W be cold and warm window attempts. The fault estimate is
`((C64-C4)-(W64-W4))/15360`. This removes fixed setup and the successful
program loop, retaining the faulting instruction's extra attempt. Function
counts use the same linear combination. This is a marginal cost, not a
single-fault latency trace. Use capprof's per-window `all_functions`, whose
counts are exclusive of callees; do not sum inclusive call costs.

Baseline on runtime `1ad3511096f0` (module unchanged from `c866e1d2fe76`),
QEMU `0542c5556c2d`, patched Linux 6.1, cap4 SDK and plain musl, p12:

```
A fault pages 15360 total/page 4700.1548828125 busy/page 4601.07109375
plain fault pages 15360 total/page 4093.0784505208335 busy/page 3994.7899088541667
A touch pages 15360 total/page 6521.627083333334 busy/page 6299.8626953125
plain touch pages 15360 total/page 5926.385807291666 busy/page 5703.329036458334
```

`busy` applies the north-star function-bucket exclusions, which are an
accounting proxy; generic mm list helpers can fall into the excluded fs
bucket. Retain total counts alongside them. Every invocation returned zero
with one complete window; fault modes reported `bad=0`.

Raw evidence: p12 `~/lanes/laneM7a/runs/{A,plain}` and
`~/lanes/laneM7a/logs/{baseline-cost.txt,micro-A-plain.json}`. No timing,
cycle, full SPEC or complete runtime qualification follows from these runs.
QEMU counts CSRUNV as one instruction, without pricing its internal state
save/restore work. These measurements cannot justify direct trap delivery
on the grounds of hardware switch latency.
