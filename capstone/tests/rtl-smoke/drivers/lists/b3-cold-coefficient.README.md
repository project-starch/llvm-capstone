# `b3-cold-coefficient.txt` — B3, the cold per-node revocation constant

**One line per repetition, and ONE LINE PER BOOT — not twelve.** Each invocation runs the whole
`wide` series, all twelve points, so a line is a repetition rather than a point. `board-r1e4.sh`
slices twelve list lines per boot, so this list must be fed to it **one line at a time** (a temp
single-line list per boot); feeding it whole would run all five repetitions in one boot.

## Why one repetition per boot

- **Node budget.** One invocation mints **41,280** nodes — 63 % of the 65,532-index pool. The pool
  does not reset between invocations *within* a boot, only across boots. Reclamation makes five in
  one boot probably fine (the M1 pilot reached 200,000 allocations without exhausting), but
  "probably" is not a basis for a five-boot campaign, and a run that exhausts traps and loses its
  end line.
- **Independence.** R1's precedent is repetitions in distinct boots, and its own repeatability
  figure rests on that.

## Why the arena is 8 MiB and not 4

`carve_root` never reuses: each point carves a fresh pool and fresh `leaf[]`/`alias[]` backing, so
the requirement is the **sum across the twelve points — 2.83 MiB — not the largest (1.12 MiB)**.
4 MiB would leave almost no headroom over the root, which must stay non-empty.

## The gate

`R1_QEMU_GATE='R1 s='` with `R1_QEMU_GATE_MIN=5`. The driver's default `"R1 lat"` matches **nothing**
this harness emits and would refuse every boot. Emulator log: all twelve points, `nd = 2n` exact,
`bad=0 ok=1`, zero R1 refusals.

## Pre-registered, before the first boot

Warm points (`nd` ≤ 512) reproduce R1's **≈22.91 cycles/node** — which doubles as the check that
making `leaf`/`alias` pointers perturbed nothing. Deep-cold (`nd` 8192, 16384) rises toward ~5×,
from the 9.00-vs-48.2-cycle dependent-load gap. **Whether the transition is a step or a ramp is
deliberately unpredicted** — that is what distinguishes pure capacity from capacity plus something
else.

The same-record positive control is `bk` and `fl`, which rise 470 → 114,710 and 1,289 → 327,689
across the identical records. A flat `rv` against those is a property of revocation; without them it
would be an unproven instrument. *A flat sweep with no witnessed separation is VOID, not negative.*
