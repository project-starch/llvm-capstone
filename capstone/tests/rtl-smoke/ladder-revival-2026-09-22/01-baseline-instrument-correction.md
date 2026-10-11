# I used the RETIRED baseline instrument, and the registry said so (2026-09-22)

## What happened

Phase 1 and phase 2 measured the baseline half with `run_ladder_base_fpga.py`, the **Linux-userspace**
baseline. Its control rung read `clean = 1/15` — never one uninterrupted pass — and I spent the
evening diagnosing that as a live defect in the measurement vehicle.

It is a **known, fixed issue**, and the fix has existed since July. `ISSUES-ARCHIVE.md:4221`:

> ### I-2 — Linux baseline served interrupts inside the bracket `FIXED`
> **Fixed 2026-07-28 by removing the OS**, not by modelling the error. The baseline now runs as a
> bare-metal S-mode OpenSBI payload (`build-ladder-base-bare.sh`, `fpga_driver/run_base_bare_fpga.py`).
>
> - **Proof:** the `ctrsanity` control — identical 5-instruction loop on both sides — reads
>   **600,041 cyc bare vs 600,309 cyc capability, ratio 1.000** (Linux was 728,727, 1.21×). Quality
>   went from **1/15 passes tied at min instret** to **15/15 with spread 0**.

**`1/15` is the exact number I measured today**, on the exact rung, and read as a new finding. The
published §2 table even names its instrument in its own heading — *"uniform −O1, **bare-metal
baseline**, one harness, one session"* — which I quoted while extending that section without noticing
it described a different vehicle from the one I was running.

## What this invalidates, and what survives

**Invalidated — a re-derivation of documented prior art:**

- The framing "the baseline vehicle is broken and needs repair". It was repaired on 2026-07-28 by
  replacing it. I used the superseded one.
- The conclusion "the overhead table cannot be extended until the baseline is fixed." It can; the
  right instrument existed the whole time.
- The length-vs-`clean` relationship I measured across eleven rungs is real but is a property of the
  **retired** instrument, so it describes something nobody should be using.

**Survives, because it is about a different thing:**

- **Every capability-half measurement.** Those are domain runs, scored against native oracles, and
  they reproduce to the instruction across boots. The headline — all eight previously-unmeasurable
  rungs now measure — does not depend on any baseline.
- **The dispatch-table defect is genuinely new.** The *Linux* controller's rung table was
  hand-maintained and stopped at sixteen entries, so five rungs were silently never dispatched. The
  bare-metal controller's equivalent table is already generated (`ladder_base_bare.c:189`:
  *"Generated rather than hand-maintained: the Linux controller keeps its own table"*), which is why
  this never bit there. The fix stands on its own.
- **The floor-versus-pass-2 reporting fix** stands: even on the retired instrument, printing one
  sample beside a computed minimum is wrong, and it is what produced the retracted 7.8 % figure.
- **`ctrsanitys`** stands and is in both controllers.

## The check that would have caught it, and why the existing rule did not

CLAUDE.md already says to search the issue registry before investigating, including archived entries.
I searched `ISSUES.md` for `R-1` at the start of this work and found the archive that way — so the
habit fired, but only for the *subject* of the experiment and never for the *instrument*.

The cheaper discriminator was in front of me twice and I read past it both times: the runner's own
comment cites "issue I-2" for its least-disturbed-pass logic, and the published table's heading names
a bare-metal baseline. **A tool that cites an issue number in its source is telling you its own
limitation has a registry entry.** Reading that number costs one grep.

## Consequence for the numbers

No ratio computed against the Linux baseline is quoted anywhere — not in §2, not in the summaries.
The pairing gate refused to emit on phase 1 and, where it emitted on phase 2, those rows are
superseded by the bare-metal run rather than published.

---

## The bare-metal baseline, run (2026-09-22 20:02–20:03)

65 rungs in **81 seconds** — the 2.2 MB firmware loads in a fraction of the 15 MB Linux image's time,
which is I-2's documented side benefit. Quality is what I-2 promised: **`15/15 passes at min instret,
spread=0`** on essentially every rung, against `1/15` on the retired instrument.

**It reproduces the published July baselines exactly**, two months and several bitstreams later:

| rung | bare-metal 2026-09-22 | published 2026-07-28 |
|---|---|---|
| `ctrsanity` | 600,041 / 500,022 | 600,041 / 500,022 |
| `beebs_prime` | 9,283 / 2,704 | 9,283 / 2,704 |
| `beebs_aha_mont64` | 283,612 / 256,622 | 283,612 / 256,622 |
| `rv8_sha512` | 540,073 / 462,646 | 540,073 / 462,646 |
| `rv8_sha512s` | 117,035 / 69,108 | 117,035 / 69,108 |
| `beebs_ns` | 88,451 / 62,097 | 88,451 / 62,097 |

Cycles and instructions both, to the digit. The baseline half has not moved; everything that moved is
on the capability side or in the silicon.

## Overhead against a clean baseline, `caplifive_m1_054cea69b`

| rung | cap cyc | base cyc | **cycles** | instr | CPI | 2026-07-28 |
|---|---:|---:|---:|---:|---:|---:|
| `beebs_prime` | 9,684 | 9,283 | **1.043×** | 1.000 | 1.044 | 1.054× |
| `beebs_aha_mont64` | 286,485 | 283,612 | **1.010×** | 1.000 | 1.010 | 1.023× |
| `rv8_sha512` | 543,646 | 540,073 | **1.007×** | 0.996 | 1.010 | new |
| `rv8_sha512s` | 118,603 | 117,035 | **1.013×** | 1.092 | 0.928 | new |
| `beebs_crc32` | 48,310 | 42,795 | **1.129×** | 0.959 | 1.177 | new |
| `beebs_ns` | 102,661 | 88,451 | **1.161×** | 0.987 | 1.176 | new |
| `coremark_matrix` | 53,345 | 36,691 | **1.454×** | 1.229 | 1.183 | new |
| `matmult_int` | 19,134 | 11,312 | **1.691×** | 1.337 | 1.265 | new |
| `beebs_janne` | 593 | 323 | **1.836×** | 0.943 | 1.947 | new |
| `beebs_insertsort` | 2,518 | 1,193 | **2.111×** | 1.183 | 1.784 | new |

**Two published rows reproduce within ~1.3 %** — `beebs_prime` 1.054 → 1.043, `beebs_aha_mont64`
1.023 → 1.010 — which is the cross-check that the pairing is sound.

## ⚠ The control rung does NOT reproduce, and that blocks quoting these beside the published table

`ctrsanity` published at **1.000×** cycles. It now reads **1.167×**.

Its instruction ratio is 1.00002 — eight instructions out of 500,030 — so the halves are matched and
the vehicle is certified. Its *baseline* is byte-identical to July (600,041). Everything that moved is
the capability half: **600,309 → 700,268 cycles for the same instruction count.**

So on this bitstream, capability-mode execution of a pure-compute loop costs 16.7 % more cycles than it
did on the July build, with no change in the work done. The second control makes it stranger rather
than clearer: `ctrsanitys`, the same kernel at 1/100th the length, reads **1.045×**. The penalty grows
with the length of the run — away from 1.0, not toward it — which is neither the fixed effect nor the
proportional one that pair of rungs was built to separate.

**Consequence.** The ten rows above are sound measurements *of this bitstream*. They may not be merged
into §2's table, because that table's control reads 1.000× and this one's reads 1.167×, so a reader
comparing rows across the two would be reading a silicon change as an ABI cost. What settles it is
either an explanation of the control's 16.7 %, or a re-measurement of the old rows on this bitstream —
and the second is now cheap, because the bare-metal baseline runs 65 rungs in 81 seconds.

Four rows carry instruction ratios below 1.0 (`beebs_janne` 0.943, `beebs_crc32` 0.959, `beebs_ns`
0.987, `rv8_sha512` 0.996). Both halves are built today from one spec at one `-O`, so these are
matched measurements and the capability target simply retires fewer instructions for those kernels —
but the published table never went below 0.999, so they are flagged rather than explained.
