# The M1 reclaiming lists, and which image each one needs

An M1 list is only half of a run: the other half is the image, because `M1_MAXRET` is a build-time
define and three of the four Design arms are bounded by it rather than by `--cap`. A list run against
the wrong image measures the buffer and reports a clean-looking `stop=buffer`.

| list | arms | image it needs | why |
|---|---|---|---|
| `m1-pilot.txt` | `drop` ×3, rising `C` | any | the platform pilot; `C` small enough that the buffer never binds |
| `m1-series.txt` | all four at `C` = 65,532 | none that exists | see below — `release` is unreachable at this `C` |
| `m1-series-nopressure.txt` | `drop`, `ring`, `release` at `C` = 65,532 | `1b7a04fe237e1580` (`M1_MAXRET`=4096) | drove boot 2 on 2026-09-18; `drop`/`ring` complete, `release` truncates |
| `m1-order9-ctl.txt` | `drop`, `ring` at `C` = 65,532 | `c685b0d7a95492ef` (`M1_MAXRET`=4096, `R1_STACK`=1 MiB) | the allocation-order control — see below |
| `m1-maxret43296.txt` | `pressure`, then `release` at `C` = 4,329 | `249cfda958f22f16` (`M1_MAXRET`=43,296) | the largest buffer that stays at allocation order 8 |
| `m1-relbuf43296.txt` | `release` at `C` = 65,532 | `29d9099326304ee5` (same, `M1_RELEASE_AT_BUFFER=1`) | the re-specified trigger — see below |

## Which arms are bounded by `M1_MAXRET`, measured rather than assumed

In `run_m1`, an arm that is neither `drop` nor `ring` takes the retain-every-alias branch, and the
`release` arm's phase 2 is gated on `alloc >= target`. So:

- `drop` retains nothing and `ring` retains 16 — **neither needs the buffer**, at any `C`;
- `pressure` always stops at `alloc = M1_MAXRET + 1`, whatever `C` is. Its result is the **fraction of
  distinct indices covered**, `M1_MAXRET / 65,532`, not an allocation count;
- `release` needs `M1_MAXRET >= 10C` to reach phase 2 at all. At `C` = 65,532 that is 655,320
  capabilities — **10.5 MB**, since a `void *` here is a 16-byte capability — against a ~2 MB region.
  **`m1-series.txt` therefore cannot be satisfied by any image on this platform**, and is kept only
  because it is what boot 1 was scheduled from. Use `--cap M1_MAXRET/10` for a release arm.

A smoke run at `C` = 64 hides all of this: 4,096 covers 10C with margin there and every arm reads
`stop=target`. That is why the emulator pass did not catch it.

## The allocation-order control

`domdata-budget.py`'s model, verified against two built images, is `declared = 278,064 + 16 x M1_MAXRET`
for this harness, and the module rounds `code_len + 8,192 + declared` up to a power-of-two page count.
So `M1_MAXRET` moves two things at once: the **storage carve** the entry glue walks, and the
**allocation order** of the region the module hands the domain. 4,096 lands at order 7 and 65,532 at
order 9, with order 8 never built.

`m1-order9-ctl.txt` separates them. Its image keeps `M1_MAXRET` = 4,096 — the pilot's storage carve,
76,832 bytes, byte-identical arithmetic — and reaches order 9 by padding `R1_STACK` to 1 MiB alone.
Boot 1 wedged at `mcause` 28 (`RISCV_EXCP_CAP_OOB`) with `mepc` at `_start+0xec`, the entry glue's
byte-wise zeroing store, so the region and the carve are both live suspects and this image tells them
apart: wedging indicts the region size, completing indicts the carve.

## The two release arms, and why there are two

The lead ruled on 2026-09-18 that both be run, because they answer different questions.

`m1-maxret43296.txt` keeps the **approved algorithm** and reduces the capacity until phase 2 is
reachable: `C` = 4,329, so 10C = 43,290 fits inside the 43,296 buffer with six to spare. It is a real
release arm at **6.6 % of the protocol's production capacity**, and that reduction is the thing to state
whenever the number is quoted.

`m1-relbuf43296.txt` keeps `C` at the **production capacity** and moves the trigger instead:
`M1_RELEASE_AT_BUFFER=1` starts phase 2 when the retained buffer fills rather than at 10C. It is *not*
the protocol's release arm and a transcript from it must never be reported as one — which is why the
define lowers `target` rather than hiding the change, so the start line reports the trigger it used and
carries `rel_at_buffer=1`.

The two images differ by **exactly** that define: same `M1_MAXRET`, same allocation order 8, storage
704,032 against 704,048 (the sixteen bytes are the extra start-line field). And the define is opt-in and
free: with it present but off, `M1_MAXRET=4096` rebuilds to `1b7a04fe237e1580` byte for byte — the image
currently on the board — and `M1_MAXRET=43296` is unchanged. That was checked rather than assumed, and
the check earned itself: the first version of the change put the extra term in the loop's phase-2
condition, where it folds away at compile time, and the default image still moved (`598dce77`).
