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
| `m1-maxret43302.txt` | `pressure`, then `release` at `C` = 4,330 | `113aceea5a6070c0` (`M1_MAXRET`=43,302) | the largest buffer that stays at allocation order 8 |

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
