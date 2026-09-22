# The M1 exhaustion run is unreachable on this platform, and the leak coefficient stays bounded — settled at the desk, no board time spent

**2026-09-22.** The outgoing RTL lane's handover lists one item in flight for this lane: *"a dedicated
exhaustion run… they will produce a leak coefficient."* It cannot be produced for the realistic
workload, and the arithmetic that says so needs no boot. Recorded here so nobody schedules it.

## The finding

**Exhausting the 65,532-index pool with the M1 harness's own workload needs ≈ 1.07 × 10⁹ allocations
≈ 2.09 hours inside a single boot.** The driver's stage budget is 300 s, so it is over by **25×**, and
the pool does not reset within a boot — it resets *across* boots — so the work cannot be split.

## Why, from the data rather than from the model

`minted − revoked = **31**` on **all 322 snapshots** of the 09-18 no-pressure invocation, and 31 is
exactly the `fixture_nodes` that run declares. Verified by re-parsing
`experiments/results/M1/2026-09-18-reclaiming-series/raw/boot2-nopressure.r1-lines.txt`, not taken from
the bundle's summary. A *constant* difference means mints and revokes are balanced; combined with the
harness's handle reuse — `give(i)` frees `leaf[i]`'s node to a LIFO free list and the very next `take(i)`
pops the head, so it returns under the next generation (`r1_slots_pools.c:513-518`) — there is no
per-allocation handle accumulation. That leaves **retirement** as the only permanent consumer.

The same invocation reached **655,320 allocations** (`10C`) without exhausting. Had even one index been
consumed per allocation it would have exhausted ten times over.

Retirement consumes one index per 16,384 allocations regardless of rotation width, so 65,532 indices
need 65,532 × 16,384 allocations — the design's own figure
(`docs/plans/2026-09-16-revnode-reclamation-v2.md:133`). At the measured 175.3 cycles/allocation on an
unclosed 25 MHz clock that is 7,529 s.

## The retirement correction is real, negligible where we can measure, total where we cannot

The RTL lane warned twice that `65,532/A` is a *combined* rate needing
`handle-leak = (65,532 − retired)/A` with `retired ≈ A/16,384`, and that the correction "is not a small
correction". Computed rather than accepted — `M1_LIVE = 16`, so the rotation spreads sixteen ways and
first retirement is at 16,384 × 16 = **262,144** allocations:

| run | A | vs first retirement | retired | correction to `c` |
|---|---|---|---|---|
| pilot (`c < 0.3277`) | 200,000 | **below** | **0** | **0.00 %** |
| 09-18 boot (`c < 0.0498`) | 1,314,737 | past | 80 | **0.12 %** |
| exhaustion by retirement | 1.07 × 10⁹ | far past | 65,532 | **100 %** |

**The pilot's bound is uncontaminated** — the warning's own escape clause ("unless reuse spreads sixteen
ways") is satisfied. At 1.3 M allocations the correction is a twelfth of a percent. It becomes total
only in the regime that cannot be run. So the two mechanisms are separable wherever we can measure and
inseparable only where we cannot, which is the opposite of the way the caveat was framed.

Also: the bound in circulation is **stale**. `c < 0.3277` is the 200,000 pilot; the 09-18 boot's
1,314,737 allocations give **`c < 0.0498`**, 6.6× tighter.

## Why a synthetic leak-dominant arm does NOT rescue it

A workload that drops without reusing its handle leaks ≈1 node per allocation (handover bound 2: a
node is reclaimable only in the walk that invalidates it, so a DROP'd node leaks for the life of the
boot), giving `c ≈ 1` and exhaustion at ≈65,532 allocations — about half a second, and below the first
retirement, so the confound vanishes by construction.

**It measures a different question.** The leak fraction is a property of the workload's
allocate-to-free ratio, and the incoming RTL lane has already measured **0.529** for the
`r12-recl-freelist` shape (two nodes allocated per round, one freed; linear and flat to three digits
from 64 to 34,821 allocations) — with the explicit instruction **never to carry it across to another
workload**. A synthetic arm here would add a second such number under the same restriction. It would
confirm bound 2, which is worth having, but it would not be "the" leak coefficient and must not be
reported as one.

## What to say instead

**The boundary moves by the reciprocal of the workload's leak fraction; it is not removed.** The
fraction is **measured at 0.529 for one synthetic shape** (boundary ×1.89) and **bounded below 0.0498
for the realistic one**. Never write "the reclaimer lifts the 65,532 ceiling" without an
allocate-to-free ratio beside it — for a workload whose allocations are all handles the fraction is 1.0
and nothing moves at all, with the reclaimer working exactly as designed.

**No board time was spent reaching this conclusion, and none should be.**
