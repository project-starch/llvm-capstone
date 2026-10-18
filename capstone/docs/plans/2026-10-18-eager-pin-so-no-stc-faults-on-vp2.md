# Scoping: "no STC ever faults on translation" by eager pinning — cost, placement, and the question that decides it

**Scoped, not built, and explicitly not to be landed before the RTL arm's reading.** The candidate
it serves: on `bf3db3175` an STC whose translation faults retires with no exception and no kind-4
escape, leaving a ghost entry in the store buffer's commit queue. If that reads as predicted, the
only software workaround is that no STC ever faults on translation.

## 1. The change is ~2 lines, because the eager path already exists

This is the first thing to know, and it was not obvious from the proposal. `add()` already pins
every page of a grant eagerly — but **only for non-linear grants**:

```c
if (!r->linear) {
    n = pin_user_pages_fast(r->address, r->bytes / PAGE_SIZE, FOLL_WRITE, pages);
    if (n != r->bytes / PAGE_SIZE) goto unpin;
}
```

and the arena's lazy flag is simply the linear flag, through a **positional** struct literal:

```c
*a = (struct arena){r->address, r->bytes, ancestor, pages,
                   r->bytes / PAGE_SIZE, r->linear};
```

So `a->lazy == r->linear`, and `resolve()` only ever serves linear arenas
(`if (!a->pages || !a->lazy || ...) continue`). **The workaround is: make linear grants eager
too** — drop the `!r->linear` guard and pass `false` for the lazy field. Not a new subsystem.

*(Noted in passing, not part of this change: that positional literal is the shape the synthesis
rules warn about — a field added to `struct arena` silently shifts `lazy`. Worth converting to
designated initialisers the next time the struct is touched.)*

## 2. The cost, measured — and it is ~30x larger than the proposal assumed

The proposal estimated "the 6 MiB heaps of the kit apps". Measured from the `CAPSTONE_VM_ADD_OK`
rows of one full suite run on the `svadu=false` QEMU profile:

| | grants | total | largest single grant |
|---|---|---|---|
| `linear=0` — **already eager** today | 105 | 44.9 MiB | 4.0 MiB |
| `linear=1` — **lazy** today, would become eager | **242** | **201.7 MiB** | **64.0 MiB = 16,384 pages** |

And the lazy baseline this replaces, from the same run's `VM_STATS`: **peak 1,591 pinned pages
(≈6.2 MiB)** with 244 faults in the heaviest step. So the figure the proposal quoted is roughly
today's *peak residency*, not the cost of making every grant eager.

The ABI ceilings bound the worst case: `CV_MAX_REGION_BYTES = 256 MiB` per region
(**65,536 pages** eagerly pinned for one grant), `CV_MAX_BYTES = 1024 MiB` mapped per context, in a
**2 GiB** guest. Pinned pages are unswappable, so this is committed residency, not virtual.

One cost moves the other way and should be said: escapes. Today each first touch costs a kind-4
escape, an ioctl and a one-page GUP (`pin_ahead_pages = 1`). Eager replaces up to 16,384 of those
with a single bulk GUP. **Fault count per step should fall from 244 to ~0, and that is a speedup,
not a cost.** The cost is residency and grant-time latency, not throughput.

## 3. Where it goes

- `add()` — the guard above. The `pages` array is **already** allocated for both paths
  (`kvcalloc(r->bytes / PAGE_SIZE, ...)` is unconditional), so that allocation is not new cost.
- `remap_end()` — the mremap path, which must pin whatever the new extent adds.
- heap growth — same requirement as remap; growth that extends an arena must pin the extension
  before any STC can reach it.
- `resolve()` becomes dead for linear arenas. It should **not** be deleted: it is the only
  instrument that reads which refusal fires, and it still serves any arena that is lazy for another
  reason. Leaving it in place also keeps the two-sided comparison possible.

## 4. What the sparse steps then exercise — a semantic change, not just a cost

`sparse-churn` and `sparse-stale` exist to touch a **large arena sparsely**. Eagerly pinning the
arena makes that impossible: the pages are resident before the step begins, so the steps stop
measuring sparseness and become dense-arena churn tests. Their `fault_causes` expectation at
`run.py:222` would also change, since the first-touch faults disappear.

That is not an argument against the workaround — it is a statement that **those two steps stop
being evidence about the thing they were written for**, and the suite's expectations have to be
re-pre-registered rather than patched. The plain-touch steps are unchanged.

## 5. The question that actually decides this, and it is cheaper than the workaround

**Is the first touch of a fresh page in these arenas always a PLAIN access?**

The candidate only bites when an **STC** is the access that faults. Plain loads and stores escape
and resolve as they do today. So if the allocator always writes a granule header — a plain store —
before any capability store lands in a fresh page, then no STC ever faults on translation
*already*, and the workaround is unnecessary.

This is answerable from `mallocng`'s and the contract's source, with no board time and no RTL
reading, and it should be answered **before** 201.7 MiB of eager pinning is committed to. If the
answer is "not always", the counter-examples are also exactly the sites a narrower fix would
target.

**A narrower design, if the answer is "not always":** keep grants lazy, but make the first kind-4
fault in an arena pin the **whole arena** rather than `pin_ahead_pages`. Untouched arenas then cost
nothing, and after the first fault no further access in that arena can fault. It does not help if
the *first* access is itself an STC — which is the same question as above, so the two decisions
collapse into one.

## 6. The two-sided check, with the pin as the single variable

On the `svadu=false` QEMU profile, module as the only difference:

- **Predicted if the pin works as intended:** capability-store escapes drop to **zero**; total
  faults per step fall from 244 to ~0; `peak_pages` rises from 1,591 to the arenas' full extent;
  and **every verdict is unchanged at 34/41 with the same seven false checks** — that last is the
  gate, because a verdict change means the pin broke something rather than fixed something.
- **The control that makes the zero meaningful:** the fault counter must be shown to be non-zero on
  the unmodified module in the same configuration, which the run above already provides (244).
  Without it, "escapes dropped to zero" is indistinguishable from a counter that stopped counting.
- `sparse-churn`/`sparse-stale` are expected to change and must be scored separately, per §4.

## 7. Not to be landed yet

Per the requesting lane: do not land before the arm's reading, and note that r12's fence result is
*predicted to be "not the fence"* under this candidate — so a null fence result is not evidence
against the ghost-entry story.
