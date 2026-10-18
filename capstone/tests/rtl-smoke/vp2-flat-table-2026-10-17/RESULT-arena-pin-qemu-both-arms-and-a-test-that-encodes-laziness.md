# Result: the arena pin cuts faults 244 → 176 with no exit moved — and trips a test that encodes laziness

Both arms on the reference pin (`b290f565`, qemu `7a4f3a64776f0c88`, `svadu=false` via the knob;
the resolved `-cpu` string is in each run's `cpu.txt`). Module `3005ed52e860c7af` (board) /
`5bfc38f808e1effd` (QEMU), tag `arena-pin-1`, built from one invocation over one tree.

| | `arena_pin` off | `arena_pin=1` |
|---|---|---|
| pass | **34 / 41** | **33 / 41** |
| false | the usual **seven** | the same seven **+ `linux_demand_faults`** = **eight** |
| `faults` | **244** | **176** |
| all 22 `VM_EXIT` lines | — | **identical to the off arm** |

## The pin works, and the measure that shows it is the fault count

`faults` falls **244 → 176**, a 28% drop, with **no exit changed**. That is exactly the intended
effect: fewer demand faults, same verdicts. The drop is the result; the pass count is not.

## The eighth false check is the pin working, not a defect

Read out of `run.py` rather than inferred:

```python
demand_stats = [s for s in base_stats if s.get('faults', 0) >= 384]
tests['linux_demand_faults'] = len(stats) == expected_stats and len(demand_stats) == 3
```

The check requires **exactly three steps to exceed 384 faults**. Reducing demand faults is the
pin's whole purpose, so fewer steps clear the threshold and the check fails. **It is a test
encoding the lazy behaviour the pin removes.**

Consequences, both of which matter more than the number:

* **A board run with `arena_pin=1` must score against the EIGHT**, not the seven. Scoring it
  against the seven reports a false failure on a working pin — which is how a correct change gets
  reverted.
* The check is not wrong and should not be edited to pass. Under the pin its premise simply does
  not hold, so it is **expected-false by construction** and belongs in the pre-registration as
  such.

## Why `resolve_dirty`'s flip does not move the QEMU prediction

`vm` reads **1** on QEMU in every configuration measured: `arena_pin` off, `arena_pin=1`, and the
earlier arms that had `resolve_dirty` on by default. QEMU does not reproduce S-24 at all — with the
repeat bound forced to 1 the suite produced zero `-ELOOP` lines — so the Y-for-`vm`-alone flip
changes nothing here. On silicon it is the difference between 139 and 1.

## The implementation, and what its name overstates

A one-line widening of `pin_ahead`'s limit: `(arena_pin || count + 1 < pin_ahead_pages)`. That loop
is already batched, deadline-aware through `cv_step_stop`, validates anon and mapcount per page,
NULLs what it could not take, and stops at the VMA edge and the first already-pinned page — so it
is idempotent across faults and needs no flag on `struct arena`, which also avoids touching the
positional struct literal in `add()` where a new member silently shifts `lazy`.

**It pins FORWARD from the faulting page**, so it is "the rest of the arena from here", not the
whole arena. That covers S-25's case because an arena's first access is `enframe`'s plain byte read
of the slot prefix near the arena's start; it does not cover a first access landing late in an
arena. And it cannot help the sparse steps at all — their first store to a slot page can itself be
a capability store, which no pin on the fault path gets ahead of. They stay NOT RUN under S-25.

**Default off**, and silicon has now confirmed the inert case: Tier 1 passes with the pin code
present and off (`vm-contract` identity, banner `arena-pin-1`, fault line equal to boot 4c's by
script). After the Q-26 regression — where a default change broke every virtual context while QEMU
read 34/41 — shipping inert and proving the off case first is the sequence that earns the on case.
