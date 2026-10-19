# The cursor-window gate on v6: result, read against PREREGISTRATION.md (0a503d4817d4)

Ten boots on qkit7-v6wc, nested-churn 3,000 rounds, nc-old d7b0ad9c524b2c15 and nc-new 247413702490a65f alternating.
Result lines only; the consoles stay outside the repo.

| | old (5 boots) | new (5 boots) |
|---|---|---|
| controls: counter on, window file present, NC_DONE rounds=3000, exit 0 | 5/5 | 5/5 |
| escape_out per boot | 6,071 in every boot | 0 in every boot |
| op_out per boot | 0 | 0 |
| distinct escape sites | 1: VA 0x238ac in `alloc_slot`, 30,355 in total | none |
| escape_first (the first escaped value: base:end@cursor) | a 16 KiB group, cursor 0x0, every boot | none |

## Against the predictions

1. **Controls: hold.** No boot is void.
2. **Old, an escape in a census window: DOES NOT HOLD.** No escape fell in either `bind_slots` loop or at either
   `backend_pointer` pair. By the pre-registration's own rule, the run therefore cannot show the LIVE window being
   hit: it is not a pass for the fix on that question.
3. **New, no site in the changed functions: holds.** New has no site anywhere.
4. **No op site in those functions, on either image: holds.** op_out is 0 on both.
5. **Sites elsewhere: one, on old only.** It is analysed below. It was not predicted.

## What old's escapes are (read after the run, not pre-registered)

- 0x238ac is `lbu a3, 0x8(a0)` in musl's `alloc_slot`. It is the first touch of the group that the call at 0x23800
  returned, and that call goes to 0x1f3cc, `__capstone_malloc_group_map`. Between that call and 0x238ac nothing
  writes `a6`.
- In `group_map`, `a6` holds `group - base` (0x1f5a8/0x1f5ac). That is the census's negated own cursor, hoisted
  before the slot loop. Its cursor is 0 and its bounds are the group's: exactly what `escape_first` reads.
- So each old escape is the census value left DEAD in a caller-saved register. It is counted when the first touch
  of a new group faults, once per `group_map` call, which is why the count is the same in every boot. The counter
  counts dead temporaries by design (the pre-registration's upper-bound note).
- New's 0 means the fix removes the value itself, not only its live use.

What this run does not show: a preemption or fault inside the live window, where the compact profile's save would
change the bounds the loop goes on to use. It did not occur in 5 boots of this workload.
