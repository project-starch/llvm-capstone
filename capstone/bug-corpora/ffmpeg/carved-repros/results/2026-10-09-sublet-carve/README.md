# carved-repros: the Sublet port of the carve (`sublet-carve`), 2026-10-09

**Question.** Column 3 of the per-bug table for the carved corpus: if the carving routine is ported
to Sublet -- the block lent LINEAR by the Sublet heap and split into one Sublet region per carve,
each issued as a bounded alias -- which crossings are caught?

**Build.** `-DFFC_SUBLET_CARVE` with the Sublet-heap SDK (`shared/corpus.h`, `shared/driver.c`):
`ffc_block_alloc` takes the block with `__capstone_sublet_malloc_linear`, `ffc_carve` cuts it on
16-byte boundaries with `sublet_carve`, `sublet_take` issues each region's alias and
`__builtin_capstone_cap_shrink` narrows it to exactly the carve; `ffc_recarve` gives a region back
(one revoke) and `free` is the heap's single revoke of the block. Run by
`tools/run-capstone-domain.py --arm sublet-carve`.

**Result: 12 of 12 CAUGHT at the labelled probe, every fixed arm FIXED, as pre-registered
(40175e883ca3).** Cause 7 on the stores and cause 5 on the loads: the catch is the BOUND the port
puts on each region, the same mechanism as `capstone-carve-bounds` -- no buggy sequence in this
corpus re-carves, so none is a revocation catch.

**Controls, same boot.** The carve control (99, one byte past a 16-byte carve) faults with cause 7
at the write probe; the re-carve control (98: a region carved, written, carved again, and its OLD
alias written) faults with cause 24 -- the revoked alias -- at the write probe, and both fixed arms
complete. 98 is the only evidence here that the port's revocation works, because the cases never
re-carve.

**Granularity.** Sublet regions are whole 16-byte capabilities. A carve whose end is not on a
16-byte boundary takes the rest of the block as its region and the following carves share it: their
bounds stay exact (the alias is shrunk) but they are revoked together. That is case 4 (44-byte
slots in the buggy arm, 132-byte in the fixed one, which the fixed run's `region=[0,512)` lines
show); every other case's regions are their own (e.g. case 0: `[1152,1472)` and `[1472,1792)`).

**The first attempt produced no reading.** Before ed9f91e69c5d `ffc_carve` began with
`(char *)block + off`, and under the switch `block` is an untagged address: offsetting an untagged
capability faults on Capstone (cause 24). The carve control caught it on both arms and the runner
exited 75 without scoring a case.
