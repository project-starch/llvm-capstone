# Cursor-window fix (runtime lane 3e616209 / 6b1a5e8f): the provoked two-sided control, pre-registered 2026-10-19

**Mechanism** (read at capstone-qemu b290f5656600):
- a virtual-context escape stores x1..x31 with `cap_compress` (capstone_supervisor.c virtual_store_reg :750-768);
- a resume reloads them with `cap_uncompress`, restoring exact bounds only under `capstone_exact_bounds`
  (virtual_load_reg :726-744);
- CINCOFFSET itself keeps fat bounds in registers (op_helper.c :803-815).
So a capability live in a register with its cursor outside its representable window is corrupted only by a preemption
that lands while it is live, and only on the compact profile.

**SDKs:** t3-freeze-nb (runtime heap.c as built, v9-kit source) against a copy whose `libapplication-runtime.a`
differs ONLY in its `heap.c.obj` member, recompiled with the recorded command from the same source plus the two edits.
- Control: the unmodified source recompiled at the mirrored path equals the archived member in every section; the
  original recompile differed only in `.comment`.
- The fixed object differs only in group_map, group_nested, prepare_free, free, __libc_free and realloc.

**Test:** `nested-churn.c` (rounds of 96 mallocs across many small classes, then 96 frees), built by each SDK:
nc-old 1410d336706820b6, nc-new 65f52de2988462c5.

**Static:** `negcursor-census.py` (raw words, parse count). CINCOFFSET by the NEGATED OWN CURSOR:
- nc-old 2 (group_map, group_nested); nc-new 0;
- sql0's postgres.dom 64010720f1a6b3bc 2, including the faulting site 0x214e0.

**Runs:** one boot per kit, qkit7 (compact) and qkit7-eb (exact), steps alternating old/new with 5 runs each, the same
rounds, each step printing CAPSTONE_VM_STATS (steps = entries, a count of escapes).

**Predicted:**
- old on qkit7: at least one run killed by a refused SHRINK (`[CAPSTONE] SHRINK illegal operand value` and cause 29);
- old on qkit7-eb: 0;
- new on both: 0.

**Readings:** per arm, runs completed (NC_DONE), runs killed by SHRINK, other faults, and the steps counts.

If old on qkit7 reads 0, the control did not fire. The fix then rests on the source alone, and this record says so.
