# `svadu=false` on the prediction QEMU — the first prediction on that setting, and my synthesis refuted

Two results. The setting is free, and the mechanism I proposed is wrong.

## What was pre-registered

After the kernel-version hypothesis was refuted (`can_follow_write_pte` byte-identical between the
QEMU guest's 6.1.0 and the board's 6.4.14; the only FOLL_FORCE difference in the whole write gate is
a hugetlb block that cannot apply to `MAP_PRIVATE|MAP_ANONYMOUS`), I offered a synthesis: the store
at `p[8192]` faults only on silicon because QEMU's `svadu` sets A and D in hardware, while `!same`
explains why that fault is terminal. Both branches were written down before the run
(`<svadu-off>/PREDICT.txt`):

* **synthesis holds** — `vm` exits 139 with cause 15 at arena base + 8192, as r8 read on silicon;
* **synthesis fails** — `vm` exits 1 at its own `require(tid > 0)`, i.e. the store passed and the
  step only met the one-slot thread refusal.

## The reading: the second branch

Full suite, prediction QEMU with `svadu=false`, module `cb4a8d758fd55d41`, v9 images,
`supervisor_slots` at its default 1:

```
34 / 41       false: threads, vm, heap-threads, threads_output, vm_contract,
                     heap_threads, shared_heap_pages_released
VM_EXIT:vm:1                     <- the one-slot thread refusal, NOT a store fault
cause=15 anywhere: 0             cause=25 anywhere: 0
wrapper fired: 1
```

**`svadu=false` does not make the store fault.** The step vector is identical to the `svadu=true`
run — same 22 `VM_EXIT` lines, same codes, same seven false checks. So `svadu` makes no difference
to this suite at all, and the mechanism I proposed for r8's terminal cause-15 store is **refuted**.

**`wrapper fired: 1` is why that zero means something.** The setting is injected by a wrapper that
appends `svadu=false` to the virtual-profile `-cpu` string, and the wrapper logs each time it
matches. Without that count a "no difference" reading would be void — indistinguishable from a
wrapper whose pattern never matched. `svadu` defaults to **true** (`cpu.c:1922`,
`DEFINE_PROP_BOOL("svadu", RISCVCPU, cfg.ext_svadu, true)`), so every prediction before this one ran
with the hardware setting A and D.

## What this settles, and what it does not

* **Settled, and useful for the sweep:** moving the prediction QEMU to `svadu=false` costs nothing
  in the Tier 1/2 readings. This is the first prediction on the setting the Tier 3 sweep will use,
  and it is identical to the old one, so the move is free rather than a re-baselining exercise.
* **Settled:** `svadu` is not the reason r8's store faulted on silicon and passes on QEMU.
* **NOT settled:** what is. Two hypotheses have now been refuted — the A/D loop (by r8's own absence
  of any repeat) and the kernel-version difference (by the source) — and this third, mine, by
  measurement. The `!same` / COW reading of r8's terminal fault still stands on its own evidence: a
  cause-15 store fault terminal on first occurrence can only leave `resolve()` through the
  `!same` → `-EACCES` branch, since the repeat counter below it never fired.
* The decider is the first-fault print, which must dump `a->pages[index]` and the newly pinned page
  so `same` is read rather than inferred.

`svadu=false` remains right in principle — a QEMU that sets A/D in hardware cannot reproduce a class
of silicon faults, the same faithfulness argument as C-27, Q-21 and Q-30 — but principle is all it
is here. It is not the r8 reading.
