# Pre-registration for r13: which FIELD of the refusal line decides, and what it cannot decide

Written before r13 is baked. The point is to make one line **decide** rather than describe, and to
state in advance the pair of hypotheses it provably **cannot** separate — because an instrument
that is proven to fire and still under-determines is a documented way this project has produced
retractions.

Module: the budgeted/`diag_emerg` build, board ko
`83228b7d98ed49def2cacfa1e1d1cc37b2cd451806340a587ee772f04781aaa3`, `CV_DIAG_TAG resolve-sites-6`.
Reading route: `dmesg | grep -a capstone_vm:`, with `diag_emerg=1` available in the same boot if
the ring-buffer route fails.

## The line's fields

Common to all five sites: `site`, `addr`, `cause`, `need` (the required VMA bit), `pc`, and
`prev(addr, cause, rep)` — the previous fault on that thread and its repeat count.

Site-specific on `remap`: `pfn_old`, `pfn_new`, `mapcount`, `refs`, `vm_flags`.

## The decision table

| reading | conclusion | the field that decides |
|---|---|---|
| `remap`, `cause 15`, `pfn_old != pfn_new`, `prev(addr == addr, cause 13)` | **COW break on a read-then-written page** — the live hypothesis | `pfn_old != pfn_new` **together with** `prev cause 13` at the same address |
| several `remap` lines at **different** `addr` sharing one `pfn_old` | the shared **zero page** specifically, not generic COW | one `pfn_old` recurring across addresses |
| `remap` with `pfn_old == pfn_new` | **impossible by construction** — the line only prints when `!same`. If seen, the instrument is wrong and the boot is void | the equality itself |
| `anon`, `mapcount != 1` or `anon 0` | the page stopped being exclusively-mapped anon — a COW **variant**, and *not* the `remap` path the structural deduction named | `mapcount` |
| `vma`, `flags` lacking the `need` bit | the module cannot see the mapping's protection; none of the above applies | `need` against `flags` |
| `pin`, `pin_user_pages_remote=` ≤ 0 | GUP itself refused | the return value |
| **no site line**, `-ELOOP` line with `rep = 65` | **disjunct 1 or 2** of the A/D narrowing — see the limit below | the `-ELOOP` line's presence |
| **no site line**, repeated faults with `rep` staying `1` and `addr` varying | **disjunct 3**: the faulting address is not the address GUP touched | `rep` failing to advance while `addr` changes |
| banner present, nothing else at all | the store is not failing in `resolve()`; the `!same` attribution is **refuted** | the banner, which is what makes the absence readable |

`rep` behaves as the table assumes because it only advances when **both** `addr` and `cause`
repeat; any change resets it to 1. That is what makes "`rep` stuck at 1" a positive reading for
disjunct 3 rather than an absence.

## THE LIMIT, stated before the run

**This line cannot separate disjunct 1 from disjunct 2.**

- Disjunct 1: the walker does not honour the per-address `sfence.vma`, or caches PTEs beyond its
  reach.
- Disjunct 2: the walker faults on something other than A/D, so setting them changes nothing.

From the module's side both present **identically**: `resolve()` succeeds, changes nothing the
walker observes, and the same fault returns until the bound fires. `pfn_old == pfn_new` in both, so
no `remap` line prints in either, and the only artifact is the `-ELOOP` line.

Separating them needs a reading the module does not have: the PTE's own bits after the pin, or the
walker's view. A PTE read was considered and **declined** — it means adding a page-table walk on a
path that goes to silicon, against the rule that an observation-only change stays a strict
reduction, and
`RESULT-gup-sets-ad-and-fences-so-the-ad-loop-narrows.md` shows the kernel already sets A and D, so
the interesting question is not *whether* they are set but whether the walker sees them. **That is
an RTL-side question and r13 should not be expected to answer it.**

So r13's honest best case is: COW confirmed or refuted by `pfn_old`/`pfn_new`, disjunct 3 separated
by `rep`, and disjuncts 1–2 **collapsed into one surviving alternative** — which is progress from
three to two, not to one, and should be scored that way.

## Predicted readings, so a pass cannot be read into a null

If COW is true: one `remap` line at `vm`'s faulting address, `cause 15`, `pfn_old != pfn_new`,
`vm_flags` carrying `VM_WRITE`, `prev` naming the same address with `cause 13`.

If disjuncts 1–2: **no** site line, one `-ELOOP` line, `rep = 65`, at `arena base+0x2000` with
`pc ...124ac` — the launcher line r8–r11 have all shown.

If the module is not where the failure is: banner only. **This is the outcome that redirects the
investigation**, and the banner at `pr_emerg` is what distinguishes it from a filtered print.

## Controls that must hold or the boot is void

1. the banner, at `pr_emerg`, independent of the console level;
2. `thread create refused` from the `threads` step, now on a deterministic budget rather than a
   time-based limiter — its first four occurrences always print, so its absence means it did not
   happen;
3. the `vma` lines for `vm-ro-write` and `vm-none-read`, scored on the **relation** (`need 0x2`
   against `flags` without `VM_WRITE`; `need 0x1` against `flags` without `VM_READ`) and never on
   literal hex, which differs between the 6.1.0 guest these were measured on and the board's
   6.4.14.

The refusal **count** is not a control: two identical QEMU runs produced 6 and 5 lines, because the
launcher's idempotent `CV_RESOLVE` repeats a refusal and one case did not repeat.
