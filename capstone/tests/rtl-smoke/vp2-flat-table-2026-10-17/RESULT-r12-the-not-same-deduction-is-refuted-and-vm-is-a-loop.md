# Result (r12): none of the five sites fired, the `-ELOOP` line did — `!same` is refuted and `vm` is a loop

Scored against `PREREGISTRATION-r13-which-field-decides.md` (`d9750cf1d817`) and
`PREREGISTRATION-resolve-refusal-sites.md`. Lines quoted from the board lane's per-rung transcript;
the module is `06d663f3`, `CV_DIAG_TAG resolve-sites-4`.

## The reading

```
[  130.198658] capstone_vm: loaded, diag resolve-sites-4, supervisor_slots=1
[  160.539996] capstone_vm: thread create refused, 1 supervisor slot in use
[  165.181657] capstone_vm: fault at 0x3faf4ba000 cause 15 resolved 65 times with no change; the page is pinned and the PTE is untouched by this module
[  165.184208] capstone_vm: fault at 0x3faf4ba000 cause 15 resolved 66 times with no change; ...
```

and `vm`'s launcher line: `cause=15 pc=0x3faf4124ac address=0x3faf4ba000` — arena base+0x2000, the
same code offset `0x124ac` as r8–r11. **No `resolve refused at` line in the `vm` window.**

## What it settles

* **The `!same` page-identity deduction is REFUTED.** It was the only surviving explanation and it
  was a *structural* deduction from the module's source, never a reading. None of the five terminal
  sites fired — not `remap`, not `anon`, not `vma`, not `pin`, not `arena`.
* **`vm`'s 139 is a resolve LOOP**, ended by the S-24 bound: the same address and cause repeated
  past 64 and the run was failed with `-ELOOP` instead of hanging. r7, which had no bound, sat for
  25 minutes on this with nothing to show.
* **Disjunct 3 of the A/D narrowing is refuted too**, by the field the pre-registration named for
  it: `rep` *advanced* to 65 at a **stable** address, so "the faulting address is not the address
  GUP touched" is out. Three disjuncts become two.
* **r8's "the A/D candidate is refuted"** — already withdrawn this morning on method, for resting
  on an absence the console level made unreadable — was wrong **in substance** as well.

## Why the instrument had to be five sites and not one

The board lane asked for a print on the `!same` path. Printing only that would have produced
**silence**, and silence would have read as "not `!same`" — when in fact *no site fired at all*,
which is a different and much more informative result. The five-site design is what made the
difference between "the deduction is unconfirmed" and "the deduction is refuted". The must-fire
`thread create refused` line is what makes the absence a reading rather than a void window.

## The limit held exactly as pre-registered

The pre-registration says this line **cannot separate disjunct 1 from disjunct 2** — a walker that
does not honour the per-address `sfence.vma`, versus a walker that faults on something other than
A/D. The reading is precisely that ambiguous case: `resolve()` succeeds, nothing the walker observes
changes, `pfn_old == pfn_new` so no `remap` line, and the only artifact is the `-ELOOP` line. So
r12 delivered its pre-registered **best** case — three to two — and not a decision.

## The next instrument follows from `pte_same`, and it is one instruction

`RESULT-gup-sets-ad-and-fences-so-the-ad-loop-narrows.md` established that GUP sets A and D and
issues `sfence.vma` — **but only when `!pte_same(*pte, entry)`**, i.e. only when the PTE actually
changed. On the second and later faults of this loop A and D are **already set**, so there is no
`set_pte_at` and therefore **no sfence at all**. Under disjunct 1 that is exactly the missing
operation.

So an explicit `sfence.vma` for the faulting address on the resolve path, after the pin, separates
them behaviourally:

* the loop **ends** → disjunct 1: the walker needed a fence it was not getting;
* the loop **persists** → disjunct 2: the walker is unhappy about something other than A/D, and the
  question leaves the module entirely.

That is a one-instruction change, it needs no PTE read — which stays declined — and as a parameter
it is two-sided inside one boot.

## Controls, all of which held

the banner (console, once); `thread create refused` inside the `vm` window, so the window reads;
and the two `vma` lines scored on the **relation** the pre-registration specified rather than on
literal hex — `need 0x2` against flags without `VM_WRITE`, `need 0x1` against flags without
`VM_READ`. A third `vma` line appeared for `vm-guard`, which the pre-registration did not name and
which is consistent with it.
