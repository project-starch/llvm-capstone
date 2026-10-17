# Result: no fence stands between the module's stores and the domain entry, and adding one is free

Two findings, one measured in the built object and one in a QEMU control.

## 1. The resume and entry paths carry no fence — measured, not inferred

`run()` — the `.insn r 0x5b, 1, 0x24` domain entry (CSRUNV) — inlines to **four** sites in
`capstone_vm.ko`. **None has a `fence` within the 500 instructions preceding it.** The whole
module contains **2 fence instructions in 4323**, neither on this path.

Two things that look like barriers here and are not:

- the entry asm's `"memory"` clobber is a **compiler** barrier and emits no instruction;
- `local_irq_save` ahead of each entry compiles to `csrrc a4,sstatus,2`, which does not order
  memory on RISC-V.

`mmap_read_unlock`'s release semantics were checked as the one real counter-candidate and emit no
fence within 500 instructions of any entry, so they do not cover this either.

On the **entry** path the last frame store sits two instructions ahead of the entry:

```
sd  a5,8(a0)      frame[1] = virt_to_phys(c->table)
sd  s11,56(a5)    frame[7] = 3
ld  a4,40(s2)
CSRUNV
```

**Correcting the framing this came from:** on the **resume** path (a kind-4 page-fault escape back
to action 0) the module stores **nothing to the frame at all**. The event words at `+16..+40` are
`frame[2..5]` and are *read* after the loop, never written; the reply at `+224` is
`frame[28]/[29]` and is written only on an action-2 reply, before the loop. `resolve()` writes
only `a->pages[index]` and two statistics counters — kernel-side memory the hardware walk never
reads. The page-table stores on that path come from inside `pin_user_pages_remote()`, which
populates and breaks COW; the module touches no PTE anywhere.

So a fence means different things on the two paths — frame visibility on entry, GUP's page-table
visibility on resume — and the two should not be conflated when the result is read.

The entry-path offsets are `0`, `8` and `56` (`frame[0]`, `frame[1]`, `frame[7]`), not the
`+16..+40` / `+224` the question was framed around.

## 2. `fence_before_entry`: one module, two arms, one boot — and free when off

A `module_param(..., bool, 0644)` rather than a build flag, because the board kernel cannot unload
this module: a build flag would have cost a second boot for the second arm. `run()` reads the bool
on every entry, so a write to
`/sys/module/capstone_vm/parameters/fence_before_entry` takes effect at the next one.

Verified in the object rather than trusted to the compiler or the source line:

- the `fence rw,rw` lands at **all four** CSRUNV sites, each gated by an `lbu` of the parameter and
  a `beqz` that skips it, one to two instructions before the entry;
- the module's fence count moves **2 → 6** — the four added, nothing displaced;
- decoding the `__param` section's `kernel_param` entries, **exactly one of eight** has
  `perm=0o644` and the other seven are `0o444`.

**QEMU control: `34 / 41`**, the same seven false checks, and the `EXITS`, `FALSE` and `rc` strings
**byte-identical** to both the print-only control and the `svadu=false` baseline (diffed as
strings, not read side by side). Banner present. So the parameter costs nothing when off.

Module hashes: board `f02ca82ea9b2dd61075460548e899a095a0d0545345bff89df16f5f7219b1284`
(`vermagic=6.4.14`, RVC 0, 6/6 diagnostic strings at KERN_WARNING, `CV_DIAG_TAG resolve-sites-3`).
The control ran against the 0444 build's QEMU arm; the board arm is identical to it instruction for
instruction and the QEMU arm differs only in two `__LINE__` immediates (`1186→1193`, `1369→1376`,
both +7 for seven added comment lines) feeding a kernel diagnostic.

## What this does NOT claim

It does not claim the race is real. It makes it refutable: if the wedge survives
`fence_before_entry=1`, the ordering candidate is **refuted** rather than left open, which is the
property worth having before a board session rather than after.

**A discrepancy to carry forward rather than smooth over:** the two controls produced **6 and 5**
refusal lines on identical configurations, all at site `vma` on the same deliberate-violation
cases. The lines come in pairs because the launcher's idempotent `CV_RESOLVE` repeats a refusal, so
an odd count means one case did not repeat. That is nondeterminism in the retry path, not in the
diagnostic — so the **site, flags and cause** are the reading, and the refusal **count** is not a
stable figure.
