# Result: every `resolve()` sets A **and** D and fences the address — so "the A/D loop" narrows to three disjuncts

Traced in the board kernel's own source (6.4.14), after r8's "the A/D candidate is refuted" was
withdrawn and the candidate came back onto the table. Read-only; no run.

## The chain, quoted

`pin_user_pages_remote` passes `FOLL_TOUCH` **unconditionally** — `mm/gup.c:3148`,
`FOLL_PIN | FOLL_TOUCH | FOLL_REMOTE`. Under that flag, `follow_page_pte` (`mm/gup.c:478-488`):

```c
if (flags & FOLL_TOUCH) {
        if (flags & FOLL_WRITE)
                entry = pte_mkdirty(entry);
        entry = pte_mkyoung(entry);
        if (!pte_same(*pte, entry)) {
                set_pte_at(vma->vm_mm, address, pte, entry);
                update_mmu_cache(vma, address, pte);
        }
}
```

The module always passes `FOLL_WRITE`, so **both A and D are set, on a present PTE, not only on a
fault.**

And `update_mmu_cache` on riscv is not a no-op — `arch/riscv/include/asm/pgtable.h:450-460` calls
`local_flush_tlb_page(address)`, which is `sfence.vma %0` (`asm/tlbflush.h:23-26`).

**The one way that could have been patched away was checked and is not.** `ALT_FLUSH_TLB_PAGE` is
an `ALTERNATIVE`, but it only swaps the address form `sfence.vma %0` for the whole-ASID form
`sfence.vma` under the SiFive CIP-1200 errata — **never for a nop** — and this kernel enables no
`CONFIG_ERRATA*` at all. `vmlinux` carries 42 `sfence.vma`.

So on every `resolve()` where the PTE actually changes, the kernel sets A and D and fences that
address, on the hart that then issues the CSRUNV.

## A hypothesis of mine, withdrawn before it reached a record

I was about to report that GUP takes a reference on an already-present PTE *without* touching A,
which would have made a plain A/D loop viable and would have contradicted a peer's populate
argument. The source refutes it: `FOLL_TOUCH` is unconditional in that entry point. Withdrawn here
rather than in a correction later.

## What survives, and it should be pre-registered as three disjuncts, not as "the A/D loop"

1. the RTL walker does not honour that `sfence.vma` for the address, or caches PTEs somewhere the
   fence does not reach;
2. the walker faults on something **other** than A/D, so setting them changes nothing;
3. the faulting address is not the address GUP touched.

**The `pte_same` guard is what makes disjunct 1 live.** If A and D are *already* set, there is no
`set_pte_at` and therefore **no sfence at all** — so a walker unhappy for some other reason gets no
fence either. That is exactly the state `resolve()`'s already-pinned path sits in, and it is what
the `-ELOOP` bound was written on top of, which makes the bound's own line the discriminator
between disjunct 2 and the other two.

## One fact for the RTL side

**The module issues no `sfence.vma` and no `flush_tlb_*` of its own, anywhere.** Checked in the
source and in the built object: 0 `sfence` instructions, and the 6 `fence` instructions are the 4
gated `fence rw,rw` plus 2 pre-existing. It relies entirely on GUP's. Disjunct 1 is therefore an
RTL question and not a module one.

## Why this is filed as a narrowing and not a refutation

It does not refute A/D. It converts a single broad hypothesis into three testable ones and
identifies which instrument separates them. Disjunct 2 is readable from the `-ELOOP` line, which
only became readable at all once the console-level retraction was understood — see
`RETRACTION-pr-warn-does-not-reach-the-board-console.md`.
