# Retraction: "every `resolve()` sets A and D and fences the address" — I quoted the wrong function

## What is withdrawn

`RESULT-gup-sets-ad-and-fences-so-the-ad-loop-narrows.md` (`15a2e6620f2e`) is **withdrawn in its
central claim**. It said `pin_user_pages_remote` sets the PTE's A and D bits and issues an
`sfence.vma` on every `resolve()`. **It does neither, for the pages this module actually handles.**

## The error

I grepped `pte_mkyoung` in `mm/gup.c`, took the first hit's surrounding block, and quoted it. That
block is inside **`follow_pfn_pte`** (`mm/gup.c:475`) — the **no-struct-page** path, whose own
closing comment reads *"Proper page table entry exists, but no corresponding struct page"* and
which returns `-EEXIST`. It is for `VM_PFNMAP`-style mappings and is **not** the path an anonymous
page takes.

Anonymous pages — all of this module's arenas — go through **`follow_page_pte`** (`mm/gup.c:533`),
whose `FOLL_TOUCH` block is at **624-633**:

```c
if (flags & FOLL_TOUCH) {
        if ((flags & FOLL_WRITE) &&
            !pte_dirty(pte) && !PageDirty(page))
                set_page_dirty(page);
        /*
         * pte_mkyoung() would be more correct here, but atomic care
         * is needed to avoid losing the dirty bit: it is easier to use
         * mark_page_accessed().
         */
        mark_page_accessed(page);
}
```

It marks the **`struct page`**. It never touches the PTE's A or D bits, and the comment says so
explicitly — `pte_mkyoung()` "would be more correct here" and is deliberately not used.

**The aggravating detail: that comment was in my own tool output.** My grep printed both blocks,
and I read the one I had already decided was the answer. This is the same shape as attributing a
mechanism from a grep hit without confirming the hit lies on the executed path — the rule exists
and I did not apply it across function boundaries.

## What this voids, and what survives

**Voided:**
- "every `resolve()` sets A and D" — false for anon pages;
- the "three disjuncts" framing built on it, and in particular the claim that the interesting
  question was *whether the walker sees* A and D;
- the rationale for `resolve_sfence` as the discriminator. The missing operation is not a fence.
  (The parameter is harmless and is kept, now as a control rather than the instrument.)

**Survives, because it was measured in the object rather than reasoned from source:**
- the module issues **no** `sfence.vma` and no `flush_tlb_*` of its own;
- no `fence` stands within 500 instructions of any of the four CSRUNV sites;
- the entry asm's `"memory"` clobber emits nothing and `local_irq_save` does not order memory.

## S-24's loop is now explained, end to end

With the correction, and with the RTL side read by the runtime lane (`bf3db3175`:
`cva6_ptw.sv:477-481` raises a store page fault and installs **no** TLB entry when
`!pte.w || !pte.d`; `cva6_mmu.sv:142/:602` the same on a DTLB hit; the walker never writes A or D,
and `sfence.vma` does flush the entry, so the TLB is not the loop):

1. the context stores to a present, writable page whose **D is clear**;
2. the hardware raises a store page fault and installs no TLB entry;
3. `resolve()` runs, GUP returns the already-pinned page and **leaves D clear**;
4. `resolve()` reports success having changed nothing the walker cares about;
5. the store faults again — "resolved 65 times with no change", which is exactly the `-ELOOP` line
   r12 read.

This is disjunct 2 of the withdrawn framing, and it is **not a walker defect**: faulting on a clear
D is permitted, and setting D is the kernel's job through its own write-fault path.

## The fix this implies, module-side and needing no bitstream

Resolve a kind-4 **store** fault on a present page through the kernel's own write-fault path
before pinning — `fixup_user_fault(mm, address, FAULT_FLAG_WRITE, NULL)`, which reaches
`handle_pte_fault`'s writable-pte case and performs `pte_mkdirty` + `ptep_set_access_flags` +
`update_mmu_cache`, i.e. sets D **and** issues the `sfence.vma`. A read fault keeps today's path.

`fixup_user_fault` is `EXPORT_SYMBOL_GPL` with the same signature in **both** kernels
(6.4.14 `mm/gup.c:1321`, 6.1.0 `mm/gup.c:1392`), so no version guard is needed and the fix is
testable on QEMU — provided QEMU faults on a clear D for a store at all, which is the next thing to
check and which, if it does not, makes the board the only reader.

**Pre-registered prediction for the next boot's PTE print at the loop: `D = 0`, with V, R, W and A
set.** That is the RTL lane's prediction and mine; a print showing `D = 1` refutes this whole
account.
