# Result (r13): S-24 resolved — the PTE reads `D = 0` on silicon, and the fix clears the loop

Two-sided on one boot, scored against `PREREGISTRATION-r13-which-field-decides.md` (`d9750cf1d817`)
and the prediction recorded in
`RETRACTION-gup-does-not-set-A-or-D-on-an-anon-page.md`. Module `a56fc415ad7133aa`
(`CV_DIAG_TAG resolve-sites-10`); lines quoted from the board lane's transcript.

## The reading

**`resolve_dirty=N`** — `VM_EXIT:vm-dirty0:139`:

```
capstone_vm: fault at 0x3fa4006000 cause 15 resolved 65 times with no change; ...
    pte rc=0 val=0x20331857 present=1 write=1 young=1 dirty=0 pfn=0x80cc6
```

**`resolve_dirty=Y`** (read-back `Y`; `threads2`'s must-fire at 176.57 s) —
`VM_EXIT:vm-dirty1:1` with `VM_CONTRACT_FAIL:execute through mapping PCC`.

## The pre-registered prediction held exactly

`val = 0x20331857`, decoded from the RISC-V PTE layout **by computation, not by eye**:

| bit | V | R | W | X | U | G | A | D |
|---|---|---|---|---|---|---|---|---|
| value | 1 | 1 | 1 | 0 | 1 | 0 | 1 | **0** |

and `val >> 10 = 0x80cc6`, which **agrees with the `pfn` the same line printed** — an internal
consistency check on the instrument, not merely on the subject.

The pre-registration said: *board `D = 0`, with V, R, W and A set*; and *a print showing `D = 1`
refutes the whole account including the fix's premise*. **`D = 0`.** The account is confirmed on
silicon rather than inferred from kernel source.

## S-24, end to end

1. the context stores to a page that is **present, writable, young — and not dirty**;
2. the RTL raises a store page fault and installs no TLB entry (`cva6_ptw.sv:477-481`: a store
   needs W **and** D), and the walker never writes A or D itself;
3. `resolve()` runs; GUP returns the already-pinned page and **leaves D clear**, because
   `follow_page_pte`'s `FOLL_TOUCH` block marks only the `struct page`;
4. `resolve()` reports success having changed nothing the walker cares about;
5. the store faults again — 65 times, then the bound.

**Not a walker defect.** Faulting on a clear D is permitted; setting D is the kernel's job, and
`fixup_user_fault`'s own docstring says so: it *"will in turn perform all the necessary SW fixup of
the dirty and young bits in the PTE, while get_user_pages() only guarantees to update these in the
struct page… on such architectures, gup() will not be enough to make a subsequent access succeed"*
(`mm/gup.c:1251-1258`). vp2 without Svadu is exactly such an architecture.

## The fix works, and what its arm does and does not show

With `resolve_dirty=Y` the loop is **gone**: `vm` no longer returns 139 but **exit 1**, at
`VM_CONTRACT_FAIL:execute through mapping PCC` — QEMU's own status and QEMU's own failure point. So
the step now fails where it fails on QEMU, for a different and pre-existing reason, which is the
best available evidence that the S-24 loop specifically was removed rather than masked.

**No new bound line appears after the flip, and that is correct, not a gap.** If the fix works the
bound never trips, so `resolve_pte` prints nothing in that arm — pre-registered as expected
behaviour. `dmesg` is cumulative, so the only bound lines in that window are the 166 s ones from
the `dirty=N` arm; they must not be read as the flipped arm still looping. `threads2`'s refusal line
at 176.57 s is what proves the post-flip window reads at all.

## Controls

the banner (console, once, `pr_emerg`); `thread create refused` in the `vm` window **and** again
from `threads2` after the flips, the second being the post-flip window's own must-fire; and the
sysfs read-back `Y` confirming the flip landed, since nothing the module prints reports the
parameter's value at an entry.

## What this closes and what it does not

**Closes:** S-24's mechanism, and the `!same` attribution r12 refuted. The three disjuncts are
resolved to one: the walker faults on something other than A — namely D — and nothing was wrong
with the fence or the address.

**Does not close:** S-25/S-26. The sparse wedge and the first-virtual-context spin are separate,
and `resolve_dirty` is scoped to `cause == 15` on an already-present page, so it cannot bear on
either.
