# Result: my QEMU control PASSED a module that breaks every virtual context on silicon

The instrument finding from boot 4, recorded because it bears on every future module validation
and not only on this regression.

## The numbers

| module | target | reading |
|---|---|---|
| `29c8e54e` (`resolve_dirty` default **ON**) | QEMU, pre-v7 pin (`809d1fc1`) | **34/41**, the usual seven false |
| the same | QEMU, reference pin (`b290f565`) | **34/41**, the usual seven false |
| the same | **silicon** | **every virtual context fails**, Tier 1 included |
| `a56fc415` (default **OFF**), otherwise identical | silicon | reads identity on the same rung |

`cause=24` lines: **1** in the QEMU serial, on **every step** on silicon.

The silicon signature: `cause 24` at code offset `0x1d94`, `address 0x2`, `entry=unknown`, before
HELLO, image-independent, with the banner printed and `resolve_dirty` reading back `Y`.

## Why the control could not have caught it

`resolve_dirty` calls `fixup_user_fault` only on the **already-pinned, cause-15** path — a store to
a page that is present, writable and **not dirty**. `RESULT-r13-S-24-resolved-...md` established
that QEMU does not put pages in that state on these paths: with the repeat bound forced to 1, the
suite produced **zero** `-ELOOP` lines, meaning no page ever faults twice there. So the call is
reached on QEMU (the mmap_lock BUG proved that much) but the *interesting* case — a present page
with D clear, where `handle_mm_fault` has real work to do — does not arise.

**So a 34/41 on QEMU is not evidence about a default that changes a page-fault path.** The control
was proven to fire for the things it does cover: it caught the `mmap_read_lock` precondition as a
kernel BUG, and it caught the Q-26 warning two-sidedly. It was simply blind to this, by
construction, and nothing in its output said so.

## The faulting instruction: content corruption, not a mapping fault

Offset `0x1d94` is `contract.dom` VA `0x11d94` = **`memset+0x2c`**, the instruction
`sb a1, 0x0(a3)`, and `a3` comes from `movc a3, a0` at `0x11d88` — memset's **destination**.
Verified against the image's own disassembly. The fault address is `0x2`.

So memset was called with a destination whose cursor is **2**. That is a garbage pointer read out
of memory before HELLO, not a fault on a mapping at the store itself. It is the reading the
mechanism hypothesis predicts: a domain that reads a pointer from a page whose backing changed
under it gets zero or stale bytes where a pointer should be.

## The reasoning error, which is the reusable part — corrected

My first account of this was wrong and the board lane's records refute it. I wrote that boot 4 was
the first time any path other than `vm` executed the call. It was not: in r13, after
`echo 1 > resolve_dirty`, `threads2`, `vm-dirty1`, `nested`, `nested-stale`, `heap-threads`,
`stale`, `bounds`, `retired`, `vm-ro-write`, `vm-none-read`, `vm-guard`, `vm-padding`, `preempt`
and the parallel pair **all** ran with `resolve_dirty=Y` and matched r12 in exits and in every
fault line, by script.

**What actually differs in boot 4 is WHEN.** There, `resolve_dirty` was on from the module's load,
so it was active for the **first** contexts — `vm-contract`'s `ok`, then everything. So the failure
needs the dirty path active *early*, before state that r13's first `dirty=N` contexts had already
established: shared runtime or image pages faulted in once, the zero page, an arena's first pin.
That is a much narrower and more testable claim than "any path", and it is the board lane's, not
mine.

**The habit this still earns, in its corrected form:** a parameter validated by flipping it
mid-suite has been validated *from the flip onwards*, on a machine whose early pages are already
populated. Promoting it to a load-time default is a new claim about the **cold** path, and needs a
run in which it is a default from load. The subset that went untested was not a set of steps; it
was a phase.

## What was NOT the cause, checked because it was the obvious suspicion

The source diff between the two modules touches **no struct field, frame word or ioctl shape**.
`diag_escapes` has no initialiser, so the escape print cannot execute — which is also why no S-26
lines appeared, an absence that was expected rather than a symptom. The Q-26 wait change leaves the
caller's outcome identical: the old `-EINTR`-on-non-zero and the new `-EINTR`-on-negative both fall
through to the same `return -EAGAIN`. And both arms of each pair are built by one invocation over
one tree, varying only `KERNEL_BUILD` and `CROSS_COMPILE`; verified in the objects, where both
carry the same diag tag and the same 20 format strings.

## The mechanism, as a hypothesis with its test named

`fixup_user_fault` calls `handle_mm_fault`, which may install a **different** page — a COW break or
a zero-page promotion — while the module still holds the old one pinned in `a->pages[index]`. The
domain would then run through a mapping whose backing the module never authorised: a capability
fault rather than a page fault, which fits `cause 24` before HELLO and fits its being
image-independent.

**Not established.** The test is a page-identity re-check after the fixup — the same comparison the
`remap` site already makes — and it is the next change, with its own control. Until then the
default is off and `vm` is read by flipping the parameter for that step alone, as r13 did.
