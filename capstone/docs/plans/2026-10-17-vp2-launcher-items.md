# vp2 launcher work items, handed over from the RTL lane

Three items the virtual launcher and module need for vp2's silicon profile, relayed by the RTL
lane from item D's design (`dev` `a838f0dbdeea`). **None is started**, and all three sit behind the
option (iii) allocation work in `2026-10-16-exact-bounds-option-iii-cost.md`. Recorded here so they
survive a lost session rather than living in message history.

Each is the RTL lane's design, not this lane's finding. Where a claim was checkable against the
tree it is checked below and the check is named; the rest is carried over and should be re-confirmed
against the RTL before any of it is implemented.

## 1. A software collector in the module

On silicon `CSRUNV`'s action 3 (collect) is **illegal** and traps with cause 2. The launcher takes
that trap as "collect in software", and the module then does in C what QEMU's collection does:

- walk the listed pages;
- read each tagged granule's node id;
- clear the tag of a dead node — the tag region is physical and writable by ordinary stores;
- then the saved contexts;
- then strobe `csupctl` bit 2 (`revnode_ns_flush`) before resuming.

One adapter serves both profiles, keyed on **which trap it gets** rather than on a build flag.

## 2. The frame is the context storage on silicon, and it is 4 KiB

**Corrected from an earlier 2 KiB figure.** It is a page, because the caller's snapshot at
+1024..+1967 leaves no room below 2 KiB for the caller's FP state.

| offset | contents |
|---|---|
| +64..+575 | the paused context's registers |
| +224 | the reply slot — **the only slot software writes while a context is paused** |
| +656..+919 | the paused context's FP state |
| +928.. | its CSRs |
| +1024..+1967 | the caller's snapshot |
| +2048..+2311 | the caller's FP state, 32 x 8 bytes plus `fcsr` |

**Checked against the tree, and it holds.** `capstone_vm.c:918` already validates a frame as
`find_arena(c, r.frame, PAGE_SIZE)`, i.e. a page, and **no 2048-byte assertion exists anywhere** in
`vm-abi.h` or `wire.h` — so the 4 KiB correction agrees with the code rather than requiring a
change to it. Also checked for collision: the module writes `frame[80]` and `frame[81]` (the
collect chain, `capstone_vm.c:351-352`), which at 8 bytes per slot is +640 and +648 — between the
register block's end at +575 and the FP block's start at +656, so it does not overlap either.

To do: confirm the ABI v6 headers carry an explicit 4096 size assertion rather than relying on
`PAGE_SIZE` at the one call site, and audit that nothing in the launcher writes any slot other than
+224 after entry.

## 3. A bad-frame `CSRUNV` is refused as a terminal event, not a synchronous trap

On silicon, a `CSRUNV` whose frame is bad — `satp` not Sv39, `srevroot` 0, an untagged PCC slot, or
options above 3 — is refused as a **terminal event** `{kind 2, cause 29, pc = the entry PCC's
cursor, addr 0}` with `rd = 2` and **nothing of the context written**. QEMU raises a synchronous 29
for the same input.

So the launcher must treat that event shape as the refusal, keyed on the profile, and any runtime
test that expects the trap — `probe.S`'s bad-frame cases, if it has any — is re-scoped to accept
either reading rather than being made profile-specific.

## Note on sequencing

These are launcher and module work, so they land after the two allocation fixes currently blocked:
the `add()` relaxation (with the project lead, because it changes what the silicon enforces) and the
granule rounding of `realloc.c`'s `needed`. Item 1 in particular touches the same module as the
`add()` change, so it should not be interleaved with it.
