# S-07 Explained: How a Capability Sporadically Comes Back Untagged

> **STATUS: root cause confirmed, and FIXED.** The fix is `5c5f4e3a7` — forbid granule
> co-residency in the write buffer (section 9). This document explains the defect; the
> engineering detail, the rejected alternatives and the acceptance criteria are in
> `rtl/ROOT-CAUSE-AND-FIX-OPTIONS.md`.

This document explains the S-07 defect intuitively, using diagrams and step-by-step examples. It focuses on the root cause confirmed in RTL and on silicon: the write buffer treats a capability entry as one word when detecting conflicts, even though it writes two words when drained.

## 1. Structure of One Granule

A capability occupies one 16-byte granule:

```text
                 One granule G, 16 bytes
        ┌────────────────────┬────────────────────┐
Address:│       G + 0        │       G + 8        │
        │    lower 8 bytes   │    upper 8 bytes   │
        ├────────────────────┼────────────────────┤
Data:   │       CURSOR       │      METADATA      │
        │       64 bits      │       64 bits      │
        └────────────────────┴────────────────────┘
                         │
                         ▼
                  one shared TAG
                 for all 16 bytes

               TAG = 1: this is a capability
               TAG = 0: this is ordinary data
```

The key point is that the two 64-bit halves do **not** have two separate tags. There is one shared tag bit for the entire 16-byte granule.

---

## 2. The Two Stores That Collide

The failing scenario contains two stores.

### 2.1 The Older Plain Store

```text
plain store G+8
```

It writes only the upper 8 bytes:

```text
        ┌────────────────────┬────────────────────┐
        │       G + 0        │       G + 8        │
        │                    │   ordinary data X  │
        └────────────────────┴────────────────────┘
                                      ▲
                                      │
                              only this word changes

                                  TAG becomes 0
```

A plain store into part of a capability granule must clear the capability tag, because the granule can no longer be treated as one intact capability.

### 2.2 The Younger Capability Store

```text
stc G
```

It writes the entire capability:

```text
        ┌────────────────────┬────────────────────┐
        │       G + 0        │       G + 8        │
        │       CURSOR       │      METADATA      │
        └────────────────────┴────────────────────┘
                  ▲                    ▲
                  └──── written by stc ┘

                         TAG = 1
```

In other words, `stc G` covers **both words**, even though its write-buffer entry is identified by the address `G+0`.

---

## 3. Where the Write Buffer Goes Wrong

The write buffer treats its entries as separate 64-bit words:

```text
Entry A:
    wtag = address of word G+8
    is_cap = 0
    data = X
    ctag = 0

Entry B:
    wtag = address of word G+0
    is_cap = 1
    data = capability
    ctag = 1
```

Visually:

```text
                   WRITE BUFFER

     ┌────────────────────────────────┐
 A   │ wtag = G+8                     │
     │ plain store                    │
     │ covers only G+8                │
     │ ctag = 0                       │
     └────────────────────────────────┘

     ┌────────────────────────────────┐
 B   │ wtag = G+0                     │
     │ capability store, is_cap = 1   │
     │ actually covers G+0 and G+8    │
     │ ctag = 1                       │
     └────────────────────────────────┘
```

The buffer searches for matching entries using `wtag`, which represents the address of a **64-bit word**:

```text
wtag(G+0) != wtag(G+8)
```

The buffer therefore concludes:

```text
"These are different entries. They do not overlap."
```

Physically, however, they do overlap:

```text
                       G+0                 G+8
                 ┌──────────────┬────────────────────┐
plain store A:   │              │████████████████████│
                 └──────────────┴────────────────────┘

cap store B:     │███████████████████████████████████│
                 └──────────────┴────────────────────┘

                                       ▲
                                       │
                         both entries write this word
```

The `wtag` comparison sees **two different words**, but it does not account for the fact that an entry with `is_cap = 1` spans both words of the granule.

That is the root defect.

---

## 4. Why the Result Is Sporadic

Entries are not necessarily drained from the write buffer in program order. A round-robin arbiter selects the drain order.

This creates two possible outcomes.

### 4.1 Outcome A: The Plain Store Drains Last

The program order is:

```text
1. plain store G+8
2. stc G
```

Logically, the younger `stc` should win, leaving a valid capability in memory.

The write buffer can instead drain the entries in this order:

```text
Actual drain order:

1. stc G
2. plain store G+8
```

Step by step:

```text
After stc G:

        ┌────────────────────┬────────────────────┐
        │       CURSOR       │      METADATA      │
        └────────────────────┴────────────────────┘
                         TAG = 1
```

Then the older plain store overwrites the upper half:

```text
After plain store G+8:

        ┌────────────────────┬────────────────────┐
        │       CURSOR       │   ordinary data X  │
        └────────────────────┴────────────────────┘
                         TAG = 0
```

Final result:

```text
The capability is lost.
An ldc later returns NOT_CAP.
```

This is the observed S-07 failure. The next instruction that requires a capability receives an untagged value and raises `mcause 25`, `UNEXPECTED_OPERAND`.

---

### 4.2 Outcome B: The Capability Store Drains Last

**This outcome needs the OPPOSITE program order.** Note the difference carefully — it is easy to
conflate the two, and they are different failures:

```text
Program order for 4.1:   plain store G+8  →  stc G       (younger stc should win)
Program order for 4.2:   stc G  →  plain store G+8       (younger plain store should win)
```

So here the program stores a capability and then deliberately overwrites part of it — a scrub:

```text
1. stc G
2. plain store G+8
```

Logically the younger plain store should win, leaving no capability. The write buffer can drain
in the wrong order instead:

```text
Actual drain order:

1. plain store G+8
2. stc G
```

Step by step:

```text
After plain store G+8:

        ┌────────────────────┬────────────────────┐
        │      old word      │   ordinary data X  │
        └────────────────────┴────────────────────┘
                         TAG = 0
```

Then the older `stc` overwrites the entire granule, undoing it:

```text
After stc G:

        ┌────────────────────┬────────────────────┐
        │       CURSOR       │      METADATA      │
        └────────────────────┴────────────────────┘
                         TAG = 1
```

```text
S-07 does not occur: the capability remains intact
But: the program's own store is silently lost
```

**Why this matters more than it first appears.** Ask what that dropped store was *for*. A plain
store over a granule holding a capability is how software DESTROYS authority it no longer wants
to hold — `memset`, `explicit_bzero`, a free-list poison, clearing a slot before reuse. When it
is dropped, **the capability survives the operation intended to destroy it.** That is a failure
to revoke by overwrite: weaker than fabricating authority, stronger than losing a scalar.

It does **not** forge a capability over attacker-chosen data. Because the capability entry writes
*both* words, the granule it leaves behind is the original capability, not a tagged mixture — and
that was measured, with the corrupted-but-tagged outcome empty in both directions.

---

## 5. The Entire Defect in One Diagram

```text
THE COMMON CAUSE — true for either program order

┌───────────────────┐         ┌──────────────────────┐
│ WBUF entry A      │         │ WBUF entry B         │
│ wtag = G+8        │         │ wtag = G+0           │
│ is_cap = 0        │         │ is_cap = 1           │
│ writes G+8        │         │ writes G+0 and G+8   │
│ ctag = 0          │         │ ctag = 1             │
└─────────┬─────────┘         └──────────┬───────────┘
          │                              │
          └────────── NO MERGE ──────────┘
              because G+8 != G+0
                       │
                       ▼
             round-robin drain does not
                preserve program order
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼

 PROGRAM ORDER:  A then B        PROGRAM ORDER:  B then A
 (scalar field, then the         (capability, then a scrub
  capability written over it)     written over it)
 B should win                    A should win

       │                               │
       ▼                               ▼
 if A drains last:               if B drains last:
       │                               │
       ▼                               ▼
 capability DESTROYED            capability SURVIVES
 TAG = 0                         TAG = 1
 S-07 fault                      the scrub is dropped:
 (mcause 25 on first use)        authority outlives the
                                 attempt to destroy it

 Note: A draining last is only wrong in the LEFT column, and B
 draining last is only wrong in the RIGHT one. In each case the
 failure is the OLDER entry landing after the younger one.
```

---

## 6. How This Appears in SQLite

The sequence from the report is:

```text
sqlite3JournalOpen
        │
        ▼
memset(p, 0, sizeof(MemJournal))
        │
        │ plain stores touch G+8
        ▼
plain store G+8
        │
        │ twelve source lines later
        ▼
pJfd->pMethods = ...
        │
        ▼
stc G
```

The `pMethods` field is the first member of `sqlite3_file` and occupies the granule `[p+0, p+16)`. A plain store from `memset` that targets the upper half of this granule can therefore be present in the write buffer at the same time as the later capability store to `pMethods`.

If the older store from `memset` drains last:

```text
Correct state:

pMethods:
┌────────────────────┬────────────────────┐
│       CURSOR       │      METADATA      │
└────────────────────┴────────────────────┘
                TAG = 1


Actual state:

pMethods:
┌────────────────────┬────────────────────┐
│       CURSOR       │  zeros from memset │
└────────────────────┴────────────────────┘
                TAG = 0
```

Later, `sqlite3OsRead` executes:

```text
ldc a4, 0(a0)       # loads pMethods
                    # receives NOT_CAP

ldc a4, 0x20(a4)    # tries to dereference pMethods
                    # mcause 25: UNEXPECTED_OPERAND
```

The failure therefore does not necessarily occur when the corruption happens. The write buffer first leaves an untagged value in memory. The problem becomes visible later, when software tries to use that value as a capability.

---

## 7. Why the Test with 64 Intermediate Stores Passes

The decisive experimental pair is:

```text
Failing form:

plain G+8
stc G

Both entries can be present in the write buffer simultaneously.
Result: 1,107 losses out of 16,384, or 6.76%.
```

```text
Non-failing form:

plain G+8
64 unrelated stores
stc G

The intervening traffic gives the first entry time to drain.
The two conflicting entries are no longer resident together.
Result: 0 losses out of 16,384.
```

The 64 stores do not repair the data. They separate the conflicting operations in time:

```text
Without the gap:

plain G+8 ─────────────┐
                       ├─ resident in WBUF together → reorder
stc G     ─────────────┘


With the gap:

plain G+8 ── drain ── no longer in WBUF

          64 stores

stc G ──────────────── alone in WBUF → no conflict
```

The `wb1` versus `wb3` comparison therefore shows that simultaneous residency of the overlapping entries is a necessary trigger.

---

## 8. Why Simply Synchronizing the Tags Is Unsafe

A tempting fix would be:

```text
If two entries belong to the same granule,
give both entries the ctag of the youngest store.
```

This is dangerous.

Suppose the older plain entry is given `ctag = 1` and then drains last:

```text
        ┌────────────────────┬────────────────────┐
        │       CURSOR       │    stale data X    │
        └────────────────────┴────────────────────┘
                         TAG = 1
```

The system would now have a tagged value with potentially corrupted metadata.

A test that checks only the tag would report:

```text
PASS: TAG = 1
```

But the capability could already be invalid or unsafe.

For that reason, changing only `ctag` is not a valid fix. The implementation must eliminate the existence of two independently drained entries that physically overlap.

---

## 9. The Fix

Two directions were considered. **Option B was implemented** (`5c5f4e3a7`); option A was
rejected as infeasible. Both are described below, with the reasons, because the reasoning is
what stops someone re-proposing the rejected one.

### 9.1 Option A: Granule-Aware Merge — REJECTED

```text
Before:

┌─────────────┐   ┌─────────────────┐
│ plain G+8   │   │ capability G    │
└─────────────┘   └─────────────────┘
 two separate entries


After:

┌───────────────────────────────────┐
│ one entry for the entire granule G│
│                                   │
│ G+0: capability data              │
│ G+8: most recently written data   │
│ TAG: value from the latest store  │
└───────────────────────────────────┘
```

The write buffer must recognize that a capability entry at `G` also covers `G+8`. The overlapping stores are then merged into one entry, so the drain order can no longer change the result.

This is the more complete solution in principle, and it is **not implementable here.** The entry
carries per-byte `valid`/`dirty`/`txblock` masks for the *lower* word only, and the metadata half
(`user`) has **no byte tracking at all**. "Bytes 8–15 are dirty" therefore cannot be represented,
and bytes 8–11 would set the same mask bits as bytes 0–3. Making it representable means widening
those masks and every consumer of them across five files — roughly +224 flops and several hundred
lines.

Worse, the sketch above would have turned a visible failure into an invisible one. A scrub merged
into an entry whose transaction has already been issued sets no dirty bits, so the entry is never
re-drained and is then freed: **L1 ends up correct while DRAM still holds the tagged capability**,
and the capability reappears the moment that cache line is displaced. A test that reads back
immediately would report the fix working.

### 9.2 Option B: Forbid Co-Residency — IMPLEMENTED (`5c5f4e3a7`)

```text
A capability entry for G is already in WBUF
                    │
                    ▼
          a plain store to G+8 arrives
                    │
                    ▼
       detect a conflict at granule level
                    │
                    ▼
       STALL until the capability entry drains
                    │
                    ▼
       only then accept the plain G+8 store
```

The reverse must also apply. If a plain entry for `G+8` is already resident, a capability entry for `G` must not be allocated until the conflicting entry has drained.

**Why this one won.** It is about ten lines, adds no flops, and shares all but the low bit of a
comparator already on that path. Crucially it **mutates nothing** — and that turned out to be the
deciding property rather than a nicety, because `ctag` is sampled *twice*: once at transaction
issue, which is what reaches DRAM, and again at transaction return, which is what reaches L1. Any
fix that changes a resident entry after its transaction has issued makes those two disagree, with
L1 winning every immediate readback and DRAM winning after eviction. Option B never touches a
resident entry, so the question does not arise.

The synthesis-risk worry about `rdy` was investigated and did not survive: `rdy` has exactly one
consumer, its entire fan-in is registers, this module carries no combinational-loop warning, and
the timing-loop criticals cannot arrive through this path because the non-idempotent decode they
pass through is constant-folded away in this configuration. The measured cost is one extra level
of logic.

Same-word stores deliberately do **not** stall — they merge into the single existing entry, where
the existing rules already give the correct answer. Only a *different word of the same granule*
with a capability on one side stalls, which is rare in ordinary code.

**Validation.**

```text
s07-wbuf-tag-reorder      4 faults → 1     (the 1 is the test's own positive control)
s07-wbuf-tag-reorder-ctl  1       → 1     (matched control, unchanged)
full 81-test sweep        differs in exactly ONE row: the reproduction above.
                          All 80 others byte-identical — verdict, cycles,
                          trace hash and fault count.
s07-wbuf-liveness         18,488 cycles against a 400,000 timeout, 0 faults
                          → every stall resolved; no deadlock
```

Liveness was the risk this fix introduces, so it is measured rather than argued.

---

## 10. Shortest Possible Explanation

> The write buffer treats a capability store as a one-word operation when checking for matching or conflicting entries, but the same entry writes two words when it drains. As a result, the buffer allows two entries that appear different by `wtag` but physically overlap. The round-robin arbiter can then drain them in either order, allowing an older store to overwrite the result of a younger store.

The tag loss is the visible symptom. The deeper defect is that **the write buffer tracks and merges an `is_cap` entry as one word even though that entry spans two words**.

---

## 11. Two Things This Does Not Cover

Neither is fixed by `5c5f4e3a7`, and neither should be credited to it.

**Store-to-load forwarding is word-granular in the same way.** A load that hits the write buffer
compares at word granularity and selects metadata bytes using the *lower* word's valid mask, so a
capability load can be answered from a resident entry while a pending store to the other half is
invisible to it. This exists independently of the drain-order defect. Forbidding co-residency
makes it unreachable — two conflicting entries can no longer coexist — but it does not repair the
comparison itself, and it deserves its own investigation rather than a paragraph here.

**AMO over a capability granule is untouched.** Atomics bypass the write buffer entirely and are
excluded from the tag path, so an AMO landing on a capability granule leaves the tag set. That is
a separate, pre-existing residual (invariant I4). Any statement that "tags are correct now" must
exclude it explicitly.
