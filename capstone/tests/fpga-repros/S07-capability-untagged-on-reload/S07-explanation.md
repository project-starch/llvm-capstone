# S-07 Explained: How a Capability Sporadically Comes Back Untagged

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

The write buffer can choose the opposite order:

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

Then `stc` overwrites the entire granule:

```text
After stc G:

        ┌────────────────────┬────────────────────┐
        │       CURSOR       │      METADATA      │
        └────────────────────┴────────────────────┘
                         TAG = 1
```

The capability remains intact, but the result of the older plain store disappears.

```text
S-07 does not occur: the capability remains intact
But: the plain store is silently lost
```

This is an integrity failure rather than tag loss. The capability-store entry writes both words, so the final capability remains intact instead of becoming a tagged capability with corrupted metadata.

---

## 5. The Entire Defect in One Diagram

```text
PROGRAM ORDER
───────────────────────────────────────────────►

 plain store G+8                   stc G
 older                             younger
 should be overwritten             should win
       │                               │
       ▼                               ▼

┌───────────────────┐         ┌──────────────────────┐
│ WBUF entry A      │         │ WBUF entry B         │
│ wtag = G+8        │         │ wtag = G+0           │
│ is_cap = 0        │         │ is_cap = 1           │
│ writes G+8        │         │ writes G+0 and G+8   │
│ ctag = 0          │         │ ctag = 1             │
└─────────┬─────────┘         └──────────┬───────────┘
          │                              │
          └────────── NO MERGE ─────────┘
              because G+8 != G+0
                       │
                       ▼
             round-robin drain does not
                preserve program order
                       │
              ┌────────┴────────┐
              ▼                 ▼

         A drains last      B drains last

         capability        capability intact,
         destroyed         plain store lost

         TAG = 0           TAG = 1
         S-07 fault        silent data loss
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

## 9. Two Correct Fix Directions

### 9.1 Option A: Granule-Aware Merge

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

This is the more complete solution, but it requires substantial changes to the merge path.

### 9.2 Option B: Forbid Co-Residency

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

This approach is simpler to reason about and obviously prevents the overlap. However, it adds another condition to the `rdy` logic and may increase synthesis risk on an already problematic timing cone.

Both are real fix directions. The final choice involves a tradeoff between implementation completeness and synthesis risk.

---

## 10. Shortest Possible Explanation

> The write buffer treats a capability store as a one-word operation when checking for matching or conflicting entries, but the same entry writes two words when it drains. As a result, the buffer allows two entries that appear different by `wtag` but physically overlap. The round-robin arbiter can then drain them in either order, allowing an older store to overwrite the result of a younger store.

The tag loss is the visible symptom. The deeper defect is that **the write buffer tracks and merges an `is_cap` entry as one word even though that entry spans two words**.
