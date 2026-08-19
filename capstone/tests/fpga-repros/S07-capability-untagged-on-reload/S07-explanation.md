# S-07 Explained: How a Capability Sporadically Comes Back Untagged

> **STATUS: root cause confirmed, and FIXED.** The fix is `5c5f4e3a7` — forbid granule
> co-residency in the write buffer (section 8). This document explains the defect and the
> fix that shipped. The engineering detail, **the alternatives that were considered and
> rejected**, and the acceptance criteria are in `rtl/ROOT-CAUSE-AND-FIX-OPTIONS.md` — this
> document deliberately does not carry them.

This document explains the S-07 defect intuitively, using diagrams and step-by-step examples. It focuses on the root cause confirmed in RTL and on silicon: the write buffer treats a capability entry as one word when detecting conflicts, even though it writes two words when drained.

```text
                    THE WHOLE STORY, AT A GLANCE

   §1-2   a capability = 16 bytes + ONE shared tag,
          written by an stc that covers BOTH 8-byte words
                              │
                              ▼
   §3     the write buffer COMPARES entries by 64-bit word address,
          so "plain G+8" and "stc G" look like different addresses
                              │
                              ▼
   §4-5   they are not — they overlap. And the drain arbiter is
          round-robin, so the OLDER entry can land LAST
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        tag CLEARED                     scrub DROPPED
        capability destroyed            capability survives
        → mcause 25 on next use         → authority outlives
          (this is what §6 shows           the attempt to
           SQLite hitting)                 destroy it
                              │
                              ▼
   §7     confirmed by the wb1/wb3 pair: 64 intervening stores
          separate the two entries in time and the loss goes to ZERO
                              │
                              ▼
   §8     THE FIX — make the overlapping pair impossible to create
                              │
                              ▼
   §10    two adjacent defects this does NOT repair
```

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

**Why this matters more than it first appears.** Ask what that dropped store was *for*:

```text
          what a plain store over a capability granule IS

   memset()  explicit_bzero()  free-list poison  clear-before-reuse
        │            │                │                 │
        └────────────┴────────┬───────┴─────────────────┘
                              ▼
              "software DESTROYS authority it
                  no longer wants to hold"
                              │
                   the store is DROPPED
                              │
                              ▼
        ┌───────────────────────────────────────────────┐
        │  the capability SURVIVES the operation        │
        │  intended to destroy it                       │
        │                                               │
        │  = failure to revoke by overwrite             │
        │    weaker than forging authority              │
        │    stronger than losing a scalar              │
        └───────────────────────────────────────────────┘
```

The bound on how bad this gets — **measured**, not argued:

```text
   outcome                          observed?
   ─────────────────────────────    ─────────────────────────────
   capability lost (tag cleared)    YES  — this is S-07
   capability survives a scrub      YES  — the dropped store
   TAGGED capability over
   attacker-chosen data             NO   — empty in BOTH directions

   why: the capability entry writes BOTH words, so what it leaves
        behind is the ORIGINAL capability, never a tagged mixture

        ┌──────────────┬──────────────┐
        │   CURSOR     │   METADATA   │   both halves from the same stc
        └──────────────┴──────────────┘   → intact, or gone. No hybrid.
```

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

```text
   struct MemJournal  (p = its base)

   p+0                     p+8                    p+16
    ├───────────────────────┼──────────────────────┤
    │  pMethods : CURSOR    │  pMethods : METADATA │   ← ONE granule,
    ├───────────────────────┴──────────────────────┤     ONE tag
    │  ... the rest of the struct ...              │
    └──────────────────────────────────────────────┘

   memset(p, 0, sizeof(MemJournal))  writes p+8  ─┐
                                                  ├─ SAME granule,
   pJfd->pMethods = ...  (stc)       writes p+0  ─┘  DIFFERENT word
                                                     → no merge
                                                     → reorderable
```

Note that ordinary, entirely correct C is enough to build this: a struct zeroed and then given
a capability-typed first member. Nothing unusual is required of the source program.

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

## 8. The Fix: Forbid Granule Co-Residency

Implemented in `5c5f4e3a7` (option B in `rtl/ROOT-CAUSE-AND-FIX-OPTIONS.md`, where the
alternatives and the reasons they were rejected are recorded).

### 8.1 The Rule

```text
                    a store request arrives
                              │
                              ▼
             ┌─────────────────────────────────┐
             │ does a RESIDENT entry sit in    │
             │ the SAME 16-byte granule but    │
             │ the OTHER 64-bit word?          │
             └────────────────┬────────────────┘
                     no       │       yes
              ┌───────────────┴───────────────┐
              ▼                               ▼
        accept as before          ┌────────────────────────────┐
        (merge or allocate)       │ is EITHER side is_cap = 1? │
                                  └──────────────┬─────────────┘
                                        no       │      yes
                                 ┌───────────────┴──────────┐
                                 ▼                          ▼
                           accept as before             ***STALL***
                        (both write ctag = 0            do not accept
                         to the same bit —              until the resident
                         same value, no hazard)         entry has DRAINED
```

Symmetric by construction — it is the *pair* that is forbidden, not an order:

```text
   capability G resident, plain G+8 arrives   →  STALL
   plain G+8 resident, capability G arrives   →  STALL

   so this state is now UNREACHABLE:

     ┌─────────────┐        ┌─────────────────┐
     │ plain G+8   │   ✗    │ capability G    │      never co-resident
     └─────────────┘        └─────────────────┘      → no drain order
                                                        can exist to be wrong
```

### 8.2 What It Costs

```text
   lines of RTL      ~10, beside the existing ni_conflict stall
   new flops         0
   new comparator    0  — shares all but the low bit of one already
                          on that path
   timing            one extra level of logic
   lint              every counter identical to baseline

   what does NOT stall — deliberately:

     same-word stores ──► merge into the one existing entry, where the
                          existing rules already give the right answer

     two plain entries ─► both write ctag = 0 to the same tag bit.
                          Same value. No hazard.

   only "different word of the same granule, capability on one side"
   stalls — which is rare in ordinary code.
```

### 8.3 Why This Shape and Not a Cleverer One

The deciding property is that the fix **mutates nothing.** That sounds like a nicety. It is
not, and here is the reason, which governs every candidate fix and every acceptance test:

```text
              ctag IS SAMPLED TWICE, NOT ONCE

   wbuffer entry
        │
        ├─── at TRANSACTION ISSUE ───► ctag copied ──► AXI ──►  DRAM
        │      (wt_dcache_wbuffer.sv:298-299)
        │
        └─── at TRANSACTION RETURN ──► ctag copied ──►  L1 cache
               (wt_dcache_wbuffer.sv:415-416)


   So a fix that CHANGES a resident entry after its transaction issued:

        ┌──────────────┐                    ┌──────────────┐
        │  L1: tag = 0 │                    │ DRAM: tag = 1│
        └──────────────┘                    └──────────────┘
               ▲                                   ▲
               │                                   │
        wins every IMMEDIATE                wins after the line
        read-back → the test                is EVICTED → the
        says PASS                           capability COMES BACK

        = a fix that looks perfect and leaves authority resurrectable
```

```text
   THE FIX vs THAT TRAP

   stall at ALLOCATION ──► no resident entry is ever touched
                      ──► the two samples can never disagree
                      ──► the question does not arise
```

The corollary is a rule for tests, not just for fixes:

```text
   ┌────────────────────────────────────────────────────────┐
   │ EVERY tag acceptance test needs a FORCED-EVICTION      │
   │ RELOAD leg. An immediate read-back cannot see the      │
   │ failure mode above.                                    │
   └────────────────────────────────────────────────────────┘
```

### 8.4 Liveness — The Risk This Fix Introduces

A stall can wedge. This one cannot, and the argument is that no *cycle* exists:

```text
   the stall holds off an incoming store
                    │
                    ▼
   it resolves when the resident entry DRAINS
                    │
                    ▼
   draining needs `checked`  ←── rd_req_o & rd_ack_i
                    │
                    ▼
   rd_ack_i comes from a STRICT-PRIORITY arbiter
   (wt_dcache_mem.sv:186-188) in which this port is
   the ONLY low-priority one (wt_dcache.sv:297)
                    │
                    ▼
   ...so does resolution depend on other ports going quiet?  YES.
   Is that a NEW dependency?                                  NO:

     • the LDC being backpressured parks in LDC_CLEAR_WAIT with
       data_req LOW — the state this stall drives the core into is
       the one that STOPS competing for the tag-check port
     • backpressure stalls issue, so the load stream dries up too
     • the pre-existing `full` and `ni_conflict` stalls resolve
       through the IDENTICAL path

   → this raises the FREQUENCY of an existing dependency.
     It does not create a new class of one.
```

An earlier version of that argument claimed drain/check/evict consult only the buffer's own
state. An audit **refuted** it — the arbiter dependency above is real — and the RTL comment
was corrected. A wrong argument for a right conclusion is still wrong.

### 8.5 Validation — Measured, Not Argued

```text
   TAG CORRECTNESS
   s07-wbuf-tag-reorder      4 faults → 1    (the 1 is the test's own
                                              positive control)
   s07-wbuf-tag-reorder-ctl  1       → 1    (matched control, unchanged
                                              — one variable between them)

   NO COLLATERAL DAMAGE
   full 81-test sweep        differs in EXACTLY ONE row: the reproduction.
                             All 80 others byte-identical — verdict, cycles,
                             trace hash and fault count.

   LIVENESS, and proof the stall ACTUALLY FIRES
                             pre-fix RTL        with the stall
                             ───────────        ──────────────
   s07-wbuf-liveness           16,998 cyc   →     18,488 cyc    0 faults
   s07-wbuf-stall-corners       8,362 cyc   →      8,505 cyc    0 faults

                             ▲
                             │  an UNCHANGED cycle count would have proved
                             │  only that the test never created the
                             │  condition. The delta IS the stalling.
```

`s07-wbuf-stall-corners` covers the two corners an audit named as unreachable by any existing
test: revocation traffic concurrent with a stall, and the buffer full while the stall is
asserted.

---

## 9. Shortest Possible Explanation

> The write buffer treats a capability store as a one-word operation when checking for matching or conflicting entries, but the same entry writes two words when it drains. As a result, the buffer allows two entries that appear different by `wtag` but physically overlap. The round-robin arbiter can then drain them in either order, allowing an older store to overwrite the result of a younger store.

```text
      COMPARED AS                          DRAINED AS
   ┌──────────────┐                  ┌──────────────┬──────────────┐
   │  one WORD    │        vs        │     word     │     word     │
   │   (8 B)      │                  │      +       │      +       │
   └──────────────┘                  │        one shared TAG       │
                                     └──────────────┴──────────────┘
          └──────────────── the mismatch ─────────────────┘

   THE FIX closes the gap not by making the comparison wider,
   but by making the overlapping pair IMPOSSIBLE TO CREATE.
```

The tag loss is the visible symptom. The deeper defect is that **the write buffer tracks and merges an `is_cap` entry as one word even though that entry spans two words**.

---

## 10. Two Things This Does Not Cover

Neither is fixed by `5c5f4e3a7`, and neither should be credited to it.

```text
   ┌──────────────────────────────────────────────────────────────┐
   │ 1. STORE-TO-LOAD FORWARDING — word-granular in the SAME way  │
   └──────────────────────────────────────────────────────────────┘

     a capability LOAD hits the write buffer
                    │
                    ▼
     compares at WORD granularity, and selects metadata bytes
     using the LOWER word's valid mask
                    │
                    ▼
     ┌────────────────────────────────────────────────┐
     │ so a pending store to the OTHER half is        │
     │ INVISIBLE to the load being answered           │
     └────────────────────────────────────────────────┘

     this exists INDEPENDENTLY of the drain-order defect

     status:   PARTLY unreachable — the WORD-MISS case IS STILL LIVE
               ─────────────────────────────────────────────────────
               an earlier version of this box said "UNREACHABLE, not
               REPAIRED". That was REFUTED on 19-08-2026 and MEASURED.
```

The reasoning that failed: forbidding co-residency stops two conflicting *entries* coexisting,
so the two-entry case is genuinely gone. But the residual needs only **ONE** entry — the stall
is an allocation-time check, and **a load never consults it at all**:

```text
     stc  G, cap     capability drains to L1 → cap_tag_q[G>>4] = 1
          │
     sd   x0, G+8    ONE plain entry, word 1, ctag=0 — still resident
          │          (no second entry, so nothing to conflict WITH)
          │
     ldc  rd, G      granule-aligned → always compares WORD 0
          │
          ▼
     ┌──────────────────────────────────────────────────┐
     │  misses the word-1 entry  →  wbuffer_be = 0      │
     │  → falls through to the STALE cap_tag_q (= 1)    │
     │  → returns a LIVE capability over memory the     │
     │    program just scrubbed                         │
     └──────────────────────────────────────────────────┘

     MEASURED, matched pair, one variable (a drain delay):

        entry still resident   ──►   8 traps / 16 legs
        300-iteration delay    ──►  16 traps / 16 legs
                                    ▲
        POLARITY IS INVERTED here ──┘  a trap is CORRECT;
        its ABSENCE is the defect. Do NOT compare these
        counts against §9's — they read backwards.

     severity, and why this is a residual and not a regression:

        pre-fix dropped scrub   PERSISTENT  ─── capability survives forever
        this window             TRANSIENT   ─── closes when the entry drains
```

The 8-of-16 is **not** a probability. The alternating pattern is *consistent with* the trap
handler draining the buffer between legs, which would reset the phase each time — that
explanation is a hypothesis, not a measurement. The **existence** is the result.

Correct wording: **the fix makes the two-entry forwarding case unreachable and leaves the
one-entry word-miss case live.** Tests `s07-wbuf-forward-residual` and its matched control
(`capstone-ariane 6175ea654`). Still deserves its own repro folder.

```text
   ┌──────────────────────────────────────────────────────────────┐
   │ 2. AMO OVER A CAPABILITY GRANULE — untouched                 │
   └──────────────────────────────────────────────────────────────┘

     AMO ──✗── write buffer          (bypassed entirely)
         ──✗── tag path              (excluded)
              │
              ▼
     an AMO landing on a capability granule LEAVES THE TAG SET

     status:   separate, PRE-EXISTING residual (invariant I4)

     ┌────────────────────────────────────────────────┐
     │ any statement that "tags are correct now"      │
     │ MUST exclude this explicitly                   │
     └────────────────────────────────────────────────┘
```
