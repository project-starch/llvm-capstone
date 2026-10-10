# d5d9ff0 — a headroom guard counts a marker's characters and forgets its terminator

**The upstream fix is cited by HASH ONLY, deliberately.** Its subject line names an outside
contributor, and a person's name must not enter a committed file in this tree.
`docs/ref/memcached-spatial-defect-triage.md:43-45` records the same constraint against this hash,
which is why this row sat undispositioned while the other two class-A candidates were resolved.

## The defect

`item_cachedump` fills a `malloc`ed buffer with one line per item and then appends `"END\r\n"`. Its
headroom guard reserves **five** bytes for that marker. The marker is five characters — but it is
written with a **terminating NUL**, because the buffer is handed back as a C string. Six bytes are
needed. An entry whose length brings `bufcurr` to exactly `memlimit - 5` passes the guard, and the
terminator lands at offset `memlimit`: one byte past the allocation.

## Upstream defect

- **Fix:** `d5d9ff0`. It changes the guard to `+ 6` and says so in its own comment: `6 is END\r\n\0`.
- **CVE:** none assigned.
- **Live at the `1.6.45` pin: NO — the fix is already in, so this is a fix-reversal.**

## The vulnerable code, quoted from upstream

The allocation and the guard, `d5d9ff0^:items.c:222-238`:

```c
    buffer = malloc(memlimit);
    if (buffer == 0) return 0;
    bufcurr = 0;

    while(1) {
        ...
        sprintf(temp, "ITEM %s [%u b; %lu s]\r\n", ITEM_key(it), it->nbytes - 2, it->time);
        len = strlen(temp);
        if (bufcurr + len +5 > memlimit)  /* 5 is END\r\n */
            break;
        strcpy(buffer + bufcurr, temp);
        bufcurr+=len;
```

and the fix's own diff:

```diff
-        if (bufcurr + len +5 > memlimit)  /* 5 is END\r\n */
+        if (bufcurr + len + 6 > memlimit)  /* 6 is END\r\n\0 */
```

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry — the latter has
called backported fixes live before — and keyed on the **identifier** rather than a bare number, so
the probe cannot match an unrelated `+ 5`. Two-sided: `1.6.45:items.c` contains the fixed headroom
(`+ 6 > memlimit`, `6 is END\r\n\0`) **1 time** and the pre-fix form **0 times**.

Liveness is **recorded, never required** — the convention is at
`../../allocator-repros/README.md:132-135`, and treating it as a requirement is what kept this
corpus at one case.

## Why this is the NOT-NESTED row, and not a slab case

`buffer` comes from `malloc` directly, at `items.c:222`. There is no slab class and no object cache
between the consumer and the platform allocator — unlike `../allocator-repros`, whose boundary is a
chunk `slabs.c` carved or an object `cache.c` handed out. So the only bound in existence is the
`malloc`'s own, and the terminator leaves it.

**Reachability note:** `item_cachedump` is reached from the `stats cachedump` command, which is
admin-facing and disabled by default in recent versions; the entry point is `process_stat`. That is a
belief about reachability, not a measurement.

## What is real here, and what is reduced

**Real:** the allocator. The platform's own `malloc`/`free`. The arms differ by exactly the upstream
fix — the headroom constant, 5 against 6 — and nothing else: the entry, its length, and the
allocation size are identical between them.

**Reduced:** `memlimit` is 64 rather than the live value, and that choice is load-bearing for the
CheriBSD prediction rather than cosmetic: 64 is **exactly a size class**, so the allocator carries no
slack and a one-byte crossing really does leave the usable allocation. This corpus's case 0 predicted
a catch at request 9 and was **refuted** because a 9-byte request yields a 16-byte capability, so the
size is now chosen deliberately and recorded. `sprintf`/`strcpy` are reduced to the writes they
perform; nothing about *where* the NUL lands is reduced, because that is the defect. The case
**asserts** that the entry is admitted and the terminator crosses at the pin, and rejected under the
fix.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control arm first; and **ASan sees it** — `heap-buffer-overflow` on the buggy arm,
silent on the fixed one.

**Does NOT measure** the Capstone, PoisonCap or CheriBSD arms. Those are declared predictions in
`case.json`. The CheriBSD prediction is **conditional on the request size** for the reason above.

**N = 1 per cell.**
