# 391f2e4762bf — Fix freesuffix corruption.

## The defect

The suffix freelist is an array of `char *`. Its grow path reallocated it to `freesuffixtotal * 2` **bytes** and then set `freesuffixtotal` to that same number of **elements**, so on a 64-bit target the array was eight times too small for the capacity it advertised — and the store that immediately follows is already outside it.

## Upstream defect

- **Fix:** `391f2e4762bf`, *"Fix freesuffix corruption."*, `memcached.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The construct was removed rather than fixed: 1.6.45 has no suffix freelist at all.

## The vulnerable code, quoted from the fix's parent

```c
        /* try to enlarge free connections array */
        char **new_freesuffix = realloc(freesuffix, freesuffixtotal * 2);
        if (new_freesuffix) {
            freesuffixtotal *= 2;
            freesuffix = new_freesuffix;
            freesuffix[freesuffixcurr++] = s;
```

## The fix

```c
        char **new_freesuffix = realloc(freesuffix,
            sizeof(char *) * freesuffixtotal * 2);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The array holds eight pointers and is grown by the two arms' own expressions; the store that follows is reduced to the labelled probe at the first byte past.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
