# 40aff8b0f113 — Fixed a memory corruption bug in stats.

## The defect

The bare `stats` command concatenates a server blob and an engine blob into a buffer sized as exactly the sum of their lengths. It then calls `append_ascii_stats` with a null key and value, whose branch is `sprintf(pos, "END\r\n")` — six bytes including the NUL — at a position that is already the end of the allocation.

## Upstream defect

- **Fix:** `40aff8b0f113`, *"Fixed a memory corruption bug in stats."*, `memcached.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The construct was removed rather than fixed: the two-blob concatenation no longer exists.

## The vulnerable code, quoted from the fix's parent

```c
        buf = malloc(server_statlen + engine_statlen);
        ptr = buf;
        memcpy(ptr, server_statbuf, server_statlen);
        ptr += server_statlen;
        memcpy(ptr, engine_statbuf, engine_statlen);
        ptr += engine_statlen;
        engine_statlen += append_ascii_stats(ptr, NULL, 0, NULL, 0);
```

## The fix

```c
        /* 6 is: strlen("END\r\n") + strlen("\0") */
        buf = calloc(1, server_statlen + engine_statlen + 6);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The two blobs are filled with marker bytes rather than real statistics, and the trailer write is reduced to the labelled probe at its first byte.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
