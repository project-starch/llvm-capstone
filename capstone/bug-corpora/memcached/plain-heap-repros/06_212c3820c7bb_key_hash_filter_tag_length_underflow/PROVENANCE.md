# 212c3820c7bb — proxy: fix underflow in key hash filters

## The defect

`mcp_key_hash_filter_tag` searched for the **closing** tag starting at the **opening** tag's own position. The function's comment explicitly permits a two-character configuration whose characters are equal, such as `"$$"`; then `memchr` returns `t1` itself, `t2 - t1 - 1` underflows to `SIZE_MAX`, and the caller hashes that many bytes.

## Upstream defect

- **Fix:** `212c3820c7bb`, *"proxy: fix underflow in key hash filters"*, `proxy_lua.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin searches from `t1+1`.

## The vulnerable code, quoted from the fix's parent

```c
        size_t remain = klen - (t1 - key);
        // must be at least one character inbetween the tags to hash.
        if (remain > 1) {
            const char *t2 = memchr(t1, conf[1], remain);

            if (t2) {
                *newlen = t2 - t1 - 1;
                return t1+1;
            }
```

## The fix

```c
            const char *t2 = memchr(t1+1, conf[1], remain-1);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The function is `static` and pure, so the reduction reproduces its control flow directly with no lua interpreter and no proxy configuration. The hasher's unbounded read is reduced to the labelled probe at the first byte outside the key.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
