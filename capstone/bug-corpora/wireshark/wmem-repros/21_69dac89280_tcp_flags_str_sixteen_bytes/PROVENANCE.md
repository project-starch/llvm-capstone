# 69dac89280 — tcp: simplify tcp.flags.str, fix off-by-one

## The defect

The buffer size is a local constant of 16 while the output is nine letters plus eight separators plus a terminator, 18 bytes, written unconditionally.

## Upstream defect

- **Fix:** `69dac89280` ("tcp: simplify tcp.flags.str, fix off-by-one").
- **CVE:** none assigned.
- **Live at the `v4.6.8` pin: NO** — the fix is already in, so this is a **fix-reversal.**

## The vulnerable code, quoted from upstream

The buffer and its size, `69dac89280^:epan/dissectors/packet-tcp.c:4304-4310`:

```c
    static const char flags[][4] = { "F", "S", "R", "P", "A", "U", "E", "C", "N" };
    const int maxlength = 16; /* Max Flags length*/
    char *pbuf;
    const char *buf;
    int i;
    buf = pbuf = (char *) wmem_alloc(wmem_packet_scope(), maxlength);
```

The fix rewrites the function rather than adjusting the constant, and its replacement is bounded by
construction:

```diff
+    const unsigned flags_count = 12;
+    /* upper three bytes are marked as reserved ('R'). */
+    buf = wmem_strdup(wmem_packet_scope(), "RRRNCEUAPRSF");
```

**Liveness**, read from the pinned source rather than from ancestry — the latter has called
backported fixes live before — and keyed on an identifier that exists on exactly one side:

read from the PINNED source, and the first probe was REPLACED because it was not two-sided: `tcp_flags_to_str_first_letter` is the function's name and exists on BOTH sides of the fix, so it cannot distinguish them. Re-keyed on the fix's own added constant: v4.6.8:epan/dissectors/packet-tcp.c contains `const unsigned flags_count = 12` 1 time and the pre-fix `const int maxlength = 16` 0 times. Two-sided. A fix-reversal.

**Nine flags, eight separators and a terminator are eighteen bytes; the buffer is sixteen.** No
crafted input is needed — the overflow is fixed and compile-time-known, which is why it was found by
inspection rather than by a fuzzer. Because the fix replaces the function, this row's two arms differ
by the **written length** rather than by one term; every other spatial row in this corpus is a
one-term reversal, so the case and this file both say so rather than letting a reader assume.

## Why this is a NESTED row

wmem, a chunk the BLOCK or BLOCK_FAST allocator carved from a block g_malloc handed out. An inner allocator carved the crossed region, so under this inventory's axis -- WHO ALLOCATED THE OBJECT -- this row IS NESTED. A malloc-granular bound cannot see the crossing: the block wmem carved it
from is one `g_malloc`, and the access stays inside that block.

## What is real here, and what is reduced

**Real:** the allocator. Upstream's own wmem, through this corpus's seam, with the chunks carved
consecutively from the same block so the successor's position can be asserted.

**Reduced:** MAXLENGTH 16 and NFLAGS 9 are upstream's own values, unreduced. The case asserts that the output does not fit and that the overrun stays inside the block.

## What the run establishes, and what it does not

**Establishes:** the crossing is created — the case's own `CHECK` assertions must hold for it to exit
0, so a reduction whose arithmetic missed fails rather than reporting a verdict about nothing — and
**stock CheriBSD does not catch it**, measured 2026-10-07 with a revocation control faulting in the
same boot.

**Does NOT measure** the Capstone, PoisonCap or native arms. Those are declared: these four cases
have not had a Capstone domain build. Note also what the corpus's own `corpus.json` says and which
applies here: every arm of this harness narrows a wmem allocation to its request via `wm_narrow()`,
so a spatial crossing faults on *all* arms and these rows do not discriminate the chunk port.

**N = 1 per cell.**
