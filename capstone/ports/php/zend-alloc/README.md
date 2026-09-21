# PHP 5.0.0 Zend allocator as a Capstone domain — CRASH-008

**Status: CAUGHT.** Matched pair passes; `run-crash008.sh` exits 0.

## The claim

PHP 5.0.0's allocator hides small heap overflows. `REAL_SIZE` rounds every request
up to a multiple of 8 (`zend_alloc.c:132`), so a 1–7 byte overrun lands in slack
that belongs to nobody and is inside the same `malloc` chunk. ASan cannot see it:
its redzone starts at the end of the *chunk*, not the end of the *object*. The
corpus has to patch the allocator — flatten `REAL_SIZE` to `(size)` and disable the
cache — before a sanitizer reports anything
(`scripts/build-php-5.0.0-variant.sh:61-65`).

Capstone needs neither patch. `_emalloc` already records the true request in
`p->size` (`zend_alloc.c:201`); bound the returned capability by *that* and the
machine enforces what ASan could only see after allocator surgery.

## The case

`CRASH-008` — `echo date("U", 999999999999999);`

`php_date()` sizes its buffer in one pass and writes it in another. The `'U'` arm
reserves a fixed ten bytes (`ext/standard/datetime.c:358-360`), allocates
`emalloc(size+1)` = 11 (`:424`), then `sprintf`s a 15-digit timestamp and `strcat`s
it (`:438-440`) — 16 bytes including the NUL into an 11-byte buffer. A five-byte
overflow, and a **write**.

Stock, it is invisible: `REAL_SIZE(11) = 16`, so the block really allocated is
header + 16 and the 16-byte write fits **exactly**. The corpus measured `OK` under
both `asan-stock` and `asan-nocache`.

## Result

Both arms are the same source; one constant differs (`ZEND_CAP_BOUND_BYTES`).

| arm | capability bound | outcome |
|---|---|---|
| control (`-DZEND_CAP_BOUNDS_REAL_SIZE`) | header + `REAL_SIZE(11)` = 48+16 = **64** | completes, `retval = 200` |
| fault (default) | header + `size` = 48+11 = **59** | halts, out-of-bounds store |

```
Cap mem access OOB: insn = 00a58023, rs1 = x11, addr = 10158c4bb, size = 1,
                    bounds = (10158c480, 10158c4bb)
domain halted by capability fault: cause = 7, badaddr = 0x10158c4bb
```

`0x4bb - 0x480 = 59` — the bound is exactly header + true size. `insn = 00a58023`
is `sb a0, 0(a1)`, the byte store in `d_strcat`. `addr` is `payload + 11`: the
twelfth byte of an eleven-byte buffer. The allocator was **not** patched — the
`REAL_SIZE` rounding and the size-class cache are both still on.

## Reading the cause code

**Cause 7, not 29.** An out-of-bounds access is raised by the `Cap mem access OOB`
path in `capstone-qemu/target/riscv/op_helper.c:1532-1540`, which maps a store to
`RISCV_EXCP_STORE_AMO_ACCESS_FAULT` (7) and a load to `LOAD_ACCESS_FAULT` (5).
The 24–30 numbering documented elsewhere (24 tag-gone, 25 revoked, 29 out-of-bounds)
belongs to a *different* delivery path and does **not** apply here. The message
text says "capability fault" for any synchronous fault taken inside a domain, so
the text alone does not identify the cause — read the `bounds = (...)` line.

## Bounds model (option A)

    base   = the header
    end    = header + sizeof(hdr) + <size | REAL_SIZE(size)>
    cursor = header + sizeof(hdr)        <- what the caller is handed

Bounding from the **header**, not the payload, is deliberate. `_efree`
(`zend_alloc.c:250`) and `_erealloc` (`:320`) recover the header by subtracting
from the caller's own pointer. With payload-only bounds that subtraction is out of
bounds and *every free* faults, long before any bug is reached — the port would be
measuring its own porting decision instead of PHP's defect. The cost is that a
header underflow is not caught; that is not the defect under test.

## Numbers do not match the corpus, and cannot

`zend_mem_header` holds `pNext` and `pLast`, which are 128-bit capabilities here,
so `sizeof(zend_mem_header)` is **48**, not the 24 measured on x86-64, and the
region is 59 bytes, not 35. Verified by `_Static_assert`. The *shape* reproduces
exactly; the arithmetic does not. Do not quote ASan's figures against this build.

## What is NOT measured here

- **The temporal axis.** `_efree` still parks blocks in `AG(cache)` exactly as PHP
  ships it, and there is **no revoke**. That is deliberate: this port measures the
  spatial axis, and a revoke-on-free arm would make a temporal fault
  indistinguishable from a bounds fault. The 23 corpus use-after-free cases masked
  by the cache are a separate experiment, and all of them are entangled with the
  zval/executor machinery.
- **Reclamation.** `ZEND_DO_MALLOC` is a bump arena; a domain has no libc and no OS.
  Nothing here says anything about fragmentation.
- **`_erealloc`.** Not exercised by this case.

## Guards

- `-O0` is required, not preferred. At `-O1`+ the store into a buffer nothing reads
  can be eliminated and the access under test is never emitted.
- The runner refuses to report if the control does not complete: **exit 75, no
  verdict**, per `capstone/bug-corpora/README.md`.
- Build-time gate: if no `shrink` is in the image, or the two arms are
  byte-identical, `BUILD-INVALID` / exit 3 — otherwise a broken build and a genuine
  MISS look the same.

## Files

| | |
|---|---|
| `zend_capstone_alloc.h` | the ported allocator; `_emalloc`/`_efree`, cache and `REAL_SIZE` unpatched |
| `crash008_domain.c` | `php_date()`'s `'U'` arm plus the freestanding string helpers it needs |
| `run-crash008.sh` | matched pair, control first, build gate |
