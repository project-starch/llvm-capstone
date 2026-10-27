# PHP 5.0.0 Zend allocator as a Capstone domain

Two axes, each a matched pair, each passing.

| axis | trigger | runner | result |
|---|---|---|---|
| spatial | **CRASH-008**, a real corpus case | `run-crash008.sh` | CAUGHT, out-of-bounds store |
| temporal | synthetic, allocator-level | `run-uaf.sh` | CAUGHT, revoked alias |

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

## Temporal axis — revoke-on-free (`run-uaf.sh`)

**Synthetic, and labelled as such.** Not a CRASH-nnn case: all 23 cache-masked
use-after-free cases in the corpus are entangled with the zval/refcount/executor
machinery. What this reproduces is the allocator-level property the corpus itself
isolates at `logic-bugs.md:643-644` — `efree()` parks the block in `AG(cache)` and
never calls `free()`, so the shadow still says "addressable" and the dangling write
is legal to ASan.

Every allocation is carved with **SPLIT**, the only derivation that produces a
fresh revocation-tree node; `cincoffset` would inherit the arena's node and a
revoke would sweep the whole heap. Order: SPLIT → MREV (handle, while still LIN) →
DELIN (the caller's alias) → SHRINK (the option-A bound).

| arm | outcome |
|---|---|
| control (`-DZEND_NO_REVOKE`) | completes, `retval = 213` — the use-after-free **survived**, as on stock PHP |
| fault (`-DZEND_TEMPORAL`) | halts, `Cap mem access requires capability`, cause **24** |

The trigger allocates 32 bytes, frees it, **reallocates the same size class** (PHP
hands back the same block), writes through the new pointer, then writes through
the stale one. That is the classic reuse hazard: on stock PHP the stale write
lands in the new tenant's memory, and ASan cannot see it because the block was
never released. The domain fails loudly with `ZEND_UAF_NOREUSE` if the block was
*not* recycled — otherwise a fault, or its absence, would say nothing about
use-after-free.

**Cause 24 (tag gone), not 25 (revoked).** At `-O0` the alias is spilled and
reloaded, so the tag is already gone by the time the access issues. That is
indistinguishable from an unrelated spill until the control shows the same program
completing — which is exactly why the control is mandatory, not politeness.

### Address reuse under revoke-on-free — measured, not assumed

An earlier version of this file claimed revoke-on-free and address reuse were
**mutually exclusive**, reasoning that a revoked capability cannot be parked in
`AG(cache)` and that `SPLIT` is one-way so the range cannot be re-minted. The
first half is true. **The second half is false, and the conclusion was wrong.**
It was inferred from `revoke_on_free_alloc.h:148` *discarding* revoke's return
value — someone else's design choice — rather than tested. Two probes settle it:

| probe | question | answer |
|---|---|---|
| `probes/revoke-reclaim.c` | does REVOKE hand authority back? | **yes** — returns tagged, `base` and `end` intact over the block, and a re-minted write lands |
| `probes/revoke-reuse-safety.c` | after re-minting, is the PREVIOUS tenant's alias still dead? | **yes** — it faults (cause 24); reuse does not resurrect it |

So the cache stays **on in both arms**, and recycling is a capability operation
rather than a pointer assignment:

    _efree   REVOKE(handle) -> reclaimed LIN authority -> park THAT in AG(cache)
    _emalloc pop -> MREV (fresh node) -> DELIN -> SHRINK to the NEW request

Two consequences worth stating:

- **What the cache parks changes.** PHP parks a `zend_mem_header *`. That cannot
  work when the block is bounded per request, because the bound belongs to the
  size the block was last handed out at, and the cache is a size-*class* cache:
  bucket `i` holds anything that rounded to `i*8`, so a block freed as 9 bytes can
  be re-served for 11. The unshrunk block is parked and re-bounded on handout,
  which keeps PHP's size-class policy exactly and gives each lifetime its own
  correct bound.
- **The arena still is not reclaimed.** `SPLIT` remains one-way, so address space
  is consumed monotonically by *first* allocations. Reuse recycles a block through
  the cache; it does not return space to the arena.

**Both arms therefore run PHP's allocator unpatched** — `REAL_SIZE` still rounds to
8, `AG(cache)` still recycles — which is the whole claim, and is now true of the
temporal arm as well as the spatial one.

Rejected alternative: park the block without revoking and revoke only on eviction.
That preserves recycling, but the use-after-free window becomes exactly the cache
residency — i.e. it does not fix the bug at all.

### A control is not free of the mechanism it removes

`-DZEND_NO_REVOKE` cannot simply delete the revoke and leave the rest: with no
revoke there is no reclaimed LIN authority, so the parked block is still the
*delinearized* alias, and `MREV` on a NONLIN capability trips
`assert(rs1_v->val.cap.type == CAP_TYPE_LIN)` in QEMU's `helper_csmrev` and
**aborts the emulator**. The control therefore recycles by handing the same alias
straight back — which is precisely what stock PHP does. Not re-minting is a
consequence of removing the revoke, not a second difference.

This was found only because the trigger was changed to reallocate after freeing.
The single-allocation version passed while never executing the cache path at all.

### Keeping the axes apart

The temporal trigger is correctly sized and every access is well inside its
bounds, so a bounds fault cannot masquerade as a temporal one. `run-uaf.sh` fails
explicitly if it sees a `Cap mem access OOB` line.

## What is NOT measured here

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
| `uaf_domain.c` | the synthetic use-after-free trigger |
| `run-crash008.sh` | spatial matched pair, control first, build gate |
| `run-uaf.sh` | temporal matched pair; also fails on an OOB line |
| `probes/revoke-primitive.c` | Step-0: SPLIT → MREV → DELIN → REVOKE, with its own control |
| `probes/revoke-reclaim.c` | does REVOKE return usable authority? (reports; never asserts) |
| `probes/revoke-reuse-safety.c` | after re-minting, does the old alias stay dead? |
