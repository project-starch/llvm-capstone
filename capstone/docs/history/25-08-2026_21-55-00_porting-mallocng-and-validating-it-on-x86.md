# Porting mallocng to capabilities, and proving the port on x86 first

The six musl files that do not compile are all mallocng, and all six fail on one
line. That line is not the problem.

## What actually blocks mallocng

The compiler stops at `meta.h:20`:

    char pad[UNIT - sizeof(struct meta *) - 1];

`UNIT` is 16 and a capability is 16 bytes, so `16 - 16 - 1` underflows to
SIZE_MAX. Measured, not assumed: `sizeof(void*) == 16` and the pad expression
exceeds 1000. Setting `UNIT` to 32 makes all six compile, and is NOT a fix:
`UNIT` is mallocng's allocation granularity and is woven through the size
classes, `IB` and the group layout. That changes the allocator's behaviour, not
its portability.

The real blocker is in `get_meta`:

    const struct group *base = (const void *)(p - UNIT*offset - UNIT);
    const struct meta *meta = base->meta;

mallocng recovers its metadata by walking BACKWARDS out of the user pointer and
reading in-band bytes at `p[-2]`, `p[-3]`, `p[-4]`. A capability bounded to the
allocation cannot do that. `libc-ext/malloc.c` already records exactly this in
its header comment, which is where anyone should look first.

## What makes the port possible

`sizeof(void*)` is 16 but `sizeof(uintptr_t)` is 8. Addresses are cheap;
authority is separate. That asymmetry gives the rule:

> In-band heap metadata carries ADDRESSES, never capabilities. Authority comes
> from one allocator-private arena capability.

Concretely, four edits:

    #ifdef __capstone
    extern unsigned char *__mng_arena;
    static inline void *A(uintptr_t a) { return __mng_arena + (a - (uintptr_t)__mng_arena); }
    #else
    static inline void *A(uintptr_t a) { return (void *)a; }
    #endif

    uintptr_t meta;                          /* was struct meta *meta */
    char pad[UNIT - sizeof(uintptr_t) - 1];  /* 16-8-1 = 7, upstream's layout */

    const struct group *base = A((uintptr_t)p - UNIT*offset - UNIT);
    const struct meta  *meta = A(base->meta);
    const struct meta_area *area = A((uintptr_t)meta & -4096);

plus three `(uintptr_t)` casts where the field is written (`donate.c`,
`malloc.c` x2). `UNIT` stays 16, so size classes, strides and space overhead are
upstream's unchanged.

**The fourth edit was missed on the first attempt.** All six files compiled
without it and the port would have faulted at `area->check` on the first free.
`--strict-ptr` found it; the clean compile did not. See the round-trip note.

## Why the user pointer can then be bounded

`free(p)` no longer needs `p` to reach anything: `cap_get_base(p)` is one
instruction and no memory access, and every derivation from that address goes
through `A()`. So the returned capability may be shrunk to its own allocation and
free still works. That is strictly stronger than upstream, whose header is
corruptible by a 4-byte underflow; here it is not reachable at all.

## What the port does NOT give

- **No revocation.** mallocng's group model shares one derivation across 32
  slots. Per-allocation SPLIT is what `revoke_on_free_alloc.h` does, and it
  fragments monotonically because SPLIT is one-way. Real trade-off, not an
  oversight: mallocng buys space and time behaviour, `rof` buys temporal safety.
- **Arena-wide authority inside the allocator.** Same as upstream. Not a
  regression, not a win.
- **`glue.h` is the next problem and it is not mallocng's.** Line 49 casts a
  `size_t` from the aux vector to `char *`. Integer to pointer cannot produce a
  valid capability; the loader has to hand over capabilities.
- **No mmap in a domain**, so allocations over `MMAP_THRESHOLD` (131052) need
  another path.

## Validating it on x86 BEFORE touching Capstone

The port is written so that on a non-capability target `A()` is the identity and
`uintptr_t` is pointer-width, which makes the patched allocator layout-identical
and semantically identical to upstream. So the restructuring can be tested where
a working reference exists.

Three musl 1.2.5 trees, built natively, libc-test run against each:

| build | libc-test FAILs | REPORT |
|---|---|---|
| stock | 16 | 438 tests, 498 lines |
| port | 16 | **byte-identical to stock** |
| port + one-token sabotage | **66** | 50 new segfaults |

The sabotage is a plausible off-by-one in exactly the line the port rewrites:
`A((uintptr_t)p - UNIT*offset)`, dropping the `- UNIT`.

**The negative control failed on the first attempt, and the reason was mine.**
`cp -r port broken` copied the already-built objects, `make` judged them current
and never recompiled the header, and the two libraries came out with the same
md5. A control that reports "no difference" because it built the same binary
twice looks exactly like a passing test. Rebuild from a fresh tree; compare the
md5 of the artifacts before believing any differential.

## What this proves, and what it does not

PROVES: the restructuring does not change mallocng's behaviour, over 438 tests,
with the check demonstrated able to fire.

DOES NOT PROVE: anything about capability correctness. On x86 `A()` is the
identity, so the differential cannot see whether `A()` is the right abstraction
for Capstone, nor whether the bounded user pointer actually works. That is a
different question and needs the Capstone-side tests: the assert-enabled soak
(mallocng's `get_meta` is a full heap-invariant checker on every free, which is
the best heap oracle available here), and the directed capability probes
(`p[-1]` must fault, `p[n]` must fault, a stored capability must survive
realloc). Those need the resumable domain runner, because a fault ends the boot.
