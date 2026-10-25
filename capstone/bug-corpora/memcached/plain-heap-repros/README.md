# memcached plain-heap spatial defects — the not-nested row

Upstream memcached defects whose access crosses **the `malloc` bound itself**. A sibling of
[`../allocator-repros`](../allocator-repros/README.md), deliberately separate: that corpus's
boundary is memcached's *nested* allocators — `slabs.c` and the per-thread object cache — and all
eight of its cases cross a bound those own. These cross the system allocator's own bound, so there
is no inner layer to port and no slab geometry to derive.

**Why the corpus exists.** The not-nested spatial row of
[`docs/ref/spatial-and-temporal-bug-inventory.md`](../../../docs/ref/spatial-and-temporal-bug-inventory.md)
stood empty, and the reason given for it did not hold. The hunt had required its candidates to be
**live at the pin**; no document asks for that, and **27 of this tree's 33 cases carry
`live_in_pin: false`**. The convention is stated at
[`../allocator-repros/README.md:132-135`](../allocator-repros/README.md): a fix that precedes the
pin is reconstructed by running the pre-fix consumer shape against the shipped allocator. Liveness
is a field recorded in the case, not a gate on building one.

## Shapes

| shape | cases |
|---|---|
| a terminator written one byte past an allocation sized to the exact input length | 0 |
| a terminator written past an allocation whose RESERVED HEADROOM is one byte short | 1 |
| a realloc sized in bytes where the capacity is counted in elements | 2 |
| an array of pointers sized by the object the pointers point at | 3 |
| a buffer sized for its payload with an uncounted trailer appended | 4 |
| a guard reserving two bytes before a copy that writes three | 5 |
| a length computed as a pointer difference that underflows | 6 |
| a counted string printed with a %s conversion | 7 |
| a shuffle whose bound is decremented after the loop | 8 |

**Both crossings are a terminating NUL, and they are not duplicates.** In case 0 the *size argument*
is wrong — the buffer is `calloc(1, sb.st_size)` where the content needs `sb.st_size + 1`. In case 1
the size is right and the *headroom guard* is one short: it reserves five bytes for `"END\r\n"`,
which is five characters, and forgets that the marker is written with a terminator because the buffer
is returned as a C string. Different error, different place, and each is a one-term reversal of its
own fix. Each case's `distinguishing` field says so.

**Case 1's upstream fix is cited by HASH ONLY**, deliberately: its subject line names an outside
contributor, and a person's name must not enter a committed file in this tree. That constraint is
also recorded at `../../../docs/ref/memcached-spatial-defect-triage.md:43-45`, and it is why the row
sat undispositioned while the other two class-A candidates were resolved.

**The pool is small and the reason is structural, not a shortfall.** Searching memcached's whole
history for spatial wording gives **16** commits before the `1.6.45` pin and **0** after it — the pin
*is* `origin/master`. Of those 16, **8 are `proxy`/`mcplib`**, whose buffers are not the slab, cache
or plain-malloc boundary any corpus here covers, so they are a different corpus's material. Of the
remaining 8, one is this case, one is a DoS/memory-growth fix with no bound crossed (`7eeac6e`), two
are integer-overflow-of-a-length with the spatial consequence downstream (`229cf41`, and `e3b7d33`
which is temporal), and one is `e8364b5`, whose buffer is `do_cache_alloc`'d and therefore belongs to
`../allocator-repros`, not here. **memcached will not reach FFmpeg's 15, and that is a fact about a
25-kLOC server.**

## The cases

| case | upstream | the crossing |
|---|---|---|
| **0** | `ddee3e2` `authfile.c` | `fgets`'s terminating NUL at offset `sb.st_size` of a `calloc(1, sb.st_size)` — **one byte past the allocation**. The fix is `+ 1`; our pin carries `+ 2` |
| **2** | `391f2e4762bf` `memcached.c` | `realloc(freesuffix, freesuffixtotal * 2)` sizes the array in BYTES while the capacity is set in POINTERS -- the next store is at byte 64 of a **16-byte** allocation, **48 bytes past** |
| **3** | `16a809e2a062` `cache.c` | `calloc(initial_pool_size, bufsize)` sizes a `void**` freelist by the CACHED OBJECT's size -- with a 1-byte object, 64 promised slots get **64 bytes** and `ptr[63]` is **440 bytes past** |
| **4** | `40aff8b0f113` `memcached.c` | `malloc(server_statlen + engine_statlen)` leaves `ptr` exactly at the end, then the six-byte `END\r\n\0` trailer is written there -- **6 bytes past** a **64-byte** allocation |
| **5** | `49f3b0ca9b57` `memcached.c` | `memcpy(c->wbuf + len, "\r\n", 3)` plants the literal's NUL at index `wsize` when `len + 2 == wsize` -- **1 byte past** a **16-byte** allocation |
| **6** | `212c3820c7bb` `proxy_lua.c` | `memchr(t1, conf[1], remain)` finds the OPENING tag, so `*newlen = t2 - t1 - 1` underflows to **SIZE_MAX** and the hasher reads unboundedly past a **16-byte** key |
| **7** | `0f605245cf3f` `memcached.c` | `fprintf(stderr, "Deleting %s\n", key)` scans a key that has no terminator -- **past the end** of a **16-byte** allocation |
| **8** | `fa51ad8452d5` `slabs.c` | `slab_list[x+1]` with `x < slabs` and `slabs == list_size` reads index `list_size` -- **one element (8 bytes) past** a **16-byte** array |

## Measured, 2026-10-06

[`results/20261006-native-plain-heap/`](results/20261006-native-plain-heap/result-lines.txt). Both
native arms, two-sided:

| arm | buggy | fixed |
|---|---|---|
| `native-fix-differential` | `cap=9 touched=9 crossed=1` → DEFECT-REPRODUCED | `cap=10 touched=9 crossed=0` → FIXED |
| `native-detect` (ASan) | **`heap-buffer-overflow`, WRITE of size 1, 0 bytes after 9-byte region** | silent, exit 0 |

The arms differ by exactly one term — the `+ 1` in the allocation size. The line, its length and the
write offset are identical in both, so a changed reading is attributable to the fix and to nothing
else.

**ASan reports this one, and that is the contrast the corpus draws.** The sub-object corpora record
ASan *blind*, for the opposite reason: there the crossing stays inside a single allocation, so no
redzone sits where it lands. Here it leaves the `malloc` bound, and a redzone sits exactly there.
Put beside each other the two readings are the project's own axis, measured rather than argued: a
crossing that leaves the allocation is seen by every tool; one that stays inside is seen by none.

## Arms not measured here

`spatial`, `sublet` and `cheribsd-revocation` are **declared predictions**,
each with its mechanism, because this corpus has no Capstone-domain runner yet. Two of them are
worth stating plainly:

- **`spatial` and `sublet` are predicted to FAULT**, and the corresponding reading already exists in
  the port rather than only in a prediction: memcached app **fixture 20** carries this defect's
  shape and reads `level0` RETURN, `shrink` FAULT `oob`, `sublet` FAULT `oob` in
  [`ports/memcached/app/results/2026-10-05-qemu-classa-fixtures/`](../../../ports/memcached/app/results/2026-10-05-qemu-classa-fixtures/README.md).
- **`cheribsd-revocation` is predicted to CATCH.** It is the first spatial row in either corpus
  family predicted caught by stock CheriBSD, and the reason is structural: CHERI bounds each
  `malloc`, and this crossing leaves the `malloc` bound rather than staying inside a slab page or a
  struct. Revocation is irrelevant to it; the bounds are not. A miss would refute the bounds claim
  rather than add a data point.

## Running it

```sh
bash runners/run-native.sh [OUT_DIR]
```

Exit 0 means the plain pair reproduced **and** the sanitiser fired on the buggy arm **and** stayed
silent on the fixed one; any one of the three missing makes it non-zero. Exit 75 is an
infrastructure failure and is never a verdict.
