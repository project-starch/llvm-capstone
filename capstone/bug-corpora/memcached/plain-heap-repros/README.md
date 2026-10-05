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

## The case

| case | upstream | the crossing |
|---|---|---|
| **0** | `ddee3e2` `authfile.c` | `fgets`'s terminating NUL at offset `sb.st_size` of a `calloc(1, sb.st_size)` — **one byte past the allocation**. The fix is `+ 1`; our pin carries `+ 2` |

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

`spatial`, `sublet`, both PoisonCap arms and `cheribsd-revocation` are **declared predictions**,
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
