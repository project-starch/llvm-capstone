# Perl 5.36.3: spatial and temporal defects live at the pin

**Scope: spatial and temporal memory defects only.** That is this study's subject,
and it is the only thing this corpus admits. A defect that is a null dereference, a
type confusion, an integer overflow with no out-of-bounds consequence, or a crash
from stack exhaustion is not here, however real it is.

Eleven cases. Each is a defect upstream Perl fixed after 5.36.3, whose fix never
reached the 5.36 maintenance branch, and whose trigger is the test upstream added
with that fix.

## Liveness is measured, in both directions

A case is here only because the measurement puts it here:

| | at the pin (5.36.3 + ASan) | on master (`v5.45.3-85-gdb19522155` + ASan) |
|---|---|---|
| reports | **11 of 11** | 0 of 11 |
| silent | 0 of 11 | **11 of 11** |

Same trigger, same harness, same `ASAN_OPTIONS`, two builds. Full table in
[`results/20261006-host-differential`](results/20261006-host-differential/).

**The dates do not settle this, and would get four cases wrong.** Four of the eleven
fixes are *dated before* the `v5.36.3` tag (2023-11-28) and are still absent from it:
they landed on blead and were never backported. So each case records the ancestry that
was actually checked — `git merge-base --is-ancestor <fix> v5.36.3` false,
`--is-ancestor <fix> <blead>` true — rather than a date comparison.

## What the pin's own ASan can and cannot see

This is the finding that motivates the nested-allocator arm, and it is visible in the
host measurement before any capability hardware is involved:

| what the pin reports | cases | where the memory lives |
|---|---:|---|
`heap-use-after-free` | 3 | `malloc`/`realloc`/`calloc` — the **system allocator**, so ASan holds the allocation record |
`SEGV on unknown address`, no allocation context | 5 | a stale `SV*` or a wild pointer; ASan does not know this memory |
Perl's own refcount check | 2 | the SV head arena; ASan silent |
wrong bytes only (`$&` comes back as `b\0`) | 1 | a stale COW `subbeg`; both oracles silent |

**Three of eleven are visible to ASan.** The other eight are in Perl's own arenas, or
are derefs of a pointer that never had an allocation record. A sanitizer that works at
the `malloc` boundary cannot see a lifetime that never crosses it.

## The arms

Four measured arms. `sysalloc-bounds` is the **baseline**: per-object heap bounds are
what an application gets by default since PR #170, so a catch is a case the bounds arm
does **not** fault on.

| arm | what it is | catches |
|---|---|---:|
| `sysalloc-none` | the first-fit heap with `-DCAPSTONE_LEVEL0_OBJECT_BOUNDS=0`: every pointer carries the whole arena's bounds | 6 |
| **`sysalloc-bounds`** | the same heap as applications get it today. **The baseline** | **6** |
| `sysalloc-sublet` | the Sublet heap as the **system** allocator: bounded, and every free revokes | 6 |
| `sublet-svheads` | the baseline's `level0` **outer** heap, plus the SV head arena through the lifetime adapter of [`../../../ports/perl/sv-heads`](../../../ports/perl/sv-heads) in `PERL_SUBLET_MODE=1` | **7** |

The images are one source each and differ only in the heap they link, so a difference
between them is the heap's protection and nothing else. Note what the sv-heads arm is
*not*: its outer allocator is `level0`, the same as the baseline's, which is `build.py`'s
own rule for a nested-allocator arm (`--nested` with `--heap sublet` is a `ValueError`).
Otherwise two things would change at once and the comparison would measure whichever
one you did not intend.

The one addition over the baseline is `05_17535c984a`, a cloned constant sub whose
`CvXSUBANY` SV head is shared without a reference — a head in the arena the adapter
owns. The spatial case `08_b7b77ffc1e` is caught by all four, but the three bounded arms
report it as a **bounds** violation (cause 5) where the unbounded one sees only an
untagged dereference (cause 24): same catch, better precision.

### The denominator is 9, not 11

Two cases do not execute their defect in a domain at all, and scoring them as misses
would be counting the harness:

- `10_254b30e378` carries **`harness_limit`**. A probe on the bounds arm shows the
  in-memory filehandle it needs accepts its writes and throws them away: `open` returns
  1, `print` returns 1, and the backing scalar is still `undef`, where the host build
  holds the full 52-byte string. The overlapping `Move` that *is* the defect is never
  reached.
- `03_9e298ab597`'s trigger **passes its own assertion** on all four arms, so the stale
  element slot it depends on never holds a stale pointer there.

Of the nine that do exercise their defect, **seven are caught**. The other two are not
gaps in the hardware:

- `11_af11b0c528` is **out of reach by construction**. The stale `subbeg` length makes
  `$&` read one byte past the logical string into the NUL terminator, which lies *inside
  the same allocation*. No per-object bounds and no revocation can see an in-bounds read
  of a live object; the only oracle is the wrong output, which every arm does produce.
- `02_d2cddbe1df` is a **deliberate carve-out**. Upstream Perl reads a freed head's flags
  on purpose, so the adapter answers `SvIS_FREED` and `SvTYPE` from its sidecar without
  touching the head; making that path fault stops three upstream test files. The defect
  is still detected — by Perl's own `panic: attempt to copy freed scalar`.

### The arms carry their own positive control

Three arms reporting the same six is also exactly what a build whose revocation never
fires would show, so `sysalloc-sublet`'s silence needed proving to be a fact about the
cases rather than about the arm.
[`results/20261006/heapprobe.c`](results/20261006/heapprobe.c), compiled with each arm's
own SDK and run as a domain, does `malloc`, `free`, then a read at an argv-derived index
whose value is printed so the load cannot be folded away:

| | `sysalloc-bounds` | `sysalloc-sublet` |
|---|---|---|
| read after `free` | `READ-FREED-OK got=Z` — completes | the process ends at the read |
| `realloc` 26→52, 1000→4000 | `moved=1` | `moved=1` |

So revocation is live in the sublet arm, `l0_free` really does only mark (it sets
`b->free = 1` and coalesces), and a dangling pointer into a grown buffer *is* possible
here — "realloc grew in place" was considered and refuted. `sysalloc-sublet` adds nothing
over the baseline because the two system-allocator temporal cases it exists for are the
two that do not execute.

### The fourth arm is missing, and why

`cheribsd-revocation` is **not measured**. The static purecap interpreter faults with
SIGPROT inside `perl_construct`, the stock loader refuses the dynamic one, and the only
platform that loads it has an `sshd` that dies on a poison exception mid-copy — while
mruby's static purecap binary runs on that same image. The localisation and what would
settle it are in
[`results/20261006-cheribsd-blocked`](results/20261006-cheribsd-blocked/). The mruby
corpus has four measured arms; this one has three.

**`sublet-svheads` protects exactly one of Perl's nested allocators.** The SV *bodies*,
the hash entries and the OP slabs keep their upstream allocators. A case whose stale
pointer lives in those is outside that arm's reach by construction, and its arm entry
says so rather than being scored as a miss.

## Layout

```
NN_<fix>_<slug>/
  case.json        the schema's fields, including the measured arm oracles
  trigger.pl       upstream's own test, on the shared harness
  PROVENANCE.md    the fix, the unfixed code at the pin, and both host measurements
harness/shim.pl    plan/ok/is/like/... collecting failures instead of aborting
results/           one bundle per measurement day
```

Every trigger's first line is `require "shim.pl";`, resolved through `@INC`, so the
same file runs on the host, in a Capstone domain and on CheriBSD with nothing but
`PERL5LIB` changing.
