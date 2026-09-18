# CPython pymalloc defects — inventory at the 3.13.7 pin

*Which upstream CPython defects are usable as specimens for the nested-allocator
corpus, which allocator layer each one's memory came from, and how many the
existing pymalloc port can reach. Assembled 2026-09-18 against
`ports/cpython/pymalloc/upstream.json` = 3.13.7.*

## CPython stacks three allocators, and the port covers the middle one

```
object
 └─ per-type free list   (ten of them, WITH_FREELISTS)   <- a free stops HERE
     └─ pymalloc/obmalloc (<=512 B: arena -> pool -> block)  <- or here
         └─ malloc / mmap                                <- only arena churn arrives
```

Verified rather than assumed:

- `float_dealloc` calls `_PyFloat_ExactDealloc` — the free list — **instead of**
  `tp_free`, which is what would have reached pymalloc. A freed float never
  reaches the small-object allocator at all.
- `pycore_freelist.h` names ten: floats, tuples, lists, dicts, dictkeys, slices,
  contexts, async_gens, async_gen_asends, object_stacks.
- `Python/pyarena.c` is a fourth, separate arena for parser/AST nodes, released
  wholesale — structurally like ggml's contexts.
- Free-threaded builds add mimalloc, which the port puts out of scope.

So CPython is **blinder than PostgreSQL**, not less: PostgreSQL has one layer
between the object and `malloc`, CPython has two.

## The 512-byte question, answered

`pymalloc` serves requests up to `SMALL_REQUEST_THRESHOLD` (512); above that
`PyObject_Malloc` falls through to `malloc`, where ASan **does** see a
use-after-free. That matters, and it is not only about `PyObject_Malloc`:
`PyMem_Malloc` reaches pymalloc too, because `obmalloc.c` sets
`PYMEM_DOMAIN_MEM` to `PYMALLOC_ALLOC` in the default build. `bytearray`
allocates its buffer with `PyMem_Malloc`, so its buffer is pymalloc memory when
it is small.

The filter therefore **removes no case**. It constrains the driver: allocate at
or below 512 bytes, or the specimen silently stops being blind. For the
buffer-carrying defects — the decompressors' `next_in`, `bytes.join`,
`bytearray` — the *same defect* is invisible on a small input and ASan-visible
on a large one. That is worth stating in those cases rather than choosing a size
quietly.

## Upstream says it plainly

`Doc/using/configure.rst`, on `--with-address-sanitizer`:

> "To improve ASan detection capabilities you may also want to combine this with
> `--without-pymalloc` to disable the specialized small-object allocator **whose
> allocations are not tracked by ASan**."

CPython documents that you must **switch its allocator off** for the sanitizer to
work. PostgreSQL's answer to the same problem was to hand-write Valgrind mempool
annotations; CPython's is to tell you to disable the allocator. Both are the
project conceding that a nested allocator is opaque to generic tooling — one by
working around it, one by turning it off.

## What is live in the 3.13.7 pin

The 3.13 branch is at v3.13.15, eight patch releases past our pin. Two disjoint
groups, proven live in different ways:

| | count | how it is known to be live |
|---|---|---|
| **A** fixed on the 3.13 branch after `v3.13.7` | **26** | the backport itself: a fix cherry-picked into 3.13 says the defect was in 3.13 |
| **B** fixed on main, never backported to 3.13 | 18 | nothing, until tested — see *Group B, measured* below |
| total | 44 | |

Triaged on three axes — consumer-side C defect, no true concurrency, and which
allocator layer the freed object came from:

| | count |
|---|---|
| no C code (docs, tests, build) | 4 |
| needs true concurrency | 4 |
| the allocator itself, not a consumer | 0 |
| **PyArena** — no port | 1 |
| **type free list** — no port | 2 (a lower bound, see below) |
| **pymalloc — covered by the port** | **32** |
| unclear | 1 |

Of those 32, **21 come from group A** and are proven live by their backports.
The other 11 are group B and were tested separately, below; 2 of them survive.
The number to quote is **23 reachable today**, against PostgreSQL's 8.

The recurring shape is **re-entrancy**: a Python callback runs inside a C
operation and drops the last reference to something the C code still holds.
`bytes.join` via `__buffer__`, `os.spawnv` via `__fspath__`, `Context.__eq__` via
`ContextVar.set`, `OrderedDict.copy`, `itertools.groupby` and `_grouper`, the
JSON encoder and decoder, `TextIOWrapper.tell` with a re-entrant decoder.

## Group B, measured

The 11 group-B cases in the pymalloc bucket were tested rather than assumed, the
same way the PostgreSQL revert pool was sized: extract the upstream fix and ask
whether it still applies to a pristine `v3.13.7` tree. If it applies, the pre-fix
code is there verbatim and the defect is live.

**With a control, because a test that never says "applies" would produce this
same table.** The same test was run over the 21 group-A cases, whose answer is
known independently — those fixes were cherry-picked onto the 3.13 branch at or
just after our tag, so they *should* apply:

| | cases | fix applies to `v3.13.7` |
|---|---|---|
| group A (control) | 21 | **21** — every one |
| group B (the question) | 11 | **1** |

The control fires, and the two groups separate completely. Script:
`cpy-313-apply.py`, run 2026-09-18.

### But a NEGATIVE from that test is not evidence of absence

The apply test is **sufficient, not necessary**: it proves live when it applies,
and proves nothing when it fails, because the surrounding code may simply have
moved. Reading the ten failures against the 3.13.7 source rather than counting
them:

| case | verdict at `v3.13.7` | how |
|---|---|---|
| `gh-145244` json encoder, borrowed dict key | **LIVE** | `Modules/_json.c:1621` runs `while (PyDict_Next(dct, &pos, &key, &value))` and passes the borrowed `key`/`value` straight into `encoder_encode_key_value` with **no `Py_INCREF` at all** — not even the free-threaded one the pre-fix main had. The function was renamed on main, which is the only reason the patch does not apply |
| `gh-154189` `functools.partial_vectorcall` | unresolved | the same borrowed-pointer shape is present — `pto_args = _PyTuple_ITEMS(pto->args)` then a call through `pto->fn`, no strong references — but the 3.13 trigger path has not been read |
| `gh-116946` `_tkinter` GC protocol | unresolved | partly a cycle fix; its temporal half ("deallocation cancels any pending timer so its callback cannot run on freed memory") plausibly applies, but driving it needs Tcl/Tk and an event loop |
| `gh-142349` lazy-import specialization | not live | no `lazy_import` in 3.13.7 |
| `gh-153570` `bytearray.take_bytes` | not live | no `take_bytes` in 3.13.7 |
| `gh-154751` `curses.initscr` after `newterm` | not live | no `topscreen`, `PyCursesScreenObject` or `_curses_newterm_impl` in 3.13.7 |
| `gh-155864` `use_screen` | not live | symbol absent |
| `gh-155875` `new_prescr` | not live | symbol absent |
| `gh-126703` pycfunction freelist | wrong layer | belongs to the free-list bucket, not pymalloc; see below |
| `GH-127705` better double-free message | noise | changes message text only |

So group B contributes **2** live cases — one by the apply test
(`local_timezone_from_timestamp`), one by inspection (`gh-145244`) — with **2
unresolved** that could raise the total to 25.

The 3.10 exercise predicted roughly this: of 75 fixes that never reached 3.10,
only 9 still applied, about 12%. Group B here scores 1 of 11 on the same test and
2 of 11 after reading. "Never backported" usually means the code did not exist
yet, and it has now been measured on this pin rather than borrowed from that one.

## Two remaining weaknesses, stated so the numbers are not over-read

**The free-list count is a lower bound.** Layer assignment uses the file the fix
touches, which is a proxy. `gh-126703` — "use after free in pycfunction freelist"
— landed in the pymalloc bucket because it is in `methodobject.c`, and its own
title names the layer. `Context.__eq__` is likely the same. Expect more than 2.

**One entry is noise that the C-code filter let through**: `GH-127705 better
double free message` only changes message text. It is in group B and its fix does
not apply, so it never reached the 23 — but a filter that lets it through will
let others through, and only group B has been read case by case. The 21 in group
A are trusted on their backports and their apply-test result, not on having each
been read.

## Why an older pin does not help here

Measured, because the PostgreSQL intuition does not carry over:

| pin | never backported | of those, fix still applies to the pin |
|---|---|---|
| 3.10.0 | 70–75 | **9** |
| 3.11.0 | 60 | not measured |
| 3.12.0 | 46 | not measured |
| 3.13.0 | 16–18 | — |

CPython moves fast enough that "never backported" is dominated by code that did
not exist. And the port would have to be redone: `obmalloc.c` differs from our
pin by **+1462/−979** against 3.10 and **+934/−85** against 3.12, and of the
port's three patches only `0002` still applies to 3.12.0 — `0001` and `0003`
fail.

**Stay at 3.13.7.** It yields more verified cases than any older pin and costs
nothing.

## Reaching further

Eight of the 23 now have drivers and have been run:
`bug-corpora/cpython/pymalloc-repros/results/20260918-qemu/` — 16/16 arms, with
a negative control that fires on both oracles. Fifteen do not.

| | reachable | needs |
|---|---|---|
| pymalloc port (exists) | 23 (+2 unresolved), 8 with drivers | — |
| + type free lists | 2 and more, `gh-126703` among them | porting the ten free lists |
| + PyArena | 1 | porting the parser arena |

The free-list layer is the more interesting extension: it sits **above**
pymalloc, a free stops there and never reaches it, and we demonstrate that
blindness nowhere else.
