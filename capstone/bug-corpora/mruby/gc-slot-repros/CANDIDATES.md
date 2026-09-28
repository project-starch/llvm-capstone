# Candidates for this corpus, measured at the pin

This corpus was declared empty with the note that the xlang mruby rows would be
re-run inside the complete interpreter. That note is wrong about what is worth
measuring here, and this file is the measurement that says so.

**The xlang rows cannot exercise this corpus's boundary.** All eleven of them
reach a real `free` or `realloc` at the domain allocator -- `xlang/capstone/rows.tsv`
says so itself in its `alloc_route` column (`stack realloc`, `gc sweep -> free`,
`explicit mrb_free`, or `none` for the two spatial rows), and every one was found
*by* ASan, which can only report a release the allocator saw. Revoke-on-free at
`MRBD_HEAP=sublet` answers all of them; the third arm, `sublet-gc`
(`patches/4.0.0-rc2/0008`, every GC object slot issued and revoked on its own),
is not touched by a single one.

What this corpus needs is the class ASan is structurally blind to: a slot the
interpreter vacates and hands out again without the domain allocator ever seeing
a release. The search below found seven such defects live at the pin, three of
them ASan-silent.

## How the search was run

The pin is `4.0.0-rc2` = `9d523e2f74f2e63ca02840937523de61398a617d` (2026-03-12).
The port's other pin, `head` = `ad98f216eb472202c8e5deece5ea13655d9f7969`
(2026-09-17), is **2471 commits** ahead. `build-mruby-domain.sh` already states why
that window is the interesting one: rc2 is *"the Sublet evaluation's pin: temporal
defects on mruby's own allocators, fixed later"*.

So the window `rc2..head` was swept rather than the history before rc2: a fix in
that window means the defect is **live at the pin**, and needs reproducing, not
backporting. The window names 6 GHSAs and 1 CVE
(`GHSA-jfmr-44fc-gfhg`, `GHSA-2778-fvwg-5m8w`, `GHSA-f3mm-x76x-jmcv`,
`GHSA-pv4h-vpp5-5349`, `GHSA-pmm3-g676-wxm7`, `GHSA-qj89-7wc6-8hfr`,
`CVE-2018-14337`). Of those, three are spatial (`load.c` irep validation,
`mruby-sprintf` buffer, `mruby-socket` negative length -- and that gem is not in
the port's gembox), and one is a receiver type confusion.

Nine of the candidates shipped an upstream test with their fix. Those tests are
the reproduction: extracted from the fix commit's own diff, run against the pin.

Three native builds of the pin, and one control (`probe/`):

| build | flags | what it answers |
|---|---|---|
| `host` | `MRB_DEBUG` | does mruby's own `MRB_TT_FREE` assertion fire? |
| `stress` | `MRB_DEBUG MRB_GC_STRESS` | does it need a collection at every allocation? |
| `asan` | `-fsanitize=address`, assertions off | **is the defect visible to ASan at all?** |
| control | pin + the fix hunks that apply | does the fix flip the outcome? |

The control carries six fixes (`5e8a65457`, `4663fef45`, `a54353ecf`, `08a0432d1`,
`fb4974528`, `859288c19`). Three could not be applied onto rc2 -- `39aecc143`,
`eb7693857` and `606d9a6b2` each depend on an intermediate commit -- so for those
rows the flip is unproven and only the pin's own failure is reported.

## What reproduces at the pin

| # | upstream fix | advisory | client code | pin | control | ASan at pin |
|---|---|---|---|---|---|---|
| 1 | `a54353ecf` | -- | `Hash#[]`, `#delete`, `#key?` | 9 failures, no `hash modified` | PASS | **silent** |
| 2 | `08a0432d1` | -- | `Hash#assoc`, `#rassoc`, `#==`, `#eql?` | 4 failures | PASS | **silent** |
| 3 | `eb7693857` | -- | `Hash#inspect`, `#__except`, `#rehash` | 4 failures | fix n/a | **silent** |
| 4 | `4663fef45` | `GHSA-2778-fvwg-5m8w` | hash scans, `Hash#shift` | 4 failures | PASS | heap-buffer-overflow |
| 5 | `fb4974528` | -- | `Hash#key`, `#slice`, `#slice!` | **SIGSEGV** | PASS | heap-buffer-overflow |
| 6 | `606d9a6b2` | -- | `Hash#merge`, pattern `**rest` | **SIGABRT** | fix n/a | heap-use-after-free |
| 7 | `39aecc143` | -- | `OP_ENTER` short argument list | **SIGSEGV** | fix n/a | heap-use-after-free |

Rows 6 and 7 abort inside mruby's own barriers, at exactly the assertion each fix
commit names:

    606d9a6b2 -> src/gc.c:1411  mrb_field_write_barrier: (value)->tt == MRB_TT_FREE
    39aecc143 -> src/gc.c:846   mrb_gc_mark: Assertion `(obj)->tt != MRB_TT_FREE' failed

`MRB_GC_STRESS` changes two rows and no verdict: it turns row 4 into an abort as
well, and row 7 into a plain failure rather than a fault. The other five answer the
same with a collection at every allocation as without one, so none of the seven
depends on stress to reproduce -- which is what makes them usable as cases.
`probe/survey.sh` prints that column beside the others.

### Rows 1-3 are this corpus's material

They are the three the corpus was declared for, and the only three in the set that
**ASan cannot see**.

The nested allocator is the hash's own entry array. `ar_delete` and `ht_delete`
(`src/hash.c:586`, `:951` at the pin) vacate a slot by marking its key undef and
decrementing the count; `ea_compress` and `ea_resize` (`:454`, `:448`) move entries
inside the block; a later store reuses a vacated slot. All of it happens inside one
`mrb_malloc`'d block, so no release reaches the domain allocator -- the same shape
as the GC's own `page->freelist` recycling at `src/gc.c:1188-1191`.

The client is hash.c's own lookups and scans. Every one of them can re-enter Ruby
through a key's `eql?` or `hash`, and `H_CHECK_MODIFIED` watches the capacity, the
flag bits and the two pointers. A delete followed by an insert that restores the
size moves none of those, so the guard does not fire and the scan answers from an
entry that has been vacated and refilled.

The oracle is a wrong answer, not a fault: at the pin the interpreter silently
answers from the reused slot where the fixed version raises `hash modified`. That
makes these cases cheap to arm -- no capability fault is needed to read the result,
so a case can report next to itself, unlike the corpora whose cases end the domain.

### Rows 4-7 belong with the xlang material

ASan sees all four, so they bottom out in a real release and `sublet` answers them
at the malloc layer. They are worth having as the ladder's lower rungs, but they do
not exercise `sublet-gc` any more than the existing xlang rows do.

## What did not reproduce

| candidate | expected | measured |
|---|---|---|
| `5e8a65457` / `GHSA-jfmr-44fc-gfhg` | hash iteration reads past the entry array | its own upstream test **passes** at the pin, in all three builds |
| `7d626242a` | constant cache keyed on a freed irep's address | see below |

`GHSA-jfmr-44fc-gfhg` is an out-of-bounds *read*, and the test added with the fix
asserts an exception rather than a fault, so a plain build has nothing to report.
It is not established here that the defect is absent at the pin -- only that this
test does not show it. The instrument it wants is the ASan build against a case
that dereferences what the scan read, which is not written.

`7d626242a` is the most interesting candidate in the window and the least settled.
The cache behind `OP_GETCONST` is keyed by the **irep's address**, and nothing
dropped an irep's entries when it was freed, so a later irep landing at the same
address is answered from the stale entry -- address reuse, with no dereference of
freed memory at all, which is the purest form of what this corpus is about. But the
commit's own reproducer gives `[:top, :top]` from the **first** iteration at the
pin, where the commit describes `[:foo, :top]` first and the wrong answer only from
the second. That does not match, so it is not counted. Settling it needs the C-side
helper the fix added to `mrbgems/mruby-eval/test/eval.c`, which needs a build with
tests enabled; the fix itself does not apply onto rc2 unaided.

## Gems, and what is out of reach

The port's gemboxes carry `mruby-hash-ext`, `mruby-array-ext`, `mruby-method` and
`mruby-set`, so rows 1-7 are all reachable in the domain build.

`GHSA-f3mm-x76x-jmcv` (`859288c19`) is the same shape at three sites -- a value
taken out of the structure that held it and published on the arena afterwards,
where `mrb_gc_protect()` itself can allocate and collect it. Two of the three,
`Hash#shift` and `mruby-method`'s argument shift, are in the port. The third,
`Task::Queue#__pop_try`, is in `mruby-task`, which exists at the pin but is in
none of the port's gemboxes.

## Limits of this measurement

Everything above is a **native x86-64** measurement of vanilla rc2. It establishes
that the defects are live at the pin and which instrument sees them. It does not
say what the Capstone domain does with them: no clang, no QEMU and no board were
used, the port's patches 0001-0003 and 0008 were not applied, and no arm of
`MRBD_HEAP` was run. Turning these into cases means writing each as the corpus
schema wants, then arming `spatial`, `sublet` and `sublet-gc` and measuring.

The prediction worth committing before that runs, in the spirit of
`xlang/capstone/rows.tsv`: rows 1-3 MISS under `level0` and under `sublet`, and
FAULT only under `sublet-gc`; rows 4-7 MISS under `level0` and FAULT under both
`sublet` and `sublet-gc`. If rows 1-3 also fault under `sublet`, the entry array is
being reallocated where this reading says it is reused, and the reading is wrong.
