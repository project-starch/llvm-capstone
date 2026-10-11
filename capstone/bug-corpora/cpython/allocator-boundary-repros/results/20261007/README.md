# 2026-10-07 -- allocator_consumed, re-measured on the host

This is not an arm run. No VM, no Capstone, no CheriBSD. It re-derives the one
field the three-arm results are read through -- `allocator_consumed` -- from a
measurement instead of from a judgement, for all 32 cases.

## Why the field needs its own measurement

Every arm's oracle is written in terms of which allocator issues the object. On
`cheribsd-revocation` a pymalloc block is expected to survive, because pymalloc
links the freed block onto its own pool free list and never returns it to libc,
so the platform has no free to revoke. On `sublet` the same block is expected to
fault, because that arm issues and revokes pymalloc's own blocks. The two
oracles disagree *only* through this field. If it is assigned by eye, the arms
are being compared against a guess.

## The test

Two runs of each trigger under one ASan build of the pinned 3.13.7:

| column | environment | what it shows |
|---|---|---|
| `asan_pymalloc` | `PYTHONMALLOC` unset | pymalloc is active. ASan cannot see inside a pool, so a *nested* defect is invisible here |
| `asan_libc_malloc` | `PYTHONMALLOC=malloc` | `obmalloc.c:640-665` routes RAW, MEM and OBJ to libc, so a real heap defect must report |

- heap report in the first column -> the object is **not** a pymalloc block: `no`
- heap report only in the second -> it is: `yes`
- no heap report in either -> the test measured nothing: **inconclusive**, which
  is not the same as `no`

Region size is deliberately not used. Case 07 (gh140594) is a 100-byte object
that is nonetheless non-nested, so size does not separate the two.

## The routing control

The rule above rests on a claim about CPython's domains, so that claim is
measured too, not read off the source. A 16-byte block is allocated, freed and
read back:

| | `PYTHONMALLOC` unset | `PYTHONMALLOC=malloc` |
|---|---|---|
| `PyMem_Malloc(16)` | no report -- a pymalloc block | heap-use-after-free |
| `PyMem_RawMalloc(16)` | heap-use-after-free | heap-use-after-free |

The MEM domain is pymalloc by default (`obmalloc.c:351`, `PYMEM_ALLOC =
PYMALLOC_ALLOC`); only RAW bypasses it. The `PyMem_RawMalloc` row is the control
-- without it, "no report" could as easily mean the instrumentation was dead.

A failed control is why case 01 is not usable as the ASan positive control for
this corpus: it is nested, so ASan is structurally blind to it and reports `SEGV
on unknown address 0x1` from the freelist link. Any ASan control for this corpus
must be drawn from the non-nested cases.

## Result

30 of 32 agree with the value already recorded. No case contradicts it.

Two cases the test cannot resolve, both for a reason that is a property of the
defect rather than of the run, and both recorded in the case's `layer_note`:

- **18 (gh157335)** -- the buffer *is* an mmap mapping (`mmapmodule.c`), so the
  overflowing write reaches a guard page and ASan reports SEGV in both columns.
  Its `allocator_consumed: no` rests on the mapping's provenance. Its
  `allocator_layer` said "libc malloc", which its own note contradicted; it now
  says mmap.
- **32 (gh149449)** -- the fault is an indirect *call* through the dangling
  pointer. ASan instruments loads and stores, not the jump, so both columns give
  SEGV at the PC. Its `allocator_consumed: yes` is instead carried by the
  routing control above: the struct comes from
  `PyMem_Malloc(sizeof(_PyUnicode_Name_CAPI))` = 16 bytes
  (`Modules/unicodedata.c:1465`), and that call was measured to land in a pool.

Standing distribution over the 32: 24 nested, 8 non-nested.
