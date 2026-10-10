# CPython allocator-boundary defect corpus

21 memory-safety defects live in CPython 3.13.7, in the three cells
[`pymalloc-repros`](../pymalloc-repros) does not cover. That corpus holds
consumer-side **temporal** defects whose memory comes from **pymalloc** and
states it is the whole reachable set at the pin; these are the others.

Together the two corpora are 41 cases and all four cells:

| | nested (pymalloc) | non-nested (libc malloc) | total |
|---|---:|---:|---:|
| **temporal** | 20 — `pymalloc-repros` | **3** — here | 23 |
| **spatial** | **13** — here | **5** — here | 18 |
| **total** | **33** | **8** | **41** |

## What nested and non-nested mean here

CPython takes a 1 MiB arena from the OS, cuts it into 16 KiB pools and hands out
blocks of 512 bytes or less (`SMALL_REQUEST_THRESHOLD`,
`Include/internal/pycore_obmalloc.h:157`). A use-after-free or an overflow
between two of those blocks never crosses the boundary of that one `mmap`, and
`free()` never reaches `malloc`, so a malloc-level tool has no event to see.
That is **nested**. **Non-nested** means the object came straight from libc
`malloc`, with no second allocator in between.

Two routing facts decide which applies, both read off the pinned source:

* `Objects/obmalloc.c:351` sets `PYMEM_ALLOC = PYMALLOC_ALLOC`, so the MEM
  domain is **not** a separate allocator. Only `PyMem_RawMalloc` bypasses
  pymalloc.
* Requests over 512 bytes go to libc `malloc` even through the OBJ and MEM
  domains.

## The side is measured, not read off the size

The obvious shortcut -- "ASan reported an N-byte region, N ≤ 512, so it is
nested" -- is wrong, and case 07 is why. `gh-140594` overflows a **100-byte**
region and is non-nested: `PyOS_StdioReadline`'s buffer is `PyMem_RawMalloc`'d
(`Parser/myreadline.c:207,223,325,347`), so libc owns it at any size. Under
`PYTHONMALLOC=malloc` every allocation is a libc malloc, so ASan reports a
region for everything and the size cannot tell the domains apart.

So each trigger is run **twice** and the question is whether the heap report
survives the nested allocator:

| under `PYTHONMALLOC=pymalloc` | verdict |
|---|---|
| ASan still reports a **heap** violation | the object is not a pymalloc block → **non-nested** |
| no heap report | the damage stayed inside the arena → **nested** |

A crash without a heap report in the pymalloc arm is **not** a missed
measurement. The stale read returns the freelist link pymalloc writes *inside*
the freed block (`Objects/obmalloc.c:2494`), so the program dies on that
garbage while ASan stays quiet about the heap. That is the mechanism evidence
for "consumed inside the arena", and it is what `allocator_consumed` records.

Both halves of axis 2 are present on every case: `allocator_layer` says which
allocator the memory came from, `allocator_consumed` whether the damage stayed
inside one of its blocks.

## Liveness

Every case is live at the pin and the proof is a measurement, never an
assertion: the trigger is upstream's own reproducer from the issue, and running
it on the pinned 3.13.7 ASan build produces the sanitizer report quoted in
`live_proof`. All 21 reproduce from their case directory with the oracle
recorded for them.

Two things that had to be got right for those runs to mean anything:

* `ASAN_OPTIONS=detect_leaks=0`. With leak checking on, the leak summary buries
  the report and a reproducing case reads as silent.
* A positive control before each batch. An earlier run of this corpus capped
  address space with `ulimit -v`, which stops ASan reserving its ~15 TB shadow
  region; 84 cases came back "no violation" when the instrument had never
  started.

## Arms

The three arms are named as `pymalloc-repros` names them, so the two corpora
read side by side:

| arm | what it is |
|---|---|
| `spatial` | base Capstone: the level0 heap with per-object bounds, which is what applications get since PR #170 |
| `sublet` | the same heap plus patch 0014, so pymalloc's pools and arenas are issued and revoked too |
| `cheribsd-revocation` | CheriBSD purecap, libc revocation as the platform ships it |

**All three have been run.** `results/20261006/` holds the combined
`matrix.tsv`, an `inputs.json` carrying every run's image hash and capacity
variant, and the guest's `security.cheri` subtree read before and after each
CheriBSD batch. All three arms are reported over the **same 29** of the 32
cases; the three outside that set, and why no arm reports them, are in the
results README.

| arm | detected / scored | of the detections, the mechanism |
|---|---:|---:|
| `spatial` | 14 / 29 | 14 |
| `sublet` | **27 / 29** | 27 |
| `cheribsd-revocation` | 11 / 29 | 11 |

A fault whose cause is not one the capability hardware raises is not counted as
a detection: case 32 on `spatial` raises `cause=2`, `RISCV_EXCP_ILLEGAL_INST`,
which is the CPU trapping a call into non-code and not this arm's mechanism.

### Pre-registered expectation, written before the runs

Recorded here so the result can contradict it rather than be explained by it:

1. **`spatial` and `cheribsd-revocation` should catch the same cases.** Both
   bound each allocation and neither revokes anything pymalloc does, so a
   nested defect should be invisible to both and a non-nested one visible to
   both. If they differ, the difference is the finding, not a footnote.
2. **`sublet` should catch strictly more**, and the cases it adds should be the
   nested ones: that arm is the only one that issues and revokes pymalloc's own
   blocks.
3. The 8 non-nested cases should be caught by all three, because no nested
   allocator stands between the defect and the mechanism.

Each of these can fail. In particular (2) predicts a *strict superset*; a case
that `sublet` misses and `spatial` catches would contradict it outright.

### How those three held up

1. **Held in shape, and for a reason the expectation had wrong.** `spatial` 14
   and `cheribsd-revocation` 11 are close, but not because neither revokes:
   `CAPSTONE_REVOCATION_ENFORCE` defaults to 1 and is independent of the sublet
   discipline, so the base arm revokes too, and 11 of its 14 detections carry
   `cause=24`. The arm is not the bounds-only baseline this expectation assumed.
2. **Held.** `sublet` 27 against 14 and 11, and its 29 is the same 29. No case
   is caught by `spatial` and missed by `sublet`, so the strict superset stands.
3. **Failed, on two arms.** The non-nested cases are 04, 07, 08, 12 and 18.
   `spatial` and `sublet` each catch 4 of the 4 in the delivered set, but
   `cheribsd-revocation` catches 3 of 4 -- it is silent on 04, where the
   overread stays inside the bounds its malloc actually set, since that guest
   rounds capability bounds up to a size class -- and on the three non-nested
   TEMPORAL cases `spatial` catches 0 of 3 and `cheribsd-revocation` 1 of 3. A
   non-nested defect is not visible to every arm, and the reasons differ per
   arm: allocator rounding on CheriBSD, and on base a revocation that is
   sweep-free but still only acts where the system allocator freed.

## Fidelity, stated as a limitation

The triggers are upstream reproducers driven through the real interpreter, not
C reductions against the allocator the way `pymalloc-repros` does it. They
establish that the defect is live and which allocator owns the object. They do
**not** establish which arm ought to catch it.
