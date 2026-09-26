# Perl as a Capstone domain, and its allocators under Sublet

*Evaluation, 2026-09-26, measured against perl 5.38.2, with 5.36.3 and 5.32.1 read
for comparison, and perl-cross at `c2d8f8b`. The question is whether perl is a
worthwhile third interpreter port after mruby, what a per-object Sublet arm would
have to change, and which release to pin.*

## What this answers, and what it does not

This covers the **allocator structure and the port cost**, both measured in the
source. It does **not** count perl's later-fixed temporal defects per release,
which is the other half of a pin decision and the half that ranked mruby's
releases (`ports/mruby/musl/README.md`). That count has to come from the team;
everything below is written so the count can be dropped into it.

## Perl's own allocators: four layers

| Layer | What it manages | Where | Reuse visible to ASan? |
|---|---|---|---|
| **SV heads** | one `SV` struct per value; freed heads go on `PL_sv_root`, a free list **threaded through the freed head itself** (`SvARENA_CHAIN_SET(p, PL_sv_root)`, `sv.c:263`) inside a 4080-byte arena chunk (`PERL_ARENA_SIZE`, `sv_inline.h:28`; `Newx(chunk, PERL_ARENA_SIZE, char)`, `sv.c:277`) | `sv.c`, `sv_inline.h` | **no** |
| **SV bodies** | the type-specific body (`XPVIV`, `XPVAV`, ...), one arena per type, free list per type in `PL_body_roots[]`, threaded through the freed body (`sv.c:757-763`) | `sv.c`, `sv_inline.h` | **no** |
| **OP slabs** | compiled ops, carved from slabs with a free-space cursor and a freed-op list (`Perl_Slab_Alloc`, `op.c:194-267`) | `op.c`, `op.h` | **no** |
| **perl's own malloc** | optional bucket allocator under everything else, enabled at configure time (`-Dusemymalloc`, `Configure:7848`) | `malloc.c` | **no**, when enabled |
| everything else | string buffers, AV/HV arrays, the stack, HEKs: `Newx`/`Renew`/`Safefree` | all over | yes (plain `malloc`) |

Four nested layers, three of them on by default. That is more than mruby has
(one slot heap plus two bump pools) and the same shape: **a freed slot's own
bytes hold the free-list link**, so the slot is read while free and reissued to
the next object while stale references to the old one still point at it.

## Perl ships the control we need

`-DPURIFY` (or `PERL_ARENA_SIZE=0`) routes **every SV head and body straight to
`malloc`/`free`** and disables the arenas (`sv.c:768-776`, `sv_inline.h:158-164`,
`HASARENA FALSE`). That is an upstream-supported, one-flag matched pair:

- **default build:** a use-after-free on an SV is invisible to ASan, because the
  memory never returns to `malloc`;
- **`-DPURIFY` build:** the same defect is an ordinary ASan heap-use-after-free.

So for every SV defect we can *demonstrate* that the nested allocator is what
hides it, without writing the instrumentation ourselves. mruby needed patch 0008
before the equivalent claim could be made. This is the strongest single argument
for perl as the next port.

## What each Sublet arm would cover

- **`MRBD_HEAP=sublet`-equivalent (the runtime's revoking heap under `malloc`):**
  every `Newx` buffer -- string bodies, AV/HV arrays, the stack, HEKs -- and each
  **arena chunk as one block**. SV heads and bodies inside a chunk are *not*
  separately revoked, exactly as mruby's GC pages were not. Also covers `malloc.c`
  if it is left off (`-Uusemymalloc`), which is what a first port should do.
- **A per-object arm (the analogue of mruby's patch 0008):** carve each arena
  chunk into per-head and per-body regions, take each with `sublet_take`, revoke
  on `del_SV` / `del_body`. The work is the same three rules as `gc.c`:
  - the free lists thread through freed slots (`PL_sv_root`, `PL_body_roots[]`)
    and must become a sidecar of indices;
  - **the body "ghost field" offset**: `SvANY` deliberately points *before* the
    allocated body for several types (`new_body = ((char *)new_body) - offset`,
    `sv.c:1152`; the table's `STRUCT_OFFSET(XPVIV, xiv_iv)` entries,
    `sv_inline.h:194ff`). A capability whose cursor sits below its base is the
    one construct here with no mruby precedent, and the first thing to test;
  - ~20 body types with different sizes, against mruby's one uniform 80-byte slot.
- **OP slabs** are a third, independent arm, and simpler: ops are freed in bulk
  when a sub is freed.

## Port cost

**In our favour, and the reason this is cheap:**

- **perl-cross solves the bootstrap.** `miniperl` is built with the **host**
  compiler (`$(HOSTCC) ... -DPERL_IS_MINIPERL`, `Makefile:104-132`), so the build
  never runs a target binary. This is the same split as the PostgreSQL port's host
  tools, and it is what makes perl cross-compilable at all.
- **Pointer-to-integer round trips are few.** Tree-wide: 66 `INT2PTR`, of which 27
  are in bundled extensions; **13 sites in the core `.c` files** take a pointer out
  of an SV's IV (`doio.c:1276` stores a `DIR *` in an IV and reads it back, the
  representative case). 412 `PTR2IV`/`PTR2UV` sites exist but are one-way in the
  common case (printing, hashing, debug output) and only matter where the value
  becomes a pointer again. Compare PostgreSQL, which needed `__uintcap_t Datum`
  for 781 + 729 sites: perl needs a handful of local fixes, **not a type change**.
- **No processes, sockets or threads are needed** to run a script. Configure with
  `-Uusedl` (static extensions only, as PostgreSQL's port links its modules),
  `-Uusethreads`, `-Uusemymalloc`.

**The blockers, in the order they will bite:**

1. **The test suite cannot run as-is.** 2823 `.t` files, and `t/TEST` forks per
   file (`t/TEST:349-395`). A domain has no `fork`. Validation needs a harness
   that runs a chosen set of `.t` files **in one process**, which is a real piece
   of work and the reason a perl port's "does it still behave?" gate costs more
   than mruby's single `mrbtest` image.
2. **The ghost-field offset** above, for the per-object arm.
3. **`Configure`/`perl-cross` needs a target triple it accepts**, and the domain's
   musl headers; the PostgreSQL port's `capstone-cc` wrapper is the model.

Estimate, by analogy with mruby (which took this session) and PostgreSQL: **a
first arm running scripts in a domain, a few days**; the in-process test harness
and the per-object arm, **one to two weeks**.

## Which release to pin

**Portability does not constrain the choice.** perl-cross carries a diff for
every perl5 release from **5.22.3 to 5.44.0** (40 of them, `cnf/diffs/`), and the
allocator mechanism is **identical** across the range: 5.32.1, 5.36.3 and 5.38.2
all have the same `PL_sv_root` head list, the same `PL_body_roots[]` body lists
and the same `PURIFY` switch. Two mechanical differences only:

- the arena code **moved** from `sv.c` into `sv_inline.h` between 5.34 and 5.36,
  so a per-object patch written for 5.36+ needs a rebase for 5.34 and older (the
  same situation as mruby's Prism/parse.y split between its two pins);
- 5.38 adds one body type (`SVt_PVOBJ`, the new class feature).

**So the pin should be chosen by the defect count, and the shortlist is:**

| Candidate | Why |
|---|---|
| **5.36.3** | the default recommendation on portability grounds: shares the current file layout, so one patch serves 5.36 through 5.44, and it is old enough for several years of later fixes |
| 5.32.1 | if the defect count clearly favours going older; costs one mechanical rebase of the patch into `sv.c` |
| 5.38.4 / 5.40.3 | if the count favours newer; nothing extra to do |

Before the port starts, the defect count wants the same shape as mruby's: per
release, how many later-fixed temporal defects are present, split by which layer
owns the memory (SV head, SV body, OP slab, plain `Newx`), and whether a public
reproducer exists. The SV-head and SV-body rows are the valuable ones, for the
same reason mruby's GC-slot rows were: `-DPURIFY` proves ASan cannot see them.
