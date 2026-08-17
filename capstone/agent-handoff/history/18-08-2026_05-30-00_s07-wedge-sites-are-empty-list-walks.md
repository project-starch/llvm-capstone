# S-07: every wedge site is a walk over a list whose head has no writer

Date: 2026-08-18. Bitstream: `caplifive_s06s08fix_s07tag2_618f4ce.bit`.

This note records a **reframing of the S-07 invariant** and one refutation. It does **not**
record a root cause, and it deliberately stops short of one: the decisive question is still open
and is named at the bottom. Nothing here has been applied to the sent repro folder.

## What the boot showed

Boot `/tmp/capstone/s07rate/obcombo1.txt`, three domains, control first.

| # | domain | what it is | outcome |
|---|--------|-----------|---------|
| 1 | `S7T` | control | `obs=1460078339` — **boot is valid** |
| 2 | `OB` | `SQLITE_OSREAD_STUB=1` + `SQLITE_S07_PAGER_PROBE=1` — both previously known wedge sites covered | **wedged**, mepc `0x83c309a0` |
| 3 | `XU` | historical reproducer | not reached (collateral) |

`OB` covers both sites that had wedged before, and wedged anyway — at a **third** function.
Symbolised against `/tmp/capstone/bake/OB/sqlite_silicon.dom`: DBAS `0x83C00000`, so
VA `0x409a0`, inside `sqlite3BackupRestart` (entry `0x40944`).

## The reframe: the sites were never "moving"

Three wedges, three functions, one shape — and the shape is not exotic. It is what
`for(p=head; p; p=p->pNext)` compiles to at `-O0`:

```
   40974: cincoffsetimm a0, s0, -0x40
   40978: ld            a0, 0x0(a0)      <- NULL test: PLAIN INTEGER load of the low 64 bits
   4097c: beqz          a0, <exit>
   ...
   4099c: ldc           a0, 0x0(a1)      <- reload the loop variable
   409a0: ldc           a0, 0x70(a0)     <- p->pNext   *** FAULTS, mcause 25 ***
```

Two consequences, and they change how the whole investigation reads:

1. **The "adjacent dependent `ldc` pair" invariant is not a clue about the defect.** It is the
   ordinary shape of a list walk. Every list walk in the program has it, which is why covering
   one site only relocated the wedge — the earlier reading, that the fault was *chasing* our
   probes, was over-reading a structural fact.
2. **The NULL guard is a plain `ld`, not a capability operation.** So it passes on *any* nonzero
   low-64 bits, tagged or not. Entering the loop body at all therefore proves the slot's observed
   low 64 bits were nonzero — regardless of which iteration faulted, because the incoming
   argument is `stc`'d into that slot at `0x40960` immediately before.

## The fields have no writer

Checked in the patched amalgamation `/tmp/capstone/sqlite-build/sqlite3-capstone.c`:

- **`Pager.pBackup`** — searching `->pBackup\s*=[^=]` returns **zero matches**. The backup API is
  omitted from this build. It is read at lines 61440, 62840, 66198 as
  `sqlite3BackupRestart(pPager->pBackup)` and **never assigned anywhere**.
- **`Pager.pMmapFreelist`** — the earlier `pagerFreeMapHdrs+0x4c` wedge walks this one. Its only
  writers are 63781 and 63820, both on mmap-page paths that `SQLITE_MAX_MMAP_SIZE=0` does not
  reach.

So both observed heads are fields whose value can only come from allocation-time initialisation.
The Pager block is allocated by `sqlite3MallocZero` in `sqlite3PagerOpen` (body near line 64436).

## What was ruled out, with the checks

- **The `memset` tag-check escape is NOT the cause.** `beebs_freestanding_string.c` does contain a
  `BEEBS_MEMCPY_TAGCHECK` early `return dst` that would skip zeroing entirely, but it is gated on
  `CAPSTONE_MCP_TAGCHECK`, which defaults to 0 (`build-sqlite-silicon.sh:1652`). Confirmed in the
  shipped artifact, not just the source: `memset` at `0x14c6d8` in `OB.dom` goes straight from
  prologue to the byte loop, no `lcc` guard; same for `memcpy` and `memmove`.
- **The zeroing chain is intact in the shipped binary.** `sqlite3MallocZero` (`0x1b0cc`) calls
  `memset` at `0x14c6d8` after its null check. The entry glue
  (`tests/runtime-qemu/silicon-ladder/start-gp-captable-interp.S`) zero-fills each `.bss` carve
  over its whole length (8-byte loop at `31:`, byte tail at `32:`), and
  `INTERP_GRANULE_ALIGN` — which would have left 240 bytes of the arena unzeroed — is off.
- **"Unzeroed arena residue" is REFUTED for everything QEMU exercises.** New knob
  `CAPSTONE_POISON_ARENA=1` fills the whole 256 KiB `sqlite_heap` with `0xA5` in `domain_main`,
  *after* the glue has zeroed it and *before* `SQLITE_CONFIG_HEAP` — i.e. it emulates dirty DRAM
  and defeats the glue's zero-fill outright. memsys5 carves its blocks out of `sqlite_heap`
  itself, so the arena is the backing store of every allocation. Result: **the full workload
  passes**. `sqlite3MallocZero`'s `memset` is sufficient on its own.

  The gate is negative-tested, not assumed: `CAPSTONE_POISON_NEGTEST=1` fills `0x5A` instead, and
  the run comes back `obs=3134869504` = `0xBADA5000`, the witness sentinel. A poisoned run that
  cannot fail would have proved nothing.

## What is NOT established — the open question

Everything above is a statement about **reads**. Two hypotheses remain and no evidence here
distinguishes them:

- **H-mem** — memory genuinely holds nonzero untagged residue (a wild store, or a zeroing gap on a
  path QEMU does not exercise).
- **H-load** — memory holds the correct zeros and the load path sporadically returned wrong data.
  A wrong-granule select within a cache line would return a neighbouring nonzero integer field:
  nonzero guard, untagged `ldc`, a different site each run, clean under QEMU. **Identical
  observables.**

The poison result argues against the deterministic-software flavour of H-mem, since a genuine
zeroing gap in a `MallocZero`'d struct would now reproduce under emulation and does not.

Discriminator, one boot, batched, built to RETURN rather than to wedge: plain-`ld` each suspect
field and early-return a site-coded sentinel on nonzero — at `PagerOpen` immediately after
`MallocZero`, before each `BackupRestart` call, and at the `FreeMapHdrs` loop. Nonzero already at
allocation ⇒ zeroing gap; zero at allocation and nonzero later ⇒ wild store, bisect the interval;
**all probes read zero yet it still wedges ⇒ H-load**, which is an RTL question again but a far
sharper one than before.

## Consequence for the pending retry pass

`CapstoneLdcRetry` (Phase A) **no longer discriminates anything at these sites** and should be
dropped from the boot plan. If the loaded value is a should-be-NULL field, there is no valid
capability to recover: the retry re-reads NOT_CAP and the wedge proceeds under *both* hypotheses.
Worse, the old decision table would have scored "still wedges" as evidence for one of them.
