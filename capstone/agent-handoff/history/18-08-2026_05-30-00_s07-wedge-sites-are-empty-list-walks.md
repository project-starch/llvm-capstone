# S-07: a second empty-list-walk wedge, and unzeroed arena residue refuted

Date: 2026-08-18. Bitstream: `caplifive_s06s08fix_s07tag2_618f4ce.bit`.

## RETRACTION — read this first

The first version of this note (commit `7d9edb8a0d89`) claimed a **reframing of the whole S-07
invariant** and that `Pager.pBackup` has **no writer anywhere in the build**. An adversarial audit
refuted both, and every point below was re-verified against primary sources before being written
here. What was wrong:

1. **"No code in the build ever writes `pBackup`" is FALSE as stated.** SQLite writes it through a
   pointer-returning accessor that a `->pBackup =` regex is structurally incapable of matching:
   `sqlite3-capstone.c:67217` `sqlite3PagerBackupPtr(Pager*)` returns `&pPager->pBackup`, and
   `attachBackupObject` (`:85177-85184`) does `pp = sqlite3PagerBackupPtr(...); *pp = p;`. A second
   writer is at `:85481`. `attachBackupObject` **is linked into the artifact**
   (`ob.dis` → `000000000001c928 <attachBackupObject>`). The defensible claim is only
   **"no dynamically reachable writer in this workload"**: the writers need `sqlite3_backup_*` or
   `VACUUM`, and neither appears in the domain's SQL or code.
2. **The reframe does not cover the majority of the evidence.** Of 8 mcause-25 wedges, 6 have a
   recoverable VA, and **5 of those 6 are `sqlite3OsRead+0x4c`** (`00-README.md:188-190`) — which
   is not a loop, walks no list, and whose field `id->pMethods` is legitimately written and
   legitimately non-NULL. The empty-list-walk pattern covers at most 3 of 8 wedges and none of the
   dominant site. It is a **corroborating second instance** of `00-README.md` §5, not a
   replacement framing.
3. **"Therefore a victim, NOT a lost tag" is a non-sequitur.** Plain stores through a garbage
   cursor are unchecked on this silicon — the faulting loop itself contains one
   (`ob.dis` `sw a0, 0x38(a1)`). So nonzero residue in a never-written field can be the
   *downstream product* of a genuine lost-tag load elsewhere. The two are not exclusive.
4. **Process error: I searched the wrong file.** The greps were run against
   `/tmp/capstone/sqlite-build/sqlite3-capstone.c` (sha `970014c7…`), which is **not** what was
   compiled into `OB.dom` — that is `/tmp/capstone/bake/OB/obj/sqlite3-capstone.c` (sha
   `c4fc1f5b…`), 47 diff lines apart, with `sqlite3OsRead` stubbed and a probe injected. Every
   line number in the original note was off by 3. The conclusions happened to survive the swap,
   but the check as performed was vacuous. **Verify the artifact, not a same-named neighbour.**

`00-README.md` was NOT edited — it is a sent link, and §5 already states this reasoning more
carefully than the retracted version did. One correction is owed there and is listed at the end.

## What the boot actually showed

Boot `/tmp/capstone/s07rate/obcombo1.txt`, three domains, control first.

| # | domain | what it is | outcome |
|---|--------|-----------|---------|
| 1 | `S7T` | control | `obs=1460078339` — **boot is valid** |
| 2 | `OB` | `SQLITE_OSREAD_STUB=1` + `SQLITE_S07_PAGER_PROBE=1` — both prior wedge sites covered | **wedged**, mepc `0x83c309a0` |
| 3 | `XU` | historical reproducer | not reached (collateral) |

Site: DBAS `0x83C00000`, DENT 0, so VA `= mepc − DBAS + 0x10000 = 0x409a0`, inside
`sqlite3BackupRestart` (entry `0x40944`). The identification is corroborated independently of the
arithmetic: the compiled `sqlite3_backup` struct carries an extra `char *zDestDb`, which with
16-byte capabilities puts `iNext` at **0x38** and `pNext` at **0x70** — matching
`sw a0, 0x38(a1)` and the faulting `ldc a0, 0x70(a0)` exactly. Two offsets agreeing is not
coincidence.

This makes `sqlite3BackupRestart` N=2 (with `S7B`), and the empty-list-walk observation N=3 across
two source sites.

## The shape, and why it is not a clue

```
   40974: cincoffsetimm a0, s0, -0x40
   40978: ld            a0, 0x0(a0)      <- NULL test: PLAIN INTEGER load of the low 64 bits
   4097c: beqz          a0, <exit>
   ...
   4099c: ldc           a0, 0x0(a1)      <- reload the loop variable
   409a0: ldc           a0, 0x70(a0)     <- p->pNext   *** FAULTS, mcause 25 ***
```

This is what `for(p=head; p; p=p->pNext)` compiles to at `-O0`. The "adjacent dependent `ldc`
pair" is therefore **the ordinary shape of a list walk**, not a signature of the defect — every
list walk in the program has it. That much of the reframe stands, and it explains why covering one
site relocates the wedge without any need for the fault to be "chasing" instrumentation.

Reaching the fault proves the guard's `ld` **returned** nonzero. It does **not** prove memory held
nonzero — see the open question.

## Ruled out, by check rather than inspection

- **The `memset` tag-check escape is not the cause.** `beebs_freestanding_string.c` contains a
  `BEEBS_MEMCPY_TAGCHECK` early `return dst` that would skip zeroing entirely, but it is gated on
  `CAPSTONE_MCP_TAGCHECK`, default 0 (`build-sqlite-silicon.sh:1652`). Confirmed in the shipped
  artifact: `memset` (`0x14c6d8`), `memcpy` (`0x14c1a8`) and `memmove` (`0x14c3c0`) all go straight
  from prologue to byte loop, no `lcc` guard.
- **The zeroing chain is intact in the shipped binary.** `sqlite3MallocZero` (`0x1b0cc`) calls
  `memset` at `0x14c6d8` after its null check; the entry glue zero-fills each `.bss` carve over its
  whole length, and `INTERP_GRANULE_ALIGN` (which would leave 240 arena bytes unzeroed) is off.
- **Unzeroed arena residue is REFUTED for everything QEMU exercises.** New knob
  `CAPSTONE_POISON_ARENA=1` fills the whole 256 KiB `sqlite_heap` with `0xA5` in `domain_main`,
  *after* the glue has zeroed it and *before* `SQLITE_CONFIG_HEAP` — emulating dirty DRAM and
  defeating the glue's zero-fill outright. memsys5 carves its blocks from `sqlite_heap` itself, so
  the arena is the backing store of every allocation. **The full workload passes.**
  `sqlite3MallocZero`'s `memset` suffices on its own.

  Negative-tested, not assumed: `CAPSTONE_POISON_NEGTEST=1` fills `0x5A`, the witness gate trips,
  and the run returns `obs=3134869504` = `0xBADA5000`. Scope limit: this says nothing about paths
  QEMU does not exercise.
- **Displacement (case a) is excluded for this wedge with a PROVEN instrument.** The boot's
  selftest fired: `SELFTEST post-204 = 0x41  OK: ldc_seen set and count moved by exactly 1`,
  `SELFTEST PASS` (`obcombo1.txt:1525-1529`). So `OB`'s `sw=204 = 0x00` is a **controlled**
  negative on this bitstream, not an unproven one.

## The open question — UNRESOLVED, and nothing here touches it

Every observation above is a **read**. Two hypotheses have *identical* observables:

- **H-mem** — memory genuinely holds nonzero untagged residue (a wild store, or a zeroing gap on a
  path QEMU does not exercise).
- **H-load** — memory holds correct zeros and the load path returned wrong data. If the guard's
  `ld` misread, `0x4099c` then loads an all-zero **untagged** value and `0x409a0` faults at
  *exactly the same mepc*, with the same `sw=204 = 0x00` and the same clean QEMU.

H-load additionally unifies all 8 wedges including the 5 `sqlite3OsRead` ones, which the
empty-list-walk story cannot.

A third rival the residue story does not exclude: **a use-after-free / double-close of the
`Pager`** under the OsRead-stub regime, where every read fails. `pagerFreeMapHdrs` is called from
`sqlite3PagerClose`, `sqlite3BackupRestart` from `pager_reset` — both close/error paths. One freed
`Pager` puts allocator metadata into *both* never-written fields with no silicon defect at all.
Weakened by rep-to-rep variance (identical domain state sometimes passes), which a deterministic
single-threaded UAF should not produce. **Cheap discriminator: run the OsRead-stub build under
QEMU** — not yet done.

Discriminator for H-mem vs H-load, one boot, batched, built to RETURN rather than wedge: plain-`ld`
each suspect field and early-return a site-coded sentinel on nonzero — at `PagerOpen` right after
`MallocZero`, before each `BackupRestart` call, and at the `FreeMapHdrs` loop. **Re-read the same
address N times in the probe**: a persistent value is memory, a one-shot is a bad load response.
That repetition is the part the earlier probe design lacked.

## Consequences

- **Drop `CapstoneLdcRetry` (Phase A) from the boot plan.** At these sites it discriminates
  nothing: with no valid capability to recover it re-reads NOT_CAP under both hypotheses, and the
  old decision table would have scored "still wedges" as a verdict.
- **Owed correction to `00-README.md` §5** (not yet applied — it is a sent link): the sentence
  "A non-zero cursor in such a slot means **the memory held wrong data**" overreaches in exactly
  the way retracted above. It should read: the guard's load *returned* nonzero; whether memory
  held nonzero is undetermined.
