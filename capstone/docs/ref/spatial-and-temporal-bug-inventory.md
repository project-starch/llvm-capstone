# Spatial and temporal bug inventory — memcached, FFmpeg, tshark

**What this document is.** The two tables the team asked for, and the three questions answered with
one number each. Every number here is recomputed from the `case.json` files and the ports'
`safety-expect.txt`, not carried from a summary; the script that recomputes them treats an arm whose
outcome cannot be classified as an **error**, not as a zero, so a table built on a gap cannot look
clean. Mechanism, ladders and the refutation trail live in
[`spatial-vs-temporal-three-programs.md`](spatial-vs-temporal-three-programs.md); this file is the
counts.

**Scope: the three programs the paper's target evaluation uses.** Other corpora in the tree
(cpython, httpd, PostgreSQL, sqlite, mruby, the cross-language set) are out of scope here —
`paper-bug-inventory.md` is the whole-tree inventory.

**One axis warning, because it has already caused one retraction.** *Nested vs not nested* is
**who allocated the object**: an inner allocator's sub-allocation, or a direct `malloc`. It is
**not** *which bound the access crosses* (the A/B/C axis of the companion document). The two are
orthogonal, and collapsing them is what produced the correction of 2026-10-05.

---

## Table 1 — TEMPORAL

Use-after-free and friends: the address is in bounds and the object is dead.

| program | nested | not nested | total | where |
|---|---:|---:|---:|---|
| memcached | **5** | **2** | **7** | `bug-corpora/memcached/allocator-repros/00-04`; app fixtures 17, 18 |
| tshark | **13** | **2** | **15** | `bug-corpora/wireshark/wmem-repros/00-12`; app fixtures 14, 15 |
| FFmpeg | **4** | **1** | **5** | `bug-corpora/ffmpeg/pool-repros/00-03`; app fixture 24 |
| **total** | **22** | **5** | **27** | |

All 27 are reductions of **live upstream defects**, each with its fix commit and a liveness proof in
its `case.json`.

## Table 2 — SPATIAL

An access that leaves a bound. Built from upstream defects:

| program | nested | not nested | total | where |
|---|---:|---:|---:|---|
| memcached | **3** | 0 | **3** | `allocator-repros/05-07` — item data, suffix field, unterminated key; the objects are `slabs.c` chunks |
| tshark | **5** | 0 | **5** | `wmem-repros/13-17` — cursor skip, fixed-offset loop, negative index, parity write, off-by-one size |
| FFmpeg | 0 | **3** | **3** | `subobject-repros/00-02` — three members written past, inside one `av_malloc` |
| **total** | **8** | **3** | **11** | |

And the **synthetic baseline**, which is where the not-nested spatial row really lives:

| probe | programs | role |
|---|---|---|
| fx2 `heap_neighbour`, fx3 `heap_one_past` | all three app ports | the standing malloc-granular control: `level0` RETURN, `shrink` FAULT `oob` |
| fixtures 20, 21 | memcached (new, 2026-10-05) | two more class-A probes, modelled on historical defects: **measured 15/15**, `results/2026-10-05-qemu-classa-fixtures/` |

### Why the not-nested spatial column has no live upstream defect in it

This was the team's question, and the honest answer is a measurement rather than an absence.

A spatial-wording pass over the same commit populations yields **29 class-A candidates** (tshark 21,
memcached 3, FFmpeg 5) — class A being a crossing of the `malloc` bound itself. **Eight have been
read against the source each port actually pins. None is a live class-A defect:**

| candidate | disposition in the pinned source |
|---|---|
| memcached `ddee3e2` | fix present — `authfile.c:50` is `calloc(1, sb.st_size + 2)` |
| memcached `11b5f9b` | not heap — `char temp[KEY_MAX_LENGTH + 1]` at `proxy_lua.c:717` |
| tshark `be813ede9d` | code absent at 4.6.8 |
| tshark `f207d25f4b` | `g_strdup` of a string literal |
| tshark `830cf562a0` | integer-underflow subject, excluded by the hunt's own filter 1 |
| tshark `06d08c5811` | no access leaves the allocation — `wsutil/eax.c:150` allocates `worksize`, the loop bound *is* `worksize` |
| tshark `7ffc11e38f` | fix present — `wiretap/file_access.c:1322`, `:1376` |
| tshark `3be1c99180` | fix present — `wiretap/netscreen.c:63-66`, `:311`, `:332` |
| the remaining **21** | **not individually read** — stated as unread, not as absent |

**Class A is the class upstream fixes first.** It is what fuzzers, ASan and compiler warnings find,
and these programs are pinned at recent releases (memcached 1.6.45, wireshark 4.6.8), so those fixes
are already inside the tree we build — three of the eight rows above are exactly that. What survives
into a current release is the crossing no existing tool sees: inside a nested allocator's block, or
inside a single allocation. **That is the project's thesis, and the 0 is evidence for it rather than
a hole in the data.**

So the not-nested spatial baseline is **synthetic by necessity**, and that is the right instrument:
a baseline row's job is to show the arms discriminate, not to count upstream defects. memcached
fixtures 20 and 21 are modelled on two historical defects (`ddee3e2`'s authfile scan and
`d5d9ff0`'s cachedump `END\r\n` reservation, both fixed at the pin — `items.c:678` now reads
`bufcurr + len + 6`). They are **probes, not upstream-defect reductions**, and are counted as such.

---

## The three questions

### (a) How many spatial and temporal bugs, and how many are instances of nesting?

| | nested | not nested | total |
|---|---:|---:|---:|
| temporal | **22** | **5** | **27** |
| spatial (upstream reductions) | **8** | **3** | **11** |
| **total** | **30** | **8** | **38** |

**30 of 38 are instances of nesting (79%).** On the spatial side specifically, 8 of 11. Add the
synthetic baseline probes and the not-nested spatial row grows, but those are probes and are kept
out of the defect count on purpose.

### (b) How many does Capstone catch without extra protection, and with Sublet?

"Without extra protection" is the **`spatial` arm**: Capstone bounds, revocation off.

| | without extra protection | with Sublet | with the inner allocator ported |
|---|---:|---:|---:|
| temporal, 22 nested | **0 of 22** | **21 of 22** | **22 of 22** |
| temporal, 5 not nested | — (fixtures; see note) | — | — |
| spatial, 11 upstream | **6 of 11** | **6 of 11** | tshark **5 of 5** |

- **Temporal, 0 of 22 without protection, measured** — not predicted. A bound cannot see a dead
  object: the stale address is in bounds by construction.
- **21 of 22 with Sublet.** The single miss is tshark `wmem-repros/12`, where an individual
  recycler free ends no epoch — a *recorded non-detection*, declared in the case file, not a
  silent gap. Porting the inner allocator (`sublet-chunks`, `sublet-port`, memcached's slab arms)
  closes it: **22 of 22**.
- **Spatial: Sublet adds nothing, and that is expected.** Sublet *is* revocation, and revocation has
  nothing to fire on while the object is alive. The 6 spatial catches are `shrink`'s per-object
  bounds, which the `sublet` arm inherits by construction. The two columns being identical is a
  consistency check, not a null result.

**The "without extra protection" column means a different granularity in each program.** One column
cannot carry two meanings silently:

| program | what the bound actually is on that arm | spatial caught |
|---|---|---:|
| tshark | **per-chunk, not per-malloc.** `wm_narrow()` (`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15`) narrows *every* wmem allocation on *every* arm, so this harness has **no malloc-granular arm** at all | 5 of 5 |
| memcached | **per-chunk** — the slab carve | 1 of 3 |
| FFmpeg | **malloc-granular**, and still blind: each crossing is between two members of one `av_malloc` | 0 of 3 |

So tshark's 5 of 5 must **not** be read as "bounds alone suffice". It is the opposite: that harness
narrows everywhere, and the malloc-granular contrast is only visible in the app ports' fx12 ladder,
where `level0`, `shrink` and `sublet` all RETURN and only the chunk-ported arm faults.

### (c) How many does CheriBSD catch with default revocation on?

| | caught | measured | not measured |
|---|---:|---:|---:|
| temporal, 22 nested | **0** | **18** | 4 |
| spatial, 11 upstream | **0** | **0** | 11 |

- **Temporal: 0 caught, and 18 of the 22 are MEASURED with a positive control that fires.**
  - tshark 13 — `results/20260921-cheribsd/` (`matrix.tsv`, `arm=cheribsd`): expected complete,
    passed, exit 0, **0 of 13 caught**. The control: the PoisonCap mode-1 arm in the *same bundle
    and the same boots* faults **13/13** with `SIGPROT` at the labelled read probe. The platform
    **can** be made to catch these cases, so the completion is a reading, not a dead instrument.
  - memcached 5 — stock CheriBSD 2026-09-21, revocation on and verified: each case completes
    (`completed=1`, `object_reuses=1`) while a `revocation-control` in the same boot **faults** at
    the labelled probe with `SIGPROT`/`PROT_CHERI_TAG`.
  - **The mechanism, stated so the zero is falsifiable:** the stale storage never reaches `free()`.
    It goes back on an inner allocator's own free list — `cache.c`'s STAILQ, a wmem scope reset, a
    pool return — inside a block `malloc` still owns, so libc's quarantine never holds the object
    and the revoker has nothing to sweep.
  - **4 not measured, and all 4 now declared:** FFmpeg `pool-repros`. Case 3's
    `cheribsd-revocation` arm held a bare `{"status": "not written"}` and was the one cell in either
    table with no oracle at all; it was declared on 2026-10-05 with its siblings' mechanism — the
    two side tables return to their `AVRefStructPool`s (`refs.c:153`, `:157`) rather than to
    `free()` — marked **predicted, not measured**, and naming what a reading would need. So this
    column is **22 declared, 18 measured, 0 caught**, with no blanks left in it.
- **Spatial: 0 of 11 measured.** All 11 carry a prediction that the case *completes*, and the reason
  is structural: CHERI bounds the slab page or the enclosing allocation, which is one `malloc`. Not
  measurable on this host, checked rather than assumed —
  `ports/common/cmake/toolchains/cheribsd.cmake:4-9` requires `CHERI_SDK` and `CHERI_SYSROOT` and
  `FATAL_ERROR`s without them; both are unset after sourcing the project environment, and no SDK,
  purecap sysroot or image exists here.
- **Expect a spatial CheriBSD run to TIE, not to differ.** This repo's own committed verdict:
  *"for the spatial / null / uninitialised rows, base CHERI is already sufficient — both systems
  catch them synchronously. Capstone claims no advantage here."*
  (`table6-cheri-vs-capstone-explained.md:158-162`.) A CHERI nested allocator could narrow to each
  suballocation exactly as our `chunks` arm does, so **the nested-spatial gap is a porting gap, not
  a hardware gap.** The temporal gap is the opposite: the stale address is legitimate and in bounds,
  so no bound helps, and revocation *inside* a nested allocator needs the lease mechanism.

### What a CheriBSD host run would need (handover, not a TODO)

The SDK, purecap rootfs and PoisonCap image stay outside Git by the platform README's own design, so
this is what a host that has them needs in order to extend the measured column:

- `ports/common/host/cheribsd/poisoncap/build.sh` and `run.py`, the invocations the 2026-09-21
  bundle recorded;
- `CHERI_SDK` and `CHERI_SYSROOT` exported (`ports/common/host/cheribsd/build.py:21,26`);
- the platform hashes that bundle recorded (qemu, firmware, kernel, libc, image), so a later run can
  be matched against it rather than merely compared;
- and the **same positive control in the same boot** — a free, a sweep, a read through the old
  pointer, which must fault. Without it a completion is not a reading.

---

## Reconciliation

| | count | source |
|---|---:|---|
| `case.json` files across the four corpora | 33 | `memcached/allocator-repros` 8, `wireshark/wmem-repros` 18, `ffmpeg/pool-repros` 4, `ffmpeg/subobject-repros` 3 |
| temporal corpus cases | 22 | 5 + 13 + 4 |
| spatial corpus cases | 11 | 3 + 5 + 3 |
| not-nested temporal, as app fixtures | 5 | memcached 17/18, tshark 14/15, FFmpeg 24 |
| **total defects in both tables** | **38** | 27 temporal + 11 spatial |

Arm cells for the 11 spatial cases: **36 measured, 33 unavailable, 8 declined, 3 n/a = 80**, which
is the closed accounting in the companion document.

**The two new memcached fixtures are MEASURED.** `results/2026-10-05-qemu-classa-fixtures/`,
**15 of 15 cells as predicted** against rows registered before any image existed — fixtures 20 and
21 plus fx2/fx3 as the standing malloc-granular controls and fx17 as the temporal control, one boot
per arm, images cited by hash (`level0` `9beb8bbfebdf04a1`, `shrink` `c8c1a1596787f908`, `sublet`
`b5b340c630e0ea9d`).

| | fx2 | fx3 | fx17 (temporal) | **fx20** | **fx21** |
|---|---|---|---|---|---|
| `level0` | RETURN | RETURN | RETURN | **RETURN** | **RETURN** |
| `shrink` | FAULT cause 7 | FAULT cause 5 | RETURN | **FAULT cause 5** | **FAULT cause 7** |
| `sublet` | FAULT cause 7 | FAULT cause 5 | FAULT **cause 24** | **FAULT cause 5** | **FAULT cause 7** |

`level0` RETURN beside `shrink` FAULT is what makes them class A, and the falsifier was pre-written:
had `level0` faulted too, the fixture would not have been class A. Three checks the verdict did not
need and the cells pass anyway: the cause matches the access (fixture 20 reads, cause 5, a load
`insn`; fixture 21 writes, cause 7, a store), the faulting address is the fixture's own printed
target, and fx17 faults with cause **24** — revoked authority, a different class — only on `sublet`.
The gate is negative-tested: with fx20's row forced to `FAULT oob` the judge reports `DIFFERS` while
its neighbours pass, and exits 1.

**They are probes, not upstream-defect reductions**, so they do not change the 38 in table (a) —
both modelled defects are already fixed at the pin. What they change is that the not-nested spatial
baseline is now four measured probes per arm instead of two.
