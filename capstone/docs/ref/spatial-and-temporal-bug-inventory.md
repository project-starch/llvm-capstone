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
| memcached | **3** | **1** | **4** | `allocator-repros/05-07` (`slabs.c` chunks) and `plain-heap-repros/00` (`ddee3e2`, one byte past a `calloc`) |
| tshark | **5** | 0 | **5** | `wmem-repros/13-17` — cursor skip, fixed-offset loop, negative index, parity write, off-by-one size |
| FFmpeg | 0 | **3** | **3** | `subobject-repros/00-02` — three members written past, inside one `av_malloc` |
| **total** | **8** | **4** | **12** | |

And the **synthetic baseline**, which carried the not-nested spatial row alone until
`plain-heap-repros/00` landed and which still does the job no upstream case can -- showing
the arms discriminate at `malloc` granularity on demand:

| probe | programs | role |
|---|---|---|
| fx2 `heap_neighbour`, fx3 `heap_one_past` | all three app ports | the standing malloc-granular control: `level0` RETURN, `shrink` FAULT `oob` |
| fixtures 20, 21 | memcached (new, 2026-10-05) | two more class-A probes, modelled on historical defects: **measured 15/15**, `results/2026-10-05-qemu-classa-fixtures/` |

### The one remaining zero that IS a gap: FFmpeg nested spatial

FFmpeg contributes 3 not-nested spatial cases and **0 nested** ones, and unlike the zeros discussed
below this one is a genuine gap with a named candidate.

**`9edd06f861` — `avcodec/proresenc_kostya`, "fill macroblock rows past the end of the bottom
field".** Verified live at our pin, two-sided: the vulnerable expression
`avctx->height / ctx->pictures_per_frame` occurs **4 times** in `n9.0.1`'s
`libavcodec/proresenc_kostya.c`, the fix's marker `picture_height` **0 times**, with `encode_slice`
at 5 occurrences as the positive control that the search reaches that file at all. The read is
`src = pic->data[i] + line_add * pic->linesize[i]` passed to `get_slice_data` with that height, so
for an interlaced frame the encoder reads **past the bottom field's last row**.

**Why that is the nested shape.** The overflowed object is a *frame plane*, and the planes of a
frame from `av_frame_get_buffer` are carved from **one** `AVBuffer` — literally the nested note the
triage tool carries for FFmpeg. A crossing from one plane into the next stays inside that single
allocation, so per-`malloc` bounds are in bounds for it and only a plane-granular adapter could
separate them. Same structure as tshark's wmem chunks, one axis over.

**What building it would and would not buy.** It would move this cell from 0 to 1 and give FFmpeg
its first nested spatial case. It would **not** by itself produce a discriminating reading: FFmpeg's
existing arms narrow pool buffers, not frame planes, so without a new adapter the case would read
"completes on every arm" — a legitimate measured row of the same kind as the three sub-object cases,
and the `partial²` verdict again. A discriminating cell needs a plane-narrowing port, which is new
port work rather than a new case. Before building, check the allocation's padding: `linesize` is
aligned and padded, so the crossing must be shown to leave the *plane* in a frame whose planes are
actually adjacent.

Two weaker candidates from the same pass, allocation sites **not** opened: `56309e476a`
(`vf_vif`, index mirroring with small dimensions) and `2a20737f66` (a **revert** of a bwdif
heap-overflow fix, so provenance needs care before it is called a defect). Disqualified on sight as
not in our build: `884590dd4a` (AltiVec/PPC), `8b4fad11ac` (LoongArch), `ffe0104574` (CUDA).

### Why the not-nested spatial column has no live upstream defect in it

This was the team's question, and the honest answer is a measurement rather than an absence.

A spatial-wording pass over the same commit populations yields **29 class-A candidates** (tshark 21,
memcached 3, FFmpeg 5) — class A being a crossing of the `malloc` bound itself. **Seven have been
read against the source each port actually pins. None is a live class-A defect:**

| candidate | disposition in the pinned source |
|---|---|
| memcached `ddee3e2` | fix present — `authfile.c:50` is `calloc(1, sb.st_size + 2)` |
| memcached `11b5f9b` | not heap — `char temp[KEY_MAX_LENGTH + 1]` at `proxy_lua.c:717` |
| tshark `be813ede9d` | code absent at 4.6.8 |
| tshark `f207d25f4b` | **RETRACTED 2026-10-05:** misattributed, and returned to the unread pool. The row said "`g_strdup` of a string literal" at `wiretap/libpcap.c:621`; that line is the file's only `g_strdup`, but this commit has nothing to do with it. `git show --stat` gives the subject *"wiretap: pcap[ng]: Don't let the reported length underflow w/ phdr"* over `libpcap.c` and `pcapng.c`. Written from a grep in the file instead of the commit's diff — the same defect retracted above, committed again the same day |
| tshark `830cf562a0` | integer-underflow subject, excluded by the hunt's own filter 1 |
| tshark `06d08c5811` | no access leaves the allocation — `wsutil/eax.c:150` allocates `worksize`, the loop bound *is* `worksize` |
| tshark `7ffc11e38f` | fix present — `wiretap/file_access.c:1322`, `:1376` |
| tshark `3be1c99180` | fix present — `wiretap/netscreen.c:63-66`, `:311`, `:332` |
| the remaining **22**, `f207d25f4b` among them | **not individually read** — stated as unread, not as absent. 7 dispositioned + 22 unread = 29; the row above is listed for its trail and counted in the 22, not twice |

**Class A is the class upstream fixes first.** It is what fuzzers, ASan and compiler warnings find,
and these programs are pinned at recent releases (memcached 1.6.45, wireshark 4.6.8), so those fixes
are already inside the tree we build — three of the rows above are exactly that. What survives
into a current release is the crossing no existing tool sees: inside a nested allocator's block, or
inside a single allocation. **That is the project's thesis, and the 0 is evidence for it rather than
a hole in the data.**

> **RETRACTED 2026-10-06.** What stood here said the not-nested spatial baseline is synthetic
> **by necessity**, and that the zero is "a result, not a gap". **The measurement stands and is
> unchanged: 0 of the candidates read is live at the pin.** What is withdrawn is the inference from
> it to an empty cell, which rested on a premise never stated and never checked — *that a case must
> be live at the pin to be built*.
>
> It is not. **27 of the 33 existing corpus cases carry `live_in_pin: false`**, and the convention is
> explicit at `bug-corpora/memcached/allocator-repros/README.md:132-135`: each fix is an ancestor of
> the pin, "so the shipped allocator is exercised by a pre-fix consumer shape the commit's own diff
> shows -- the FFmpeg corpus's tier." Liveness is a **field recorded in the case**, not a gate on
> building one. Every temporal case in this inventory is itself a fix-reversal.
>
> So "class A is fixed upstream first" remains true *about liveness* and explains why the live count
> is 0. It does not explain an empty cell, and it was wrong to present the cell as closed. The cell
> is **open**, with one verified buildable candidate (below).
>
> The check that would have caught it is the one already written down: test the single sentence the
> conclusion rests on. Here that sentence was "a case must be live", which no document asserts and
> the corpus contradicts 27 times.

**Re-dispositioned against the correct criterion** — *a reconstructible heap defect whose crossing
leaves the allocation*, with liveness recorded rather than required:

| candidate | under the correct criterion |
|---|---|
| memcached `ddee3e2` | **BUILDABLE.** Its subject is "Fix minor severity heap buffer overflow reading `--auth-file`"; before the fix `auth_data = calloc(1, sb.st_size)` is scanned by an unclamped `fgets(auth_cur, MAX_ENTRY_LEN, ...)`, so the read leaves the allocation. A fix-reversal case exactly like the 27 |
| tshark `7ffc11e38f` | **BUILDABLE.** The capacity question is settled by the fix itself: it adds `file_type_subtype < 0 ||` as well as the `>= len` bound, so a **negative** index was reachable and `file_type_subtype_table[-n]` reads *before* the `g_malloc`'d GArray body (`file_access.c:1203-1205`). Below the allocation, not past `len`, so the over-allocation that disqualifies the other wiretap candidates does not apply |
| tshark `f207d25f4b` | no — read at last, from the diff: it replaces `orig_size -= phdr_len` and `packet_size -= phdr_len` with checked `ckd_sub`, so the defect is an **unsigned underflow** of a reported length, the same class as `830cf562a0` that filter 1 excludes by its own wording. Its downstream consequence may be spatial; the defect is not |
| tshark `3be1c99180` | no — `ws_buffer` over-allocates, so the crossing stays inside the allocation. Class C |
| tshark `be813ede9d` | no — a fix-reversal needs the code to exist at the pin, and `etw_dump_write_ldap_event` does not |
| tshark `06d08c5811` | no — unchanged, no access leaves the allocation |
| memcached `11b5f9b` | no — unchanged, a stack array |
| tshark `830cf562a0` | no — unchanged, an integer-underflow subject |

**This does not put tshark's 21 back in play.** Most still fail on capacity-versus-length or on the
code having to exist at the pin; what changed is the criterion, not the evidence.

**Two of the three empty cells therefore have a verified buildable candidate**, each read from the
commit's own diff rather than from a grep in the file:

| cell | candidate | the crossing |
|---|---|---|
| memcached, not-nested spatial | `ddee3e2` | an unclamped `fgets` scan leaves `calloc(1, sb.st_size)` |
| tshark, not-nested spatial | `7ffc11e38f` | a negative index reads below the `g_malloc`'d GArray body |
| FFmpeg, **nested** spatial | candidates only — see the section above | a frame-plane crossing, ownership and padding still to be opened |

They are **candidates until built and measured**, and neither is counted in any table yet.


The synthetic probes keep their role regardless: fx2/fx3 and memcached 20/21 show the arms
discriminate at `malloc` granularity, which is a different job from counting upstream defects.

---

## The three questions

### (a) How many spatial and temporal bugs, and how many are instances of nesting?

| | nested | not nested | total |
|---|---:|---:|---:|
| temporal | **22** | **5** | **27** |
| spatial (upstream reductions) | **8** | **4** | **12** |
| **total** | **30** | **9** | **39** |

**30 of 39 are instances of nesting (77%).** On the spatial side specifically, 8 of 12. Add the
synthetic baseline probes and the not-nested spatial row grows further, but those are probes and
are kept out of the defect count on purpose.

### (b) How many does Capstone catch without extra protection, and with Sublet?

"Without extra protection" is the **`spatial` arm**: Capstone bounds, revocation off.

| | without extra protection | with Sublet | with the inner allocator ported |
|---|---:|---:|---:|
| temporal, 22 nested | **0 of 22** | **21 of 22** | **22 of 22** |
| temporal, 5 not nested | **0 of 5** | **5 of 5** | tshark **2 of 2** |
| spatial, 12 upstream | **6 of 11 measured** | **6 of 11 measured** | tshark **5 of 5** |

- **Temporal, 0 of 22 without protection, measured** — not predicted. A bound cannot see a dead
  object: the stale address is in bounds by construction.
- **21 of 22 with Sublet.** The single miss is tshark `wmem-repros/12`, where an individual
  recycler free ends no epoch — a *recorded non-detection*, declared in the case file, not a
  silent gap. Porting the inner allocator (`sublet-chunks`, `sublet-port`, memcached's slab arms)
  closes it: **22 of 22**.
- **Temporal, 5 not nested: 0 of 5 without protection, 5 of 5 with Sublet** —
  measured 2026-10-05, `ports/common/application/results/2026-10-05-qemu-plain-temporal-baseline/`,
  37 of 37 cells as predicted over ten boots. This row had a dash in it until then, and it is the
  **denominator**: the plain `malloc`/`free` case against which the nested ones are read. Every
  `sublet` catch is cause **24**, revoked authority, not a bounds cause. `shrink` returning on all
  five is the finding, not a gap — these objects really are freed, and a bound still cannot see a
  dead object. fx2/fx3 ran in all ten boots and faulted on every enforcing arm, so the two axes are
  separated inside each boot: the same image that returns on a temporal fixture faults on a spatial
  one.
- **The twelfth case is counted but not yet measured under Capstone.**
  `bug-corpora/memcached/plain-heap-repros/00` (`ddee3e2`) is measured on both NATIVE arms,
  two-sided — the fix differential and, unlike every sub-object case in this tree, an **ASan report**
  (`heap-buffer-overflow`, WRITE of size 1, 0 bytes after a 9-byte region). Its `spatial` and
  `sublet` arms are declared **predictions to fault**, because that corpus has no domain runner yet;
  the corresponding reading already exists in the port as memcached fixture 20, which carries the
  same shape and reads `level0` RETURN / `shrink` FAULT `oob`. So the measured catch counts above
  stay at 6 of 11 rather than silently becoming 7 of 12.
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
| spatial, 12 upstream | **0** | **0** | 12 |

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
- **The 5 not-nested temporal are the one cell this column predicts NON-ZERO, and it is unmeasured.**
  Same mechanism, run the other way: the nested cases complete because the stale storage never
  reaches `free()` — it returns to an inner allocator's own free list inside a block `malloc` still
  owns. **These five have no inner allocator**, so the quarantine does hold the object and the
  revoker does sweep it. Predicted **caught, 5 of 5**; not measurable here. That prediction is what
  gives the measured 0 of 18 its meaning — a system catching nothing anywhere would be
  indistinguishable from a dead instrument, while one that catches the plain cases and misses the
  nested ones is measuring the nesting. **It is therefore the first thing an SDK host should run**,
  and a miss would refute the mechanism rather than add a data point.
- **The new plain-heap case is the first spatial row predicted CAUGHT by stock CheriBSD**, and it
  is unmeasured like the rest. The reason is structural rather than hopeful: CHERI bounds each
  `malloc`, and this crossing leaves the `malloc` bound instead of staying inside a slab page or a
  struct. Revocation is irrelevant to it; the bounds are not. A miss would refute the bounds claim
  rather than add a data point — which is what makes it worth running first on an SDK host, beside
  the five not-nested temporal fixtures.
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
