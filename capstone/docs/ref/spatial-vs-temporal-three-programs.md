# Spatial vs temporal, nested vs plain — memcached, FFmpeg, tshark

**Scope: the three programs the paper's target evaluation uses.** Other corpora in this tree
(cpython, httpd, PostgreSQL, sqlite, mruby, the cross-language set) are deliberately out of scope
here; `paper-bug-inventory.md` is the whole-tree inventory.

**Read this first if you came here grepping for "spatial".** `spatial` is also the name of a
**measurement arm** — bounds-only Capstone, revocation disabled — and it appears in dozens of
`case.json` files and in the paper as `Capstone spatial 0/57`. That zero is a *configuration*
failing to catch *temporal* bugs. It is not a count of spatial bugs, and it is not evidence that
anything is broken. A grep for the word finds arms, not defects.

## 1. The temporal 27 were all there was, because spatial had been filtered out — no longer true

> **HEADING CORRECTED 2026-10-07.** It read *"The 27 real upstream defects are all temporal"*, which
> was the state when this document was written and contradicted its own table below as soon as the
> spatial hunt produced anything. The spatial corpus is now **32 built and measured cases**, so the
> sentence had become false in the one place a reader looks first.

| | nested allocator | plain / system allocator | total |
|---|---:|---:|---:|
| **spatial**, built and measured | **14** | **56** | **70** |
| **temporal**, built and measured | **22** | **26** | **48** |
| **total** | **36** | **82** | **118** |

Per program, recomputed from each case's `nested` boolean and its `lifetime_ender`:

| program | spatial / nested | spatial / plain | temporal / nested | temporal / plain | total |
|---|---:|---:|---:|---:|---:|
| FFmpeg | 1 | 35 | 4 | 13 | **53** |
| tshark | 9 | 12 | 13 | 10 | **44** |
| memcached | 4 | 9 | 5 | 3 | **21** |

> **UPDATED 2026-10-08.** The table above read `14 | 18 | 32` spatial and `22 | 5 | 27` temporal
> yesterday, with **temporal / plain = 0 for all three programs**. That zero was a property of
> which CORPORA existed, not of the upstream software: every temporal corpus in this tree sat on a
> nested allocator — FFmpeg's AVBufferPool and AVRefStructPool, Wireshark's wmem, memcached's
> slabs.c and cache.c — because that is what the temporal hunts were aimed at. All three programs
> also free direct allocations and use them afterwards, and there was nowhere to record one.
>
> Three `plain-temporal-repros` corpora were created and filled (FFmpeg 13, tshark 10, memcached 3),
> and the plain spatial row grew by 38.
>
> **The baseline, with its denominator named, because this document's own history shows how easily
> two differently-counted totals get compared.** All three figures below count CASES in these three
> programs, from `case.json` files, on the same basis as the table above:
>
> | state of the branch, by its last commit that day | FFmpeg | tshark | memcached | total |
> |---|---:|---:|---:|---:|
> | end of 2026-10-05 (`84990ea0ade5`) | 7 | 18 | 8 | **33** |
> | end of 2026-10-06 (`39f62dfb0354`) | 19 | 19 | 9 | **47** |
> | end of 2026-10-07 (`f2c70e991714`) = start of 2026-10-08 | 19 | 24 | 11 | **54** |
> | end of 2026-10-08 (`c3931286386f`) | 53 | 44 | 21 | **118** |
>
> Each row is counted from GIT -- `git ls-tree` of that commit, one row per `case.json` -- not
> from a table written at the time. So 2026-10-08's work is **54 -> 118, +64**, and the week's
> is **33 -> 118, +85**.
>
> **CORRECTED 2026-10-08.** An earlier version of this table labelled the 54 row "the inventory of
> 2026-10-06" and added an intermediate 62 row as the day's baseline, concluding "+56 today". Both
> were wrong. The 54 figure was right but its DATE was not -- it is the state at the end of
> 2026-10-07, while the end of 2026-10-06 was 47 -- and 62 was a point in the middle of
> 2026-10-08, after the day's first eight cases, so it is not a baseline for anything. The figures
> had been carried from a plan written earlier the same day instead of being read from history,
> which is exactly the mistake the paragraph above this table warns against.
>
> The counts are now COMPUTABLE rather than asserted: 22 temporal cases carried no `nested`
> boolean at all — the field postdates them — and were backfilled from each case's own
> `allocator_layer`, so no case is left in an "unclassified" bucket. A script that put 11 of 25
> spatial cases into such a bucket on 2026-10-06 reported a nesting share wrong by 16 points, which
> is why the boolean exists.

> **UPDATED 2026-10-07.** The spatial row read *"8 | 3 | 11"* until today. It is now **14 | 18 | 32**:
> FFmpeg went 4 -> 15 (2026-10-06), tshark 6 -> 11 and memcached 4 -> 6 (2026-10-07). The
> not-nested column grew most, because requiring liveness at the pin had been keeping it thin — a
> requirement no document ever asked for, and the retraction of that inference is what unblocked the
> growth. Counts are recomputed from each case's `nested` boolean, which exists so this row cannot
> drift from the tree again.
>
> **CORRECTED 2026-10-05.** The spatial row previously read *"11 built and measured | 0"*, putting
> every built case under *nested allocator*. **That is wrong, and the axis was the problem.** This
> table's axis is **who allocated the object**; §1a's A/B/C axis is **which bound the access
> crosses**. They are orthogonal, and I had collapsed them.
>
> On *this* table's axis: memcached cases 6 and 7 are **nested** — their objects are `slabs.c`
> chunks — even though the bound they cross is a sub-object one inside the chunk. FFmpeg cases 0-2
> are **plain**: `av_refstruct_alloc_ext` is called directly at `cbs_sei.c:257` and
> `decode.c:2352`, *not* from a pool, so the object is one `av_malloc` with no recycling layer.
> Hence **8 nested, 3 plain**, not 11 and 0.
>
> The sub-object distinction is kept in §1a because it is what decides *detectability*, which is a
> different question from *who allocated*.

**Every defect reduced from real upstream code in these three programs is temporal.**

> **RETRACTED 2026-10-05.** This section first continued *"This is not a measurement gap — it is
> what the defect hunt found, and it is the thesis the target evaluation rests on."* **That is
> false, and the refutation is in this tree.** The hunt **could not** have found a spatial defect,
> because spatial wording was a *disqualifier* on its first filter:
>
> - `wireshark-wmem-defect-triage.md:23` — filter 1 is *"the commit message reads as a lifetime
>   defect — use-after-free, freed, stale, dangling — **and not as an overflow**, a leak or a denial
>   of service"*.
> - `ffmpeg-live-defect-triage.md:26` — filter 1 is *"lifetime wording in the subject — 15 of 1,788
>   survive"*.
> - `bug-corpora/memcached/allocator-repros/README.md:52-57` — *"filtered on temporal-safety
>   vocabulary (55 hits)"*.
> - `wireshark-wmem-defect-triage.md:178` — **29 rows** of the 4.6.9 security tracker rejected en
>   bloc as *"spatial or availability"*.
> - `ffmpeg-pool-consumer-defects.md:51-54` — the population was *measured* to be **"genuinely
>   spatial-dominated — "overflow" 2,415, "out of array" 1,062"* — counted in aggregate, then never
>   read case by case.
>
> So the 0 is a property of the **search**, not of the software. A tree-wide census agrees: **0 of
> 78 `case.json` files in `bug-corpora/` carry a spatial shape**, across all eight programs — the
> signature of a single-axis hunt, not of eight clean codebases. A spatial-wording filter over the
> same populations yields **71 candidates on filter 1 alone** (wireshark 33 of 4,321, FFmpeg 17 of
> 1,786, memcached 21 of 2,349).
>
> The honest statement is: **spatial defects were excluded by design and never triaged
> individually.** Five were met incidentally and rejected with reasons — memcached `#1308`
> `raw_line()` (rejected on *reachability*, not class, `allocator-repros/README.md:79`), the 29
> Wireshark tracker rows, opcua `d24613c461`, vp9 `a024f8c541`, tdsc `fd3ee52fab`.
>
> **The hunt has since run. Its result is in §1a below and it is not zero.**

### 1a. What the spatial hunt found (2026-10-05)

Three new instruments, one per program, in `docs/ref/{wireshark,memcached,ffmpeg}-spatial-defect-triage.md`,
driven by `bug-corpora/tools/spatial-triage.py`. The classification that matters is **which bound
the overflow crosses**, because that is what decides whether anything of ours can see it:

| class | bound crossed | who faults |
|---|---|---|
| **A** | the `malloc` bound | `shrink`, `sublet`, CHERI alike — a tie row |
| **B** | a sub-allocation bound **inside a nested allocator's block** | only a ported nested allocator |
| **C** | a **sub-object** bound inside ONE allocation | nothing we have; needs per-member authority |

| program | population | filter 1 | class B | class C | live at pin |
|---|---:|---:|---:|---:|---|
| tshark | 96,806 | 303 | **15** | 0 | **2** (`1d8acb21ab`, `d24613c461`) |
| memcached | 2,349 | 20 | **0** | 0 | 0 — pin *is* upstream head |
| FFmpeg | 1,786 + history | 19 + shape search | 0 | **4** | **4** (`8864fd0aec`, `a809a784ec`, `68845e26f7`, `e058af88ab`) |

**So the spatial row is not empty: 19 class-B/C upstream defects, 6 of them live at their pin.**
Zero are built as corpus cases yet — see the honest comparison below.

**The three programs are blind for three different reasons**, which is the finding worth carrying:

- **tshark — unwired allocator coverage.** The live pair overflows a chunk in `pinfo->pool`, a wmem
  `BLOCK_FAST` allocator, and the chunk port's only wiring patch targets `wmem_block.c`; no patch
  against `wmem_block_fast.c` exists. **But 4 of the 15 are in `wmem_file_scope()`, a `BLOCK`
  allocator the port already narrows** — `0261fd7da6` (HTTP Range, also a whitelisted dissector),
  `d7d1686a95`, `1c090e9292`, `4a4871a831`. Those need no port work to discriminate.
- **memcached — structurally absent.** An item is sized exactly from the key and value lengths at
  `do_item_alloc` with the parser validating them first, so the length bugs land one layer out in
  the proxy's own `malloc`'d buffers. The four `slabs.c`/`items.c` candidates index **static global
  arrays**, not heap.
- **FFmpeg — sub-object granularity, not an allocator problem at all.** All four live defects cross
  a bound *between two members of one struct* inside a single `av_mallocz` or refstruct. No
  allocator adapter can help; the authority that would need narrowing is per struct member. That is
  the taxonomy's `partial²` cell, where CHERI and Capstone already share a verdict.

**Comparison with the temporal hunt, which produced 22 built cases:** this hunt has produced **19
triaged candidates and 11 BUILT, MEASURED cases** — tshark 5, memcached 3, FFmpeg 3. Eleven is not
parity with 22, and this is not the place to imply it is.

| program | built | corpus | measured how | live at pin |
|---|---:|---|---|---:|
| tshark | **5** | `wireshark/wmem-repros` cases 13-17 | QEMU, 12/12 on each of two builds, negative control 12/12 | 2 |
| memcached | **3** | `memcached/allocator-repros` cases 5-7 | **QEMU domain, 8/8, negative control 8/8**, plus native fix-differential 8/8 | 0 |
| FFmpeg | **3** | `ffmpeg/subobject-repros` cases 0-2 *(new corpus)* | **QEMU as probe cases 40-42, PASS/PASS on modes 0 and 2**, plus native 3/3 and ASan blind two-sided | 3 |

**The result is not the count — it is that the three programs divide on WHO CAN SEE these defects**,
and the division was measured rather than argued:

- **tshark's five all fault**, cause **5** on the three reads and cause **7** on the two writes, and
  they do **not** discriminate the chunk port: `wm_narrow()`
  (`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15`) narrows every wmem allocation on every
  arm, so that harness has no malloc-granular arm to contrast against. The contrast is the app
  port's fx12 ladder in §3.
- **FFmpeg's three are caught by NOTHING**, measured: each crosses a bound *between two members of
  one allocation*, so every per-allocation bound is in bounds for it, and ASan is blind with a
  positive control that fires. The `partial²` cell of §4.
- **memcached's three are native-only**: their Capstone readings are declared predictions pointing
  deliberately different ways, so a future reading settles something.

**The measurement was CLOSED over the ELEVEN cases this section counts.**
*(Two further spatial corpora landed on 2026-10-06 — `memcached/plain-heap-repros`
and `ffmpeg/plane-repros` — each with its own result bundle; the totals in
`spatial-and-temporal-bug-inventory.md` are the current ones.)*
Every one of the 80 arm cells across those eleven cases is
accounted for**, and the arithmetic reconciles from the case files rather than from a summary:

| | measured | unavailable | declined | n/a | total |
|---|---:|---:|---:|---:|---:|
| tshark (5 cases × 7 arms) | **15** | 15 | 5 | 0 | 35 |
| memcached (3 × 7) | **9** | 9 | 3 | 0 | 21 |
| FFmpeg (3 × 8) | **12** | 9 | 0 | 3 | 24 |
| **total** | **36** | **33** | **8** | **3** | **80** |

- **measured** — every Capstone arm of all eleven cases, plus the native fix-differentials and
  FFmpeg's ASan arm. **All eleven cases are now measured under Capstone**, not six of them.
- **unavailable** — PoisonCap and CheriBSD, 3 arms × 11 cases. Checked, not assumed:
  `ports/common/cmake/toolchains/cheribsd.cmake:4-9` requires `CHERI_SDK` and `CHERI_SYSROOT` and
  `FATAL_ERROR`s without them; both are unset even after sourcing the project environment, and no
  SDK, rootfs or PoisonCap image exists anywhere on this host.
- **declined** — the eight `native-detect` (ASan) arms, with a reason that is itself a measurement:
  the FFmpeg sub-object probe already shows the two-sided shape (silent inside the allocation, fires
  one element past it), and every crossing in those corpora stays inside one `g_malloc`'d block or
  slab page by the same mechanism.
- **n/a** — FFmpeg's `backing` arm: there is no block distinct from the object, because
  the object *is* one `av_malloc`.

### What the closed measurement shows, per program

- **tshark, 5 cases, all faulting.** Cause **5** on the three reads, cause **7** on the two writes,
  on **both** builds. They do **not** discriminate the chunk port: `wm_narrow()` narrows every wmem
  allocation on every arm, so the harness has no malloc-granular arm. 12/12 per build, negative
  control 12/12.
- **memcached, 3 cases, and the decisive one.** Case 5 **faults** (cause 7, write probe); cases 6 and
  7 **complete on both modes**. Case 6 is the result: the defect is real and corrupts the value's
  storage, the crossing stays **inside** the chunk, the slab port's bound *is* the chunk, and so
  **nothing we have detects it**. 8/8, negative control 8/8 fired.
- **FFmpeg, 3 cases, caught by nothing — now measured, not declared.** Probe cases 40-42 of the
  buffer-pool port, **PASS/PASS on modes 0 and 2**, runner exit 0 each. Each probe asserts the
  crossing happened, so a completion is a measurement and not a quiet nothing.

**Nine pre-registered predictions were refuted across this work** and each is recorded where it was
made, never silently corrected. The last four: case 13's completion predictions; the five tshark
rows' cause 5 (a store faults 7); memcached's *"0 class-B, structural"* verdict; and — the only one
that went the other way — memcached's three Capstone predictions, which **held**, including the one
that mattered.

### The two decisions that remain, and they are the lead's

1. **Wire `BLOCK_FAST` into the chunk adapter.** It would turn the two *live* tshark defects into
   app-port detections. The gap is wiring, not design: the adapter is block-generic, the only wiring
   patch targets `wmem_block.c`, and `BLOCK_FAST` is the simpler allocator.
2. **Whether any of this enters the paper.** `tab:target-security`'s rows are programs and its
   columns configurations, and "Capstone spatial" is a *configuration* scoring 0 on 57 **temporal**
   bugs — so adding spatial defects means new `Cases` and a changed `\targetCorpus`.

**Deliberately not pursued**, so nobody mistakes it for an oversight: tshark's **251** class-`?`
candidates and memcached's **152 of 157** unread item-size commits; and `a809a784ec`, whose
containment is partial.

### 1b. The 29 class-A spatial candidates, and why the "plain" column was empty

Class A is a spatial defect whose access crosses the **`malloc` bound itself**. The triage found
**29** across the three programs — tshark 21, memcached 3, FFmpeg 5 — and for a long while **none**
was built. That was a scoping decision of mine, not an absence in the software, and the reasoning
was: `shrink` catches them and CHERI catches them, so they are a *tie* and add nothing to the
project's claim.

**That reasoning was wrong for what the numbers are for.** A row where every configuration catches
is the **baseline and the denominator**: it is the evidence that the harness and the arms work at
all, and it is the only place CheriBSD can register a spatial hit. Reporting "0 not-nested spatial"
beside "22 nested temporal" invites the reading that the not-nested class does not exist here, which
is the opposite of true — it was simply not pursued.

> **RETRACTED 2026-10-05, in full.** A table stood here headed *"Verified starting points,
> allocation site opened in the pinned source"*, listing four rows. **Every one of the four is
> false.** The sites were afterwards opened, one at a time, in the source each port actually pins:
>
> | row as published | what the pinned source says |
> |---|---|
> | memcached `ddee3e2` — `authfile.c:44` `calloc(1, sb.st_size)` | **already fixed at the pin.** `authfile.c:50` reads `calloc(1, sb.st_size + 2)`, with `auth_end = auth_data + sb.st_size + 1` (56) and an `auth_end - auth_cur` clamp on the `fgets` length (60) |
> | memcached `11b5f9b` — a `realloc`'d array at `proxy_lua.c:1403` | **not heap.** The overflowed object is `char temp[KEY_MAX_LENGTH + 1]` at `proxy_lua.c:717`, a stack array; the classifier matched a `realloc` elsewhere in the same file |
> | tshark `be813ede9d` — `extcap/etl.c:1411` `Message = g_malloc(Length)` | **absent at the pin.** 4.6.8's `extcap/etl.c` is 799 lines and contains no `Message = g_malloc` |
> | tshark `f207d25f4b`, `830cf562a0` — `g_strdup` error strings | **half right, and the wrong half is retracted below.** `830cf562a0` is indeed an integer-underflow subject (*"pcap: Fix an integer underflow."*) that filter 1 excludes by its own wording. The `g_strdup` reading of `f207d25f4b` is false: its subject is *"Don't let the reported length underflow w/ phdr"* and it touches no `g_strdup` |
>
> Nothing was measured wrong and nothing was built on these rows. What was wrong was publishing a
> classifier's output under the word *verified* — the same defect, one level up, as the "29" itself.

**The 29 is a candidate count. The verified count is 0.** Seven of the 29 have now been read
against the pinned source — every one that had been called verified, plus the three tshark rows
that survived a liveness pass. None is a live class-A defect:

| candidate | disposition in the pinned source |
|---|---|
| memcached `ddee3e2` | fix present (`authfile.c:50`, `+ 2`) |
| memcached `11b5f9b` | stack array (`proxy_lua.c:717`) |
| tshark `be813ede9d` | code absent at 4.6.8 |
| tshark `f207d25f4b` | **RETRACTED 2026-10-05:** misattributed, and returned to the unread pool. The row said "`g_strdup` of a string literal" at `wiretap/libpcap.c:621`; that line is the file's only `g_strdup`, but this commit has nothing to do with it. `git show --stat` gives the subject *"wiretap: pcap[ng]: Don't let the reported length underflow w/ phdr"* over `libpcap.c` and `pcapng.c`. Written from a grep in the file instead of the commit's diff — the same defect retracted above, committed again the same day |
| tshark `830cf562a0` | integer-underflow subject, excluded by filter 1 |
| tshark `06d08c5811` | **no access leaves the allocation.** `wsutil/eax.c:150` allocates `worksize`; the loop bound *is* `worksize`. The fix moves where a one-past-the-end address is *formed* — legal C, and nothing dereferences it |
| tshark `7ffc11e38f` | fix present (`wiretap/file_access.c:1322`, `:1376`) |
| tshark `3be1c99180` | fix present (`wiretap/netscreen.c:63-66`, `:311`, `:332`). Also moot: `ws_buffer_assure_space` over-allocates, so a read past `pkt_len` stays inside the allocation — class C, not A |
| **the remaining 22**, `f207d25f4b` among them | **not individually read.** Stated as unread, not as absent. 7 dispositioned + 22 unread = 29: the retracted row is listed for its trail and counted in the 22, not twice |

**Why 0 is a result here and not a gap.** Class A is the class upstream fixes *first* — it is what
fuzzers, ASan and compiler warnings find — and these three programs are pinned at recent releases
(memcached 1.6.45, wireshark 4.6.8). Three of the eight rows above are that mechanism made visible:
the fix is already in the tree we build. The class that survives into a current release is the one
no existing tool sees: a crossing inside a nested allocator's block (class B) or inside a single
allocation (class C). That is the project's own thesis, and it now has a measured denominator
instead of an assumption behind it.

> **RETRACTED 2026-10-06.** What stood here said the not-nested spatial baseline is synthetic
> **by necessity**, and that the zero is "a result, not a gap". **The measurement stands and is
> unchanged: 0 of the candidates read is live at the pin.** What is withdrawn is the inference from
> it to an empty cell, which rested on a premise never stated and never checked — *that a case must
> be live at the pin to be built*.
>
> It is not. **29 of the 35 existing corpus cases carry `live_in_pin: false`**, and the convention is
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
| tshark `7ffc11e38f` | **RETRACTED 2026-10-06: hardening, not a reachable defect.** The row called it buildable because the fix adds `file_type_subtype < 0`, and I read the guard's existence as proof the hole was reachable. It is not. The fix guards three functions, and every caller of all three supplies a validated or construction-valid type: `mergecap.c:257` rejects a negative `-F` before `:392` uses it; `editcap.c:1018`, `:1054` and `tshark.c:3309` pass `wtap_dump_file_type_subtype(pdh)` from an open dump; `file.c:4470` is reached only through the short-circuit `save_format == cf->cd_t &&` at `:4468`, so the value equals the capture file's own type; and the Qt dialog's calls use the format list it built. That list is the complete set of callers in the fix's PARENT tree, not only the ones at our pin. **A bounds check added upstream is not evidence that the unbounded path was reachable** -- that needs a caller, and none was found |
| tshark `f207d25f4b` | no — read at last, from the diff: it replaces `orig_size -= phdr_len` and `packet_size -= phdr_len` with checked `ckd_sub`, so the defect is an **unsigned underflow** of a reported length, the same class as `830cf562a0` that filter 1 excludes by its own wording. Its downstream consequence may be spatial; the defect is not |
| tshark `3be1c99180` | no — `ws_buffer` over-allocates, so the crossing stays inside the allocation. Class C |
| tshark `be813ede9d` | no — a fix-reversal needs the code to exist at the pin, and `etw_dump_write_ldap_event` does not |
| tshark `06d08c5811` | no — unchanged, no access leaves the allocation |
| memcached `11b5f9b` | no — unchanged, a stack array |
| tshark `830cf562a0` | no — unchanged, an integer-underflow subject |

**This does not put tshark's 21 back in play.** Most still fail on capacity-versus-length or on the
code having to exist at the pin; what changed is the criterion, not the evidence.

**One of the three empty cells has been filled; the other two are still empty.** The memcached
row is now `plain-heap-repros/00`, built and measured. The tshark candidate did not survive its
reachability check and was retracted the same day, which is why this table names what each cell
has rather than what it might:

| cell | candidate | the crossing |
|---|---|---|
| memcached, not-nested spatial | **BUILT** — `plain-heap-repros/00` (`ddee3e2`) | an unclamped `fgets` scan leaves `calloc(1, sb.st_size)`; measured two-sided on both native arms, and ASan reports it |
| tshark, not-nested spatial | **none** | `7ffc11e38f` was retracted as hardening; the cell is still empty and its candidates are the 22 unread |
| FFmpeg, **nested** spatial | candidates only — see the section above | a frame-plane crossing, ownership and padding still to be opened |

They are **candidates until built and measured**, and neither is counted in any table yet.


The synthetic probes keep their role regardless: fx2/fx3 and memcached 20/21 show the arms
discriminate at `malloc` granularity, which is a different job from counting upstream defects.

**Not verified, and marked so:** FFmpeg's five were classified by a subagent, **three from the diff
alone without opening the allocation site** (`db05df9d13`, `bde5c6acb6`, `79e10e5196`). They are
candidates, not facts. They were not pursued: FFmpeg already contributes three not-nested spatial
cases (§1a cases 0-2), so a fourth candidate adds nothing the baseline lacks. The two that were
opened are `c79dfd29e6` (h264 `color_frame`) and `8e55f4f3e9` (v210dec `custom_stride`).

**Where a class-A case belongs, and why it is not this corpus.** The corpus harnesses narrow every
allocation their ported allocator makes — `wm_narrow()` on the wireshark side, the chunk carve on
memcached's — so they have **no malloc-granular arm to contrast against**. The app ports do:
fixtures fx2 `heap_neighbour` and fx3 `heap_one_past` read `level0` **RETURN** and `shrink`
**FAULT** in all three. So a class-A upstream defect belongs there as a **port fixture**, which is
also where the five not-nested *temporal* defects already live (memcached 17/18, tshark 14/15,
FFmpeg 24).


## 2. The SYNTHETIC spatial probes — and Sublet does catch most of them

> This section's title used to read *"Spatial exists only as synthetic probes"*. **That is no longer
> true:** §1a's eleven built cases are reductions of real upstream defects. What follows is about
> the port **fixtures**, which are synthetic and remain the only place the *nested-vs-malloc*
> contrast is visible — see §3.

Every spatial probe across the three ports' `app/host/safety-expect.txt`. Fixture numbers differ
per port, so each row names them explicitly.

| probe | `level0` | `shrink` / `sublet` | nested arm |
|---|---|---|---|
| `heap_neighbour`, `heap_one_past` — fx2/fx3 in all three ports | RETURN | **FAULT `oob`** | `chunks`, `slabsublet*` FAULT; FFmpeg pool arms **not registered** for fx2/fx3 |
| `global_oob`, `stack_oob` (controls) — fx7/fx8 in tshark and memcached, **fx8/fx9 in FFmpeg** | FAULT `oob` | FAULT `oob` | FAULT `oob` |
| `global_merged` — tshark fx9, FFmpeg fx10 (memcached has none) | RETURN | RETURN | RETURN (`chunks` too) |
| **tshark fx12 `wmem_neighbour`** | RETURN `c000ee` | **RETURN `c000ee`** | `chunks` **FAULT `oob`** |
| **memcached fx9 `slab_neighbour`** | RETURN `9000ee` | **RETURN `9000ee`** | `slabsublet0/1` **FAULT `oob`** |
| **FFmpeg fx16 `rs_underflow`** | RETURN | **RETURN** | `pool0`/`pool2`/`poolsublet` **FAULT `oob`**; `poolstock` RETURN |

So `sublet` **faults on plain spatial** (fx2/fx3, where `level0` returns). It returns on four spatial
probes: the **three nested-discriminating cells** — tshark fx12, memcached fx9, FFmpeg fx16, the only
three in these programs, 6 `FAULT oob` rows, all measured — **plus `global_merged`, which no arm
catches at all.**

**`global_merged` is a spatial miss that is not an allocator problem**, which is why it sits outside
the nested story. The compiler merges two 64-byte statics into one `.L_MergedGlobals`, so the bound
derived for either covers the pair; reading the second through the first is in-bounds. No allocator
is involved, nested or otherwise, and nothing in a lease mechanism addresses it — **CHERI has the
identical hole**, for the identical reason. It is recorded here so the row is not later miscounted as
a nested-spatial gap. (tshark's own catalogue notes it was added after FFmpeg fixture 8 showed a
global's bounds covering a merged group rather than the object alone.)

**FFmpeg fx14 `pool_one_past` is the instructive non-discriminator.** It faults on `poolstock` too,
because `AVBufferPool` hands out individually `malloc`'d buffers — malloc-granular bounds already
cover one-past-the-end. fx16 is FFmpeg's only genuine **sub-object** shape: the refstruct header and
its payload are one allocation, so a bound on the allocation cannot separate them.

## 3. Why Sublet returns on those three — read off the capability length

The tshark fx12 ladder, one measured `len` per arm, each from a committed bundle:

| arm | measured `len` at fx12 | outcome | bundle |
|---|---:|---|---|
| `level0` | 41 908 912 (~40 MiB, the whole arena) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety/` |
| `shrink` | 8 388 560 (8 MiB) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety/` |
| `sublet` | 1 048 528 (the 1 MiB wmem BLOCK) | RETURN `c000ee` | `ports/wireshark/app/results/2026-09-25-qemu-safety-sublet/` |
| **`chunks`** | **64** (the allocation itself) | **FAULT `oob`** | `ports/wireshark/app/results/20260929-qemu-tshark-step2/`, `.../2026-10-03-qemu-wmem-chunks-arm/` |

**Every layer narrows, and none of them reaches the object** until the nested allocator is ported. One
`g_malloc` hands wmem a 1 MiB region and every chunk carved from it inherits the block's bounds, so
on `sublet` the neighbour `q` lies *inside* `p`'s bound and the write is legal. On `chunks` the same
write faults — `sb`, insn `00c50023`, bounds ending `a5100070` against target `a5100080`.

The chunks bundle's own two-sided check on the adapter, from
`ports/wireshark/app/results/2026-10-03-qemu-wmem-chunks-arm/README.md`:

    chunks  fx12  p  cursor=a5100030  bounds=[a5100030,a5100070)  len-from-cursor=64
    chunks  fx10  p  cursor=a5100030  bounds=[a5100000,a5200000)  len-from-cursor=1048528

fx10 is BLOCK_**FAST**, which the chunk port deliberately leaves alone, and still shows the block
bounds. So the port narrows *selectively*: if it did nothing, fx12 would show block bounds too; if it
did too much, fx10 would not.

The other two cells, same mechanism:

| cell | arms that RETURN | arms that FAULT `oob` | bundles |
|---|---|---|---|
| memcached fx9 `slab_neighbour` | `level0`, `shrink`, `sublet` (`9000ee`, N=3) | `slabsublet0`, `slabsublet1` (N=6) | `ports/memcached/app/results/2026-10-01-qemu-safety/`, `.../2026-10-01-qemu-slab-sublet/` |
| FFmpeg fx16 `rs_underflow` | `level0` (`len=1572752`), `shrink`/`sublet` (`len=64`), `poolstock` (`len=64`) | `pool0`, `pool2` (cause 5), `poolsublet` | `ports/ffmpeg/app/results/2026-09-24-qemu-hardening/`, `ports/ffmpeg/sublet/results/2026-09-29-qemu/` |

### The spatial catches on the `sublet` arm are not Sublet's own mechanism

Sublet **is** revocation, and revocation has nothing to fire on while the object is alive. fx2/fx3
are caught by `shrink`'s per-object bounds, which `sublet` inherits by construction. The three
nested RETURNs are the nested allocator making **extents** invisible to the system allocator
(`malloc`), exactly as it makes **lifetimes** invisible. Same structure, one axis over: a nested
allocator makes both the size and the lifetime of its sub-objects invisible to the system
allocator, and porting it restores both.

## 4. Why there is no spatial advantage to claim over CHERI

Two independent reasons, which must not be conflated.

**(a) Spatial is a tie, by this repo's own committed verdict.**
`table6-cheri-vs-capstone-explained.md:158-162`:

> "**Tie.** *This is the intellectually honest part of the table:* for the spatial / null /
> uninitialised rows, base CHERI is already sufficient — both systems catch them synchronously.
> Capstone claims **no** advantage here. Its advantage is confined to the *temporal* class."

The reason is structural. Spatial safety needs only **bounds**, and the paper's own discussion says
CHERI has them — *"CHERI supports monotonic, unprivileged bounds derivation, so a \nestedalloc can
bound each suballocation … Temporal invalidation requires additional machinery"*
(`sections/06-discussion-and-related-work.tex`, under `\para{From spatial bounds to temporal
authority.}` — quote anchor rather than a line number, since Overleaf moves lines). So a CHERI
nested allocator could narrow to each suballocation exactly as our `chunks` arm does. **The
nested-spatial gap is a porting gap, not a hardware gap** — both systems close it by changing the
allocator.

Temporal is the opposite case: the stale address is legitimate and in-bounds, so no bound helps. It
needs revocation, and revocation *inside* a nested allocator needs the lease mechanism, which bounds
derivation cannot supply.

Do **not** restate this as "CHERI cannot bound sub-objects". The claim is about what the deployed
stack *derives*, not what is *derivable*.

**(b) We have no real spatial defects *yet* — because none were searched for.** The three
discriminating cells are synthetic probes written to measure the adapter, not reductions of upstream
bugs. See the retraction in §1: the earlier wording of this paragraph (*"and the hunt found nothing
to put in it"*) asserted a negative result from a search that excluded the class by construction.

**And reason (a) does not generalise to the nested case, which is where this matters.** The tie is
about spatial defects that cross the `malloc` bound — there bounds alone suffice and CHERI has them.
A defect whose overflow stays **inside** a nested allocator's block is a different cell: the system
allocator sees one block, so `shrink` and `sublet` return, and only a ported nested allocator faults.
That is measured today only by synthetic probes (tshark fx12, memcached fx9, FFmpeg fx16). Whether a
*real* upstream defect of that shape exists in these three programs is **open**, and it is the
question the hunt now under way is meant to answer. If one lands, the "spatial is a tie" framing
needs revisiting — which is the lead's call, not this document's.

## 5. Citing a bundle for one of these cells

The six nested `FAULT oob` rows are measured, but a bundle path only goes into a document after
grepping that bundle for the fixture number **in the format the bundles actually use**:

    fx9: AS PREDICTED: FAULT oob  (predicted FAULT oob; signal 11, len=None)

A column-shaped pattern such as `^slabsublet0 +9 ` matches nothing and returns a clean zero for five
of the six rows. That zero was caught only because the **positive control** — tshark `chunks` fx12,
known present from a README already read — returned zero as well, which indicts the regex rather
than the tree. The verified paths are in the tables above.

Two bundle names are *not* evidence for these cells, though they look adjacent:
`ports/memcached/allocators/results/20261004-qemu-corpus-defects/` and
`ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/` hold **corpus** cases (memcached fx12–16,
FFmpeg fixtures 40–47) and contain no safety-fixture rows at all.

One figure from a subagent report is **not** recorded here because no bundle carries it: the
"208-byte slab carve" for memcached fx9. Re-derive it from the slab port's own source before using it.
