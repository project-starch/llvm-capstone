# Spatial vs temporal, nested vs plain — memcached, FFmpeg, tshark

**Scope: the three programs the paper's target evaluation uses.** Other corpora in this tree
(cpython, httpd, PostgreSQL, sqlite, mruby, the cross-language set) are deliberately out of scope
here; `paper-bug-inventory.md` is the whole-tree inventory.

**Read this first if you came here grepping for "spatial".** `spatial` is also the name of a
**measurement arm** — bounds-only Capstone, revocation disabled — and it appears in dozens of
`case.json` files and in the paper as `Capstone spatial 0/57`. That zero is a *configuration*
failing to catch *temporal* bugs. It is not a count of spatial bugs, and it is not evidence that
anything is broken. A grep for the word finds arms, not defects.

## 1. The 27 real upstream defects are all temporal — because spatial was filtered out

| | nested allocator | plain / system allocator | total |
|---|---:|---:|---:|
| **spatial** | **0 — never searched for** | **0 — never searched for** | **0** |
| **temporal** | 22 | 5 | **27** |

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
> Wireshark tracker rows, opcua `d24613c461`, vp9 `a024f8c541`, tdsc `fd3ee52fab`. A hunt is now
> under way; until it reports, treat this row as **unmeasured**, not as zero.

The 22 nested ones are the committed corpus case folders, one per defect:

| program | corpus | cases |
|---|---|---:|
| memcached | `bug-corpora/memcached/allocator-repros/` (`00`–`04`) | 5 |
| tshark | `bug-corpora/wireshark/wmem-repros/` (`00`–`12`) | 13 |
| FFmpeg | `bug-corpora/ffmpeg/pool-repros/` (`00`–`03`) | 4 |

The 5 plain-heap ones are carried as port fixtures rather than corpus folders:

| program | fixture | upstream |
|---|---|---|
| memcached | 17 | `204019d` — `try_read_network` reallocs the connection read buffer, stale pointer survives |
| memcached | 18 | `e779381` — logger close clears the global slot and frees the watcher |
| tshark | 14 | CVE-2026-95391 / `030bf6ad011c` — ZigBee Touchlink, **live at the v4.6.8 pin** |
| tshark | 15 | `6e61bca421` — http2 regex unref, **live at the pin** |
| FFmpeg | 24 | `bc46eab87c4f` — vvc/thread, `fc->ft` not cleared when the frame thread is freed |

FFmpeg fixture 25 is **not** in this list: it is synthetic, and the claim that it was live at the pin
was retracted in `bfb92aea0924`.

**Three titles read spatial and are not.** memcached case `03_a8c4a82787_refcount_overflow...` — an
*integer* overflow driving an early free; FFmpeg case `01_1886c3269d_h264_refs_partial_clear` — a
partial clear leaving a live reference, not an out-of-bounds write; tshark fixture 13
`wmem_chunk_free` — a chunk freed while its block lives on. All three are temporal.

## 2. Spatial exists only as synthetic probes — and Sublet does catch most of them

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

**Every layer narrows, and none of them reaches the object** until the inner allocator is ported. One
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
nested RETURNs are the inner allocator hiding **extents** from `malloc`, exactly as it hides
**lifetimes**. Same structure, one axis over: a nested allocator conceals both the size and the
lifetime of its sub-objects from the layer below, and porting it restores both.

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
allocator sees one block, so `shrink` and `sublet` return, and only a ported inner allocator faults.
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
