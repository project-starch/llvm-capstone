# R-29 — a plain 8-byte store adjacent to a 128-bit `ldc` of the same granule loses the high half

> **Status 2026-09-10: OPEN. Reproduced on silicon and in RTL simulation on every revision we can
> build, including the one that flew before the current flash. The mechanism is NOT settled — three
> accounts are alive and the two most obvious ones are disfavoured by their own directed arms. No fix
> candidate. Do not read the file:line below as an attribution.**

**If you arrived here with a different symptom, you probably want a sibling.** This folder is about a
*plain* 8-byte store immediately before a *128-bit* load of the same 16-byte granule, where the load
returns the granule's high half stale. It is **not** `../S06-untagged-ldc-stc-high-half/`, whose
defect is the tag path and whose memcpy-shaped half is fixed and passing; R-29 was split out of that
folder on 2026-09-10 when its acceptance for the struct-assignment shape turned out to cite a
different program. It is **not** `../S07-capability-untagged-on-reload/` or
`../S10-write-buffer-forward-residual/`, which are about a capability's *tag* surviving a co-resident
plain store — related machinery, different observable. Registry entry `R-10`'s secondary half (the
`is_cap_req` / `st_wr_cap` OR-reduce) is a **sibling account of this same question** and may close
with it.

## The observable

A struct of one capability and two plain `unsigned long`s, copied by ordinary aggregate assignment:

```c
struct s06agg_s { void *p; unsigned long x, y; };   /* p is 16 B; x and y share the chunk at 0x10 */
s06agg_dst = s06agg_src;                            /* two capability-grained copies */
```

`x` and `y` share one 16-byte granule, so the compiler moves them with a single `ldc`/`stc` pair at
offset `0x10`. The rung returns `64 + r`, bit 0 set if `x` was lost, bit 1 if `y` was lost — so
**64 = clean, 66 = the high half lost with the low half intact, 65 = something other than this
defect**. QEMU returns 64. Silicon returns **66**, and in simulation `y` reads back as **zero**.

## The trigger is ADJACENCY, and that is the whole finding

From the committed image `src/s06agg.dom` (`249118220f8cf37a`), disassembled:

```
10438: sd  a0, 0x18(a2)     src.y  -- the LAST plain store into the granule
1043c: ldc a3, 0x10(a2)     the whole granule, the very next instruction
10440: ldc a0, 0x20(gp)
10444: stc a3, 0x10(a0)
10448: ldc a2, 0x0(a2)      the pointer granule, copied after
1044c: stc a2, 0x0(a0)
```

`src.x` is stored several instructions earlier, with its 4-instruction constant build in between. The
half that is lost is exactly the half whose store is adjacent to the load; the half that had time to
leave the store path survives. Write the same shape with four instructions between the store and the
load and it passes.

## Evidence

**Silicon** (`../../board-results/2026-09-05.tsv`, bitstream `caplifive_r25r26r27_66c4e7517.bit`):

| boot | firmware | reading |
|---|---|---|
| sw46 | variant D (three CCSRRW `fence.i` dropped, 149 linked) | `s06agg` **66** (oracle 64); `k800` = 4 and six BEEBS rungs at their oracles first |
| sw48 | the M-3 monitor, every `fence.i` present (152 linked) | `s06agg` **66**; `k800` = 4, `s06copy` = 32 first |

Two firmwares, one bitstream, the same reading — so it is not firmware. **There is no reading of this
image on any earlier bitstream**: the row labelled `s06agg` in boot B1 is a different program
(`32e13a81cf52d4d8`, returning 15, outside this program's 64..67 range), which is what made the S-06
folder's acceptance for this shape unsupported.

**RTL simulation** (`sim/s06agg-shape.S`, `sim/s06agg-shape-RECORDS.md`; delay-40 model, capability
mode after `CAPENTER`, the rung's exact instruction order):

| revision | what it is | adjacent | four apart |
|---|---|---|---|
| `5097eb166` | the silicon that flew BEFORE the current flash | **FAIL 11**, 1984 cyc, `y` = 0 | PASS, 1828 cyc |
| `ef5a8eaf2` | the line this cycle branched from | **FAIL 11**, 1984 cyc, `y` = 0 | PASS, 1828 cyc |
| `66c4e7517` | the flashed bitstream | **FAIL 11**, 1984 cyc, `y` = 0 | PASS, 1828 cyc |

The plain control is intact in all six runs. **So this is not a regression of the current
bitstream** — it is present on the previous one too, and was simply never measured there.

## Mechanism — three accounts, none settled

A claim-auditor pass on 2026-09-09 returned **PLAUSIBLE-BUT-UNPROVEN** on the first account, and the
directed arms since have disfavoured two of the three.

1. **The write-buffer overlay is word-gated.** A cache line is one 16-byte granule
   (`DcacheLineWidth = 128`); a 128-bit `ldc` takes the low word on `rd_data_o` and the high word on
   `rd_user_o` from bank 1 (`wt_dcache_mem.sv:279`). The overlay compares at WORD granularity
   (`:283`), a granule-aligned `ldc` selects word 0, and `wbuffer_be` (`:335`) gates **both** outputs
   — `:394` low and `:397` high. A plain `sd` to word 1 never sets that hit. The file already
   documents this hazard for the TAG at `:286`, and S-10 fixed the tag side with `wbuffer_gran_oh` /
   `wbuffer_gran_clr` into `rd_ctag_o` (`:296-315`, `:379`) without touching the data path.
   **Disfavoured**: `sim/` arm `r29-sep-forceres` forces residency and predicted FAIL 11; it read
   PASS.
2. **A second defect at the same line.** A plain store's write-buffer `.user` is provably zero
   (`store_unit.sv:363`), so a *resident plain word-0 entry* sets `wbuffer_be` to all ones and drives
   all eight lanes of `rd_user_o` from `.user = 0`. Any fix must refuse the `.user` overlay for a
   non-capability entry, not merely add a word-1 term. **Disfavoured**: arm `r29-sep-userzero`
   predicted FAIL 11; it read PASS.
3. **The store buffer's disambiguation is word-granular too** (`load_unit.sv:297` takes the page
   offset from `vaddr[11:0]`; `store_buffer.sv:279/287/293` compare `[11:3]`), so a 16-byte `ldc` at
   `…010` is not held for a pending store at `…018` and the `sd` may still be in the **store buffer**,
   never having reached the cache at all. **This is what is left standing.** It also fits the timing:
   a store-buffer window is a handful of cycles, which is what one-instruction adjacency looks like.

A fourth possibility is not excluded: the miss-refill leg at `wt_dcache_mem.sv:354-358`, since the
source region is never read before the `ldc` and this cache does not write-allocate.

**Why the two PASSes are a direction and not a refutation.** Neither arm observes where the store
physically was at the read cycle; both are equally consistent with the separator load having drained
it all the way to the array, in which case neither created its condition. That is the same shape as
the two retractions this defect has already produced (see below), so it is recorded as narrowing, not
as proof.

## Retractions on this defect, kept because they are the useful part

* **2026-09-09, first sim run reported PASS on three revisions.** The test contained the shape but
  had four instructions between the store and the load, so it never created the triggering condition;
  a clean result was briefly reported as exoneration. Rewritten in the rung's order, it fails.
* **2026-09-09, `r29-lowword` was offered as a discriminating arm and withdrawn.** Its oracle is
  inverted — account 2 predicts that arm FAILS, so its PASS means the entry was not resident — and it
  is not a one-variable pair, because its store and load share `[11:3]` and the load stalls (+56
  cycles).
* The apart-PASS run's log was later overwritten, so that particular result rests on
  `sim/s06agg-shape-RECORDS.md` rather than on a re-readable artifact.

## What would settle it

1. **A waveform probe on the existing `sim/s06agg-shape.S` run** — write-buffer hit, byte enables,
   refill valid, the high-word lane and the store-buffer queues, all sampled at the wide load's read
   cycle. That observes rather than infers and is the step that turns a direction into a verdict. The
   same probe can sample `is_cap_req` (`wt_axi_adapter.sv:196`) and `st_wr_cap`
   (`wt_dcache_mem.sv:138`) and answer R-10's secondary half in the same run.
2. **A distance ladder on the board** — the same rung with 1, 2 and 4 filler instructions between the
   `sd` and the `ldc`. Nothing has measured the residency window on silicon; the step is the datum.
3. **Then a fix candidate**, which must cover whichever account (1) selects *and* the `.user = 0`
   overlay. Note the constraint the RTL records: the tag-side term took UNOPTFLAT 39 → 40 across three
   formulations (`wt_dcache_mem.sv:384`), so this is synthesis-first — lint at baseline, auditor,
   synthesis, then one bitstream.

**Acceptance for a fix:** the rung goes 66 → 64 on the board, the distance ladder collapses, the sim
arm passes adjacent while the apart case is unchanged, and the 88-row sweep is status- and
hash-identical.

## Impact and the workaround in force

Any code whose last plain store lands in the high word of a granule immediately before a
capability-grained copy of that granule — which is what the compiler emits for a struct assignment
whose trailing scalar is initialised just before the copy. `W-12` (`SQLITE_GRANULE_GUARD`,
`../../workarounds/CLASSIFICATION.tsv`) is **KEEP** on this entry's account; it was previously marked
for retirement alongside `W-04`, and that was corrected on 2026-09-09. `W-04` (the memcpy high-half
fixup) is unaffected — the memcpy loop does not have this shape.

## Files

| path | what it is |
|---|---|
| `src/s06agg.dom` | the 10 KB board reproducer, frozen and checksummed (`249118220f8cf37a`) |
| `src/s06agg_kernel.h` | the source, and why 66 rather than merely "wrong" is the signature |
| `src/s06agg_{app,fpga_app,host}.c` | QEMU-side app, board-side rung, native oracle (64) |
| `sim/s06agg-shape.S` | the directed test that creates the shape in the rung's instruction order |
| `sim/s06agg-shape-RECORDS.md` | the six runs across three revisions, the codes, the fidelity caveat |
| `run.sh` | `verify` the frozen artifacts, or print the board staging for the rung |

The rung image is **frozen and checksummed on purpose**: this platform has a per-image entry stall
(`../R16-entry-stall/`), so a rebuilt image is a fresh draw that may simply never run. The board mode
runs the known-good control `k800` first — a boot whose control fails carries no verdict about
anything.
