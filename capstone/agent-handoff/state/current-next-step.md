# Current recommended next step

## 2026-07-30 — THE REAL BUG: the interp glue works for exactly ONE global

Not a SQLite bug. Not a size bug. Not a 2 MiB bug. Sorting the silicon results by
the domain's global COUNT gives a clean split with no exceptions:

| count | rungs | silicon |
|------:|-------|---------|
| 1 | beebs_primer1, bigwin, gpsz, gpcp, gptl, gpbg, gppv | **all PASS** |
| 6 | gpstress | wrong value (444323487) |
| 8, 16, 32, 64 | gpn8, gpn16, gpn32, gpn64, bigmany | **all HANG** |
| 1059 | SQLite | HANG |

**Every domain with one global passes; every domain with more than one fails.**

This also corrects a claim in the previous version of this doc. "All five
initializer paths pass on silicon individually" was true but misleading: `gpsz`,
`gpcp`, `gptl`, `gpbg` and `gppv` each have exactly ONE global, so none of them ever
executed the carve loop's second iteration. The five paths are validated; the loop
is not.

The gpn8/16/32/64 evidence was already sitting in an earlier UART capture and was
missed because those rungs were read as an unrelated sweep. Their symptom is
identical to SQLite's and to bigmany's: `domain ID = 0` printed, then total silence,
no fault line anywhere in the capture (`mcause`, `mepc`, `badaddr`, `panic` all
absent). On silicon a monitor fault is `C_PRINT` + `while(1)` and C_PRINT goes to the
RTL trace, so a wedge is indistinguishable from a hang on the console.

### QEMU CANNOT SEE THIS, AND STRUCTURALLY NEVER WILL

`gpn2`, `gpn4`, `gpn8` and SQLite all pass under QEMU with `DOMAIN_GLUE=interp`.
The RTL read (see below) says why this is not bad luck:

  * `helper_cssplit` operates on a full 64-bit `{cursor, base, end}` struct and never
    calls `cap_compress` (`op_helper.c:848-870`). RTL round-trips EVERY capability
    write-back through `compress_bounds` (`ex_stage.sv:1080-1098`), because on silicon
    the compressed 64-bit `cap_metadata_t` IS the architectural register state.
  * On a tagged load QEMU overwrites whatever `cap_uncompress` produced with exact
    bounds from an out-of-band shadow map (`op_helper.c:1128-1140`), so even a
    STC->LDC round trip is lossless in the model.

So any bug in capability bounds handling passes QEMU forever. Same shape as the DELIN
divergence. **A QEMU-green interp result is not evidence about silicon.**

### NARROWED: BUILDING the second entry is what breaks it

Three board sessions, each with a passing control in the same session:

| rung | table | domain reads | result |
|------|-------|--------------|--------|
| beebs_primer1 | 1 entry | slot 0 | PASS (582955588, ~9760 cyc) |
| gpn2 | 2 entries | slots 0 and 1 | FAIL |
| gpn4 | 4 entries | slots 0-3 | FAIL |
| **gpn2use0** | **2 entries** | **slot 0 only** | **FAIL** |
| **gpn2use1** | **2 entries** | **slot 1 only** | **FAIL** |

So it is NOT which slot is read, NOT "two live slots", and NOT the access pattern.
Merely building a second cap-table entry breaks the domain.

That also kills two more hypotheses:
  * *tag granularity corrupting slot 0* -- would have left slot 1 intact, so gpn2use1
    should have passed. Independently refuted in the RTL: `DcacheLineWidth` is 128 bits,
    so one cache line IS one capability, with a per-line `cap_tag_q` bit
    (`wt_dcache_mem.sv:134-136, 409-421`); QEMU buckets tags per 16 bytes
    (`cap_mem_map.c:5-19`). Both exactly 16-byte granular.
  * *the documented register-indexed-load fault*
    (`history/27-07-2026_17-05-00`) as the whole story -- its trigger is stores to more
    than ONE location in domain code, and both these rungs write exactly one global.
    (It may still be a contributing factor; it is not the whole mechanism.)

### IN FLIGHT: split vs store

`INTERP_BUILD_LIMIT=1` on `gpn2use0` -- keeps the 2-entry table geometry and the 2-record
descriptor, but runs the carve loop ONCE (one split, one stc). Control `beebs_primer1` in
the same session (count=1, so the clamp is a no-op for it).

  * PASSES -> the SECOND split or the SECOND stc is the culprit. Next: separate them.
  * FAILS  -> the damage is the table split itself (a 32-byte vs 16-byte table
    capability) or something outside the loop. Next: `gpn1use0` vs `gpn2use0`, which is
    byte-for-byte the same compute and the same slot-0-only access with the ONLY
    difference being the extra descriptor record -- a tighter control than
    beebs_primer1, already built and QEMU-gated at oracle 1463068797.

### HYPOTHESES, with the two already dead

DEAD - *descriptor record order != cap-table index order.* Refuted statically:
`emitGpCaptableTable` and `emitGpCaptableInitDesc` both walk `M.globals()` with the
same filter (`CapstoneAsmPrinter.cpp:857, 938`) and `getGpCaptableIndex` assigns
indices in that same order (`CapstoneISelDAGToDAG.cpp:134-138`). Record i IS slot i.
Would have been a perfect no-op at count 1, which is why it was worth checking.

DEAD - *unrepresentable capability bases.* See below; the glue's splits are exact.

LIVE, in rough order of suspicion:

1. **Linear-capability consumption in the loop body.** `cincoffset rd, rs1, rs2` with
   `rd != rs1` CONSUMES a linear `rs1` (C-4b). The loop does this twice per iteration:
   `cincoffset(t6, t2, x0)` and `cincoffset(a4, s1, t5)`. The glue relies on the entry
   `delin(sp)` making every derived capability NONLIN. If that reasoning is wrong on
   silicon for anything derived across iterations -- particularly `s1`, the blob view,
   which is reused by EVERY copy-path global -- the first iteration works and later
   ones get a null. Note QEMU's `helper_csdelin` was patched to be idempotent while the
   RTL's raises UNEXPECTED_CAP_TYPE, which is exactly the C-13 trap.
2. **A per-split resource.** Every `split` allocates a revocation node, and the pool is
   a fixed 1024-entry BUMP allocator with no reclamation. That cannot explain count=8
   (9 splits), so it is not the primary cause -- but it IS a hard ceiling for SQLite at
   1060 splits and must be fixed regardless.
3. **`s1` bounds after `sp` is narrowed.** `s1` is copied from `sp` BEFORE any split, so
   it still spans the original dom_data while `sp` shrinks under it. Two overlapping
   capabilities is legal for NONLIN, but this is the one piece of state the loop reuses
   across iterations.

### THE RTL BOUNDS RULE, corrected (this cost a wrong fix; do not re-derive it)

`compress_bounds` has TWO branches, selected by `bounds.start == cursor`
(`ariane_pkg.sv:749`):

  * **cursorless** (what `split` always produces -- it sets cursor == base on BOTH
    outputs, `capstone_dyn_unit.anvil:139-144`): the base is returned as
    `start: cursor` verbatim (`ariane_pkg.sv:662-665`), **exact at any alignment**. The
    top is truncated DOWN to a multiple of 2**E where E comes from the highest bit at
    which base and top differ, floored at bit 20 -- so E is 0, and the capability
    exact, whenever base and top share one 2 MiB window.
  * **full** (`ariane_pkg.sv:769-806`, reached only once cursor != base): base
    truncated DOWN and top rounded UP at
    `granule(L) = 1 << (max(0, floor(log2 L) - 12) + 3)`.

C-13 was the second branch (the monitor's `C_SET_CURSOR` moved the cursor off the
base), which is why the granule model belongs in the monitor and NOT in the glue.
Commit 765da7f8 applied it to the glue's carve on a bogus "SQLite is the only domain
with violations" correlation; 91685f14 reverts it and explains why.

Real remaining hazard from that branch: domains are exact BY CONSTRUCTION only while
they are <= 2 MiB and power-of-two-page aligned (`capstone.c:83-84`). Past 2 MiB,
interior splits straddle a window boundary and globals silently get SHORT
capabilities. `check-repr.py` now encodes the cursorless rule and fails a build at
that cliff -- which sits just past SQLite's current size.

### ALSO LANDED

`sqlite_host.c` now prints phase markers around `create_dom`, the region sharing, and
`call_dom`. Without them a wedge in the monitor and a wedge in the entry glue are the
same observation (silence after libcapstone's last line). The monitor's region path
has two more silent `while(1)`s (`sbi_capstone.c:236, 246`) and SQLite is the first
domain to share TWO regions, so that middle case was never exercised either.

### PROCESS RULES (each cost a session)

1. **ONE board session at a time.** Concurrent runners power-cycle each other
   mid-JTAG-load and produce a bootrom loop that looks exactly like corrupt firmware.
2. `pgrep -f 'fpga_driver/run_'` **matches its own command line** -- it reported
   "runners=2" for a single clean run. Use `grep -E 'run[_]'` or check `ps` output.
3. **Do not `rm -f` the board lock** to "clean up" before a run -- that defeats the
   flock the launcher takes.
4. Never rebuild an artifact a live session depends on.
5. Compare artifacts by CONTENT, never size (the stale and current SQLite domains are
   both 1,623,008 bytes).
6. A UART capture is an ACCUMULATING buffer -- it can contain several unrelated
   sessions. Check what fraction predates the run before reading anything from it.
