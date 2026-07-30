# Current recommended next step

## 2026-07-30 (evening) — SQLite on silicon: SBI args 1-3 arrive as zero

### Exact position

Deterministic, reproduces in ~4 minutes:

```
Globals offset = 0x140000        host built correctly
SQ: A/dom-ok id=0                create_dom works
SQ: B/mkregion1 / C/mkregion2    both create_regions work
SQ: D/mapped r1=12 r2=14         both map_regions work
SQ: E/share1                     first shared_region_annotated
ECSA                 dom_id  = 0    <- correct, dom_id IS 0
(SHA0)               dom_id  = 0    <- correct
RGID:00000000        region_id = 0  <- WRONG, host passed 12
APRM:00000000        perm      = 0  <- WRONG, host passed 0x1
AREV:00000000        rev       = 0  <- WRONG, host passed 0x2
<silence>
```

**SBI argument position 0 survives; positions 1, 2 and 3 arrive as zero.**

That signature matters: `copy_from_user` would have lost `dom_id` too, so the module's
struct copy is fine. The loss is in argument *registers*, between
`sbi_ecall(EXT, FID, dom_id, region_id, perm, rev, 0, 0)`
(`modcapstone/module/capstone.c:211-212`, verified correct) and the monitor's extraction.

It also explains the hang without any further hypothesis: with `region_id == 0` the bounds
check `region_id >= region_n` PASSES, so `SHAB` never fires and the handler proceeds to
operate on **region 0 instead of region 12** -- the wrong region, with perm=0 and rev=0.

### THE NEXT STEP

Compare how `arg1`/`arg2`/`arg3` are marshalled at the monitor's ecall entry against
`arg0`. Start at the trap glue (`sbi_capstone.S`) and the dispatch in
`sbi_capstone.c` around line 1181.

**The strongest clue is already in the tree:** `DOM_CREATE` works, and it packs its extra
value into the HIGH HALF of an existing argument
(`arg3 = entry_offset | (globals_off << 32)`) with the stated reason that the struct could
not be changed. If the trap glue only marshals a subset of arguments, that packing is the
workaround that has been masking this defect all along -- and every SBI call taking more
than one or two arguments is suspect.

Check specifically: how many argument registers the glue actually saves/forwards, and
whether `shared_region_annotated` is simply past that limit.

### THEN, in order

1. Re-run (~4 min). If args arrive correctly and the share completes, the next stop is
   `call_dom` -- where **R-12** becomes live for the first time: 1,060 glue splits against
   a 1,024-entry rev-node pool whose `head` is 10 bits, so allocation #1025 wraps to id 0
   and reuses live ids **silently** (`overflow_flag` reaches only a debug LED). Predicted,
   never yet observed, because execution has never got that far.
2. If R-12 does bite, the discriminator is built: `INTERP_BUILD_LIMIT` under 1024 keeps the
   table geometry identical while never exhausting the pool.
3. R-12's fix is genuinely large (widen the pool = RTL/board owner; one capability per
   section = ABI change costing the per-object property the paper claims; reclaim on drop =
   RTL implements drop as invalidate-only). That is the point to stop and take stock.

### Tools that now exist -- use them, do not rebuild them

* **Monitor errors print to the UART** (I-4). 28+ tags: `ECSA/ECSZ` dispatch, `SHA0-SHA6`
  through the share handler, `SPLA/SPLB` in `split_out_cap`, `IRQX/EXCX/ILLX` in the
  handlers, plus operand lines. `SHA5` is the last marker before M-mode is left, so
  "SHA5 then silence" exonerates the monitor and implicates the domain.
* **Idle-abort** (`SQLITE_RUN_IDLE=75`): a wedge costs ~75 s instead of 15 min. Validated
  on hardware. Any UART progress resets the clock.
* **`SQLITE_DOM` override**: probe with a bogus domain path, no firmware rebuild.
* **`bigblob`** rung: SQLite's create-time geometry (2 MiB, globals 0x140000, 9,850-word
  blob copy) that PASSES. Use it as the same-firmware control in every session.

### Hard-won rules (each cost real time today)

1. **Tags are numeric constants** -- `capstone-c` materialises them with `lui`+`addi`, so
   grepping the firmware for `"SPLA"` proves NOTHING. Check decimal immediates in the
   regenerated `.c.S`.
2. **Delete `sbi_capstone_dom.c.S`** before any monitor rebuild, or it relinks stale. A
   stale one hid an uncommitted change for hours.
3. **A report line must fit 16 bytes** (the 16550 TX FIFO). `TAG:` + 8 hex + CRLF = 15.
   16 hex digits = 23 bytes and silently truncates the operand. Do not widen; print two
   halves under separate tags.
4. **Absolute paths in board commands.** Five failures today from a path relative to a
   directory the shell had left.
5. **Compare artifacts by CONTENT.** Stale and current SQLite domains are both 1,623,008
   bytes; a size check says "match".
6. **Rebuilding the kernel invalidates `modcapstone`** -- `insmod` then fails and the run
   dies before reaching anything. Order: stage -> `linux-rebuild-with-initramfs` ->
   `modcapstone-rebuild` -> `opensbi-rebuild`.
7. **`build-sqlite-host.sh` must use the caplifive-BUILDROOT libcapstone.** It is the only
   copy with the globals-offset packing; the caplifive-system copy is 5 KB older with
   zero `globals_off` references and silently delivers gpoff = 0x1000.

### Retracted today -- do not resurrect without new evidence

"More than one global fails"; "a 16-byte global fails"; unrepresentable capability bases in
the glue; coarse capability tag granularity; the documented register-indexed-load fault as
this mechanism; two-regions-required; rev-node exhaustion as the CURRENT cause (it remains
a real, predicted, unobserved FUTURE blocker); and "the monitor is at fault" in general --
`bigblob` passes on the identical firmware, so the mechanism is sound.

### Status against the descope ladder

Level 3 -- "SQLite green in the silicon config under QEMU plus a documented board attempt
naming the specific blocker" -- is comfortably met, and the blocker is named to the
argument register. Level 1 (existence proof on silicon) is not.
