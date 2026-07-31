# Current recommended next step

## 2026-08-01 — SQLite on silicon: the pool ceiling is GONE; one entry-protocol problem left

### Where it actually is

The domain now runs on the board far past where it ever did. Board run, both binaries
hash-verified before execution:

```
SQ: E/share1 ... SHA5, SHA6, ECSZ     <- SHA6 had NEVER appeared before
SQ: F/share2 ... SHA0 .. SHA5         <- second entry, then silence
```

`SHA6` is "the domain returned from the share entry". The first entry now builds a 179-slot
cap table, runs cap-init, and returns. SQLite does **not** yet produce a row.

### What cleared the old blocker

**R-12 confirmed on hardware, then removed by construction.** Reading the debug-LED mux on
the wedged core (switch-selected, `cva6.sv:874-877`, registers at `:1184-1186`):

```
switches 249 -> rev_node_head[7:0]          = 0x4a (74)
switches 250 -> {overflow, 5'b0, head[9:8]} = 0x80  -> OVERFLOW = 1
switches 251 -> serving_idx[7:0]            = 0x00  (not advancing)
switches 225 -> stall flags                 = 0x84  -> stall_issue = 1
```

head starts at 3 and bumps per split, so head=74 after one wrap = **1095 allocations against
a usable 1021**. SQLite's 1059 carves + table split + ~35 monitor allocations.

Fixed by merging private read-only string constants: **1059 -> 179 carves**, ~215
allocations. `CapstoneMergeStrConstants.cpp`, enabled in `build-sqlite-silicon.sh` only (the
ladder keeps merging OFF so `tab:spatialcost`'s BEEBS geometry is unchanged). The container
cap of **4096 bytes is a representability limit, not a tuning knob** — bounds granule is
`1 << (max(0, floor(log2 L) - 12) + 3)`, so below 4096 it is 8 and the glue's 16-byte carve
alignment gives exact bounds. A single 21,211-byte container is NOT representable and the
domain dies before `domain_main`.

### THE REMAINING PROBLEM, and it is ours not the board owner's

The second entry is a reentry through `__test_reentry`. It needs `gp` back. Two routes exist
and **both are blocked**:

1. **Park and reload** (what the glue does): `stc gp` into the descriptor's gp_slot, reload
   with `ldc` on reentry. Fails on silicon. Clean discriminator: building with
   `INTERP_SKIP_GPPARK` makes **QEMU** fail at exactly the same entry with
   `Cap mem access requires capability, cause 24`. So the parked capability reads back
   untagged on hardware.
2. **Re-derive by re-carving**: tried, reverted. **`sp` SHRINKS across entries** — measured
   via `INTERP_PEEK_SP`:
   ```
   entry 1  base 0x10175e720  end 0x101800000
   entry 2  base 0x10175e720  end 0x1017af5d0   <- 330,800 bytes consumed by the carve
   ```
   The carve splits storage off the top, so after entry 1 the domain holds **no capability
   covering its own globals** and cannot re-derive one. Re-carving from the shrunken region
   puts every global at a new address: the attempt reached `G/enter` and produced no rows.

**So the fix must hand the domain a capability covering the carved storage on each entry.**
`cscratch` is the channel and the monitor is ours (`caplifive-system`), so this is fixable
without the board owner. Note the linearity constraint before designing: `sp`/`cscratch` are
capabilities, `movc` consumes a linear source (C-4b), and `split` consumes too — so
"keep a spare copy of the full region" is not free and needs care.

### RTL context for that design (verified by quote, do not re-derive)

* **No flush/clear on domain switch.** `capstone_dom_switcher.anvil` never references the
  cache/tag subsystem; `grep dom_switch|CAPENTER core/cache_subsystem/*.sv` = ZERO hits.
* **Tag writes are fire-and-forget.** A capability store issues the DATA write, then the tag
  byte as a SEPARATE AXI transaction (`wt_axi_adapter.sv:398-402`); its B-response is
  explicitly discarded (`:846` "silently consume it ... don't signal to dcache"). Nothing —
  no fence, no switch — waits on `tag_wr_pend_q`. QEMU cannot model this; its tag update is
  synchronous. NOT promoted to "the cause": a fire-and-forget window explaining a
  100%-reproducible failure is not obvious, and settling it needs an ILA capture.
* Second candidate, unresolved: RTL keys shadow tags on PHYSICAL address, QEMU's `cm_map` is
  populated pre-translation. Relevance depends on paging state, which was not established.

### Tools and traps that will save the next session hours

1. **QEMU `[CAPSTONE]` debug output goes to the harness `--log-file`, NEVER the console.**
   `run-sqlite-silicon.sh` writes `$CAPSTONE_TMP_ROOT/sqlite-silicon.log`. It only reaches
   the console when an exception dumps it — which is why the lines appear on FAILING runs and
   vanish on PASSING ones. The log is opened `"w"`, so each run TRUNCATES it: copy before
   re-running.
2. **`run-sqlite-silicon.sh:19` and `stage-sqlite-in-rootfs.sh:38` REBUILD the domain
   unconditionally.** A knob passed as a prefix on the build command only is silently
   discarded. EXPORT it, and check the artifact HASH CHANGED before believing any negative
   result. This invalidated four experiments in one day.
3. **Never `until ! pgrep -f <pattern>`** — the loop's own command line contains the pattern,
   so it matches itself and spins forever. Six such loops ran for up to 21 HOURS. Bound every
   poll loop with `for i in $(seq 1 N)`.
4. **Diagnostic knobs, all `#ifdef`-guarded and inert by default**: `INTERP_PEEK_SP`,
   `INTERP_PEEK_GP`, `INTERP_PEEK_SLOT=<n>`, `INTERP_PEEK_CAPINIT_TARGET`,
   `INTERP_SKIP_GPPARK`, `INTERP_SKIP_CAPINIT`, `INTERP_BUILD_LIMIT`,
   `-mllvm -capstone-cap-init-limit=<n>`, `-mllvm -capstone-cap-init-print`. `csdebugprint`
   (funct7 0x43) is **QEMU-ONLY** — never in a board build.
5. **Board probes**: `probe_sqlite_wedge.py` (halt + trap CSRs + disassembly),
   `probe_revnode.py` (allocator state off the LED mux). Read `pc` as
   `0x10000 + (pc - pcc_base)`, and remember a post-trap dump shows M-MODE registers, not the
   domain's.
6. **Gates that now exist**: firmware freshness (decompresses the initramfs and compares
   binaries), stale-boot (`sha256` on board vs local), and a staging refusal for diagnostic
   builds (`gp-carve-count.py` detects a baked `INTERP_BUILD_LIMIT`).

### Provenance answers (do not re-investigate)

* The bitstream is synthesised from **`caplifive-system/hw/rtl` = `caplifive-cva6`**
  (`scripts/build-rtl.sh:29-35`). `capstone-ariane` is referenced by NOTHING in any build.
* We cannot build a bitstream here: no Vivado, and programming is physical.

### The one genuine board-owner question

Draft at `/tmp/capstone/boardowner-revnode-request.md`. It should be trimmed to: **is the
decoupled tag write intentional, and is there any way to drain/fence pending shadow-tag
writes before a domain switch?** No such primitive exists in `capstone_dyn_unit.anvil` or
`capstone_flu_unit.anvil`. Widening the pool is now a nice-to-have (it would restore
per-object bounds for strings), not the blocker.

### Status against the descope ladder

Level 3 is met and unusually well evidenced. Level 1 (existence proof on silicon) turns
entirely on the entry-protocol fix above.
