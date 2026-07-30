# Carve count vs. the revocation-node pool — problem, three options, decision

**Date:** 2026-07-31
**Status:** DECIDED — option 3 (merge private string literals), scope limited to `.L.str*`
**Decision owner:** project lead. Recorded here so it is not re-litigated.

## The problem, measured on hardware

The gp-captable ABI gives every global its own capability. The domain entry glue carves
each one with a `split`, and **every `split` allocates a revocation node**. The RTL
allocator's `head` is 10 bits initialised to 3 (`capstone_rev_node.anvil:160,168`), so the
usable pool is **1021 allocations**; after that it wraps to id 0 and reuses live ids.

SQLite needs 1059 carves. Measured on the board 2026-07-31, reading the debug-LED mux on
the wedged core (switch-selected: `debug_byte_sel = switches[7:5]`,
`debug_reg_sel = switches[4:0]`, `cva6.sv:874-877`, rev-node registers at `:1184-1186`):

    rev_node_head[7:0]           0x4a = 74
    {overflow, 5'b0, head[9:8]}  0x80  ->  OVERFLOW = 1
    rev_node_serving_idx[7:0]    0x00     (not advancing)
    stall flags                  0x84  ->  stall_issue = 1

head starts at 3 and bumps once per allocation, so a final head of 74 after one wrap means
**1095 allocations**: 1059 carves + the cap-table split + ~35 for the monitor's
create_domain / two create_regions / two map_regions / share.

Why it hangs rather than faults: every `stc` blocks on a rev-node query with no timeout
(`capstone_dyn_unit.anvil:395-404`), and id reuse can splice a node into the `next` chain
twice; `REVOKE_NODE` (`:13-32`) has no visit bound and no cycle detection, so it walks
forever and never answers another query. The overflow flag reaches only a debug LED —
no CSR, no interrupt, no watchdog anywhere in the design. It is invisible to software by
construction.

This is R-12, predicted since it was filed and **observed for the first time here**. The
domain measured is byte-identical to the one that passes end-to-end under QEMU in the
silicon config, hash-verified on the board, with no diagnostic clamp and no feature trim.

## The population that causes it

From the built domain's `.capstone_gp_initdesc` (1059 records) cross-referenced with the
object's symbol table:

| class | count |
|---|---|
| `.L.str*` anonymous read-only string literals | **885** |
| named globals (`.rodata` tables, `func.static`, `.data`) | 174 |

So **84% of the carves are anonymous string constants**. 544 of the 1059 are ≤16 bytes.

## Option 1 — widen the RTL pool

Change `head` from 10 to 12 bits and give the node table more room.

* **Not one constant.** ~6 lines in `capstone_rev_node.anvil` (width, two `#{20'd0,*head}`
  zero-extends, the `10'd1023` overflow compare), plus hand-written SystemVerilog port
  widths that are *not* generated from the Anvil source (`ex_stage.sv:110,971`,
  `cva6.sv:953`, `capstone_unit.anvilh:500`), plus the debug-LED packing at `cva6.sv:1185`
  which only carries `head[9:8]`.
* **The memory map is a coupled pair, and this is the real hazard.**
  `MEMORY_TOP = CAP_TAG_MEM_BASE = 0xBC3C_0000` and `CAP_REVNODE_MEM_BASE = 0xBFFF_C000`
  (`ariane_pkg.sv:586-590`). The shadow tag region covers data at exactly 1:16 with **zero
  slack**, guarded by `assert(DATA_MEM_TOP == TAG_MEM_BASE)` and a fatal
  `tag_addr_q < TagMemHardTop` (`wt_axi_adapter.sv:990,995-998`). Moving the revnode base
  down without moving the tag base trips both. The project's own `calculate_memory.py`
  reproduces `0xbc3c0000` exactly and, for a 4096-node pool, yields `0xBC3B_4B4B` — **not
  page-aligned**, and whether the cache subsystem tolerates that is unverified.
* **Node ids are NOT the constraint.** They are carried as 30 bits everywhere — both the
  uncompressed and compressed capability metadata (`ariane_pkg.sv:571,629`), the message
  types (`capstone_unit.anvilh:479-482,494`), and the address formation
  `{22'd0, node_query_addr, 4'd0}` (`ex_stage.sv:1041`). Nothing is too narrow.
* **We cannot build it here.** The Anvil compiler is an external container image
  (`docker.io/corank/anvil:cva6`, reachable, ~1.1 GB) so the SystemVerilog could be
  regenerated; but Vivado is absent, the flow wants 2018.2 on a Genesys 2 (`xc7k325t`), and
  programming goes through Vivado's hardware manager physically at the board. The repo says
  so directly: "A bitstream flash is the ONLY persistent write and is a STOP-and-ask (we
  cannot rebuild a bitstream here)".
* **Two RTL submodules, different remotes.** `capstone/capstone-ariane`
  (`project-starch/capstone-ariane`) and `capstone/caplifive-system/hw/rtl`
  (`project-starch/caplifive-cva6`), both pinned at `4c661222`. The *second* feeds the
  bitstream, so a fix landing only in the first would never reach the board.

**Verdict:** ~half a day of edits, then entirely dependent on the board owner. Also note it
only buys headroom — the unbounded `REVOKE_NODE` walk recurs at the new size.

## Option 2 — regenerate the SQLite amalgamation from canonical sources

`SQLITE_OMIT_*` is only supported when building from canonical sources, because most flags
require regenerating `parse.c` with lemon.

* Toolchain is present and **proven on this machine** (`tclsh` 8.6, gcc 13.2, GNU awk; a
  full lemon + `mksqlite3c.tcl` pipeline has already run to completion for 3.50.2). Only
  `sqlite-src-3530300.zip` (~14 MB) is missing; `fetch-sqlite.sh` currently pulls the
  amalgamation-only zip.
* Payoff: the grammar OMITs (WINDOWFUNC, VIRTUALTABLE, TRIGGER, VIEW, ATTACH, ANALYZE, …)
  are worth roughly 155–280 globals.
* Available today without any regeneration: **`OMIT_INTEGRITY_CHECK` alone removes 49
  globals and links clean** — untried.

**FAILED ATTEMPT, recorded so it is not repeated.** Applying `SQLITE_OMIT_*` to the
*prebuilt* amalgamation does not work. Three distinct failure modes, all measured:
OMIT_TRIGGER/VIEW/CTE/WINDOWFUNC/UPSERT fail to **compile** (the shipped parser still calls
`sqlite3TriggerInsertStep`, `sqlite3WindowListDelete`, `sqlite3CteNew`);
OMIT_PRAGMA/VIRTUALTABLE/ALTERTABLE compile but fail to **link**; and a set that compiled
AND linked cleanly (AUTHORIZATION, TRACE, PROGRESS_CALLBACK, INTROSPECTION_PRAGMAS,
XFER_OPT, COMPLETE, DATETIME_FUNCS, the four optimisation flags, QUICKBALANCE, the two
pragma flags) cut 1059 → 1011 and then **silently broke SQLite at run time**. That last one
cost hours: the resulting fault was investigated as a cap-init bug and bisected to a
specific initializer leaf before the trim itself was tested properly. The trim is now off by
default in `build-sqlite-silicon.sh` with the reason recorded at the definition.

**Verdict:** viable, needs one download, buys enough — but it is SQLite-specific and buys
nothing for the next benchmark.

## Option 3 — merge private string literals (CHOSEN)

Give the 885 anonymous `.L.str*` constants one shared capability instead of 885.

* **Payoff:** 1095 allocations → **~222** (or ~112 if all read-only globals merged). We need
  to shed ~75. This overshoots by an order of magnitude and removes the ceiling permanently
  rather than buying headroom — it scales to whatever benchmark comes after SQLite.
* **Mechanically cheap.** `getGpCaptableIndex` is purely positional over `M.globals()` and
  its cache keys on `global_size()`, so an erase-and-replace invalidates correctly; the
  container gets exactly one slot; interior offsets already lower through `CIncOffset` on
  the loaded slot capability (`CapstoneISelLowering.cpp:9976-9981`). The old objection that
  a private `.L` container breaks the glue is obsolete — the descriptor-driven interp glue
  can name private symbols.
* **Cost, stated honestly and larger than "overread".** The carve is a bare `split` with
  **no permission tightening anywhere** (verified: zero occurrences in the glue), so a
  `.rodata` global's capability is read-*write* today. Merging therefore gives up mutual
  read AND write reach among those 885 objects. A walk off the end of one format string
  reaches the next literal instead of faulting.
* **What is retained:** all 174 named globals keep individual bounds — including exactly the
  objects an overread is interesting on (`zKWText` 666 B, `sqlite3UpperToLower` 274 B,
  `sqlite3CtypeMap` 256 B, `aKWOffset` 296 B) — plus heap, stack, and all cross-domain and
  cross-region isolation.
* **Corpus impact: none.** Every row in `tab:scope` is a heap UAF / use-after-close /
  double-free / null-deref; the two spatial rows in the xlang corpus are both
  heap-buffer-overflow. No row in either corpus is a string-constant overread.
* **Paper impact: none to the prose.** The paper does not claim one capability per object.
  Its spatial claims are "every pointer is a bounded capability, on every access, always"
  (`parts/evaluation.tex:465-466,667-668`), which stays true — a merged string is still
  reached through a bounded capability. The only "per-object" mentions in the paper concern
  the *allocator's* bookkeeping, not globals. Two INTERNAL notes assert a per-object claim
  the paper does not actually make and should be corrected when touched.

## Decision

**Option 3, scope limited to `.L.str*` private constants.** Chosen over the broader
"all read-only" variant so the named `.rodata` tables keep individual bounds; chosen over
options 1 and 2 because it needs no third party, no bitstream, and no download, and because
it eliminates the failure class instead of deferring it.

Options 1 and 2 remain open and are not mutually exclusive with this. Option 1 is the only
one that restores per-object bounds for strings as well, and is worth raising with the board
owner independently — he has said the limits "should be adjustable".

## Caveats to carry forward

1. `tab:spatialcost`'s BEEBS numbers were deliberately re-measured with GlobalMerge **off**.
   The ladder must keep merging off, or those rows must be redone.
2. If the SQLite compatibility sentence is ever tightened to mention per-object bounds, it
   needs a clause covering merged string literals.
3. Independent defect found while investigating, unrelated to this decision:
   `floatdidf_ng.o` emits a **second, orphaned** `.capstone_gp_initdesc` (count = 3) for
   `.L__const.__floatdidf.*`. The glue reads only the first header and never carves those
   slots, so `__floatdidf`'s `ldc gp[0..2]` would resolve to amalgam slots 0-2 — silent
   wrong doubles rather than a fault. Needs its own fix.
