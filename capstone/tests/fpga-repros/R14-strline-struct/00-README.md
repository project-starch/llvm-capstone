# R-14 reproducer — straight-line struct-array init of distinct string constants

Board: Genesys2 CVA6+Capstone, bitstream `working-caplifive-captype-fixed.bit`.
Measured 2026-07-31. gp-captable ABI, domains built by
`capstone/benchmarks/sqlite/build-sqlite-silicon.sh`.

## The four variants and what they show

| variant | shape | board result | expected |
|---|---|---|---|
| A | 16 distinct literals, **straight-line**, `struct{2 ptr}[64]` | **WEDGE** — no return, no output | 16 |
| B | 4 distinct straight-line + loop filler, same struct | **returns 4** | 16 |
| C | 16 distinct via **loop from a static table**, same struct | returns 16 | 16 |
| D | 16 distinct **straight-line**, flat `const char*[64]` | returns 16 | 16 |

So the failure requires **both** straight-line materialisation **and** the struct element
type. Either one alone is fine (C and D).

**B is the most useful variant**: it does not hang, it returns a WRONG VALUE. The four
straight-line entries pass and the twelve loop-assigned ones fail their
`z && y && strlen>0` check. Same construct, silent corruption instead of a wedge.

## How to run

The `.dom` files are complete domains; run each with the host loader:

    /test-domains/sqlite_host.user /test-domains/<variant>.dom

The host prints `SQ: obs=<decimal>` on return. The value encodes `0x5A6E_ssrr`, where
`ss` = stage and `rr` = the returned count. A wedge prints `SQ: G/enter` and then nothing.

Batch several in ONE boot with
`capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py` and
`SQLITE_STAGE_DOMS=<comma-separated paths>`; put variant A **last**, because a wedged
domain takes the core with it and everything after it is lost.

## Wedged-core state (variant A)

Read off the debug-LED mux, selectors verified against `cva6.sv:1090-1215`,
`debug_byte_sel=0b111`:

    sw=224   privM=1  flu_ready=1  dyn_ready=1  lsu_ready=1  ex_commit.valid=0
    sw=225   stall_issue=1, all other status bits 0
    sw=255   trap_seen=1 mcause=9   <-- STALE: see caveat
    COMMIT pc = 0x81f3c71c  ->  image VA 0x14c71c  (the bnez closing strlen's loop)

**Caveat on `mcause=9`:** the latch records any trap except cause 0 and cause 2
(`cva6.sv:1078-1083`). Linux runs in S-mode and issues SBI ecalls constantly, and
`ENV_CALL_SMODE` is cause 9, so this value is almost certainly stale kernel traffic and is
NOT evidence about the wedge. Clearing the log first (`switches=191`, `cva6.sv:984`) does
not isolate it either, because Linux keeps trapping between the clear and the wedge.
What it does support: a domain capability fault would be cause 24-28, which is nontrivial
and would have overwritten the 9 — it did not, so no capability exception was committed.
Note also that cause 2 (illegal instruction) is invisible to this latch.

## Candidate mechanism, NOT established

See `history/31-07-2026_18-30-00_ldc-load-syncer-arming-leak.md`. A verified one-line
asymmetry in `capstone_dyn_unit.anvil` (LDC's `NOT_CAP` arm at `:306` never calls
`abort_accumulation_load`, while STC's at `:369-370` does) would leave the load syncer armed
on a stale 3-bit `trans_id` and silently swallow a later unrelated load. That matches the
observed "stalled at issue, every unit ready, nothing committing" state exactly.

It is **not** confirmed as the cause of these variants: that arm raises cause 24, which
would have overwritten the latched 9, and it did not. Variant B's *selective data
corruption* also fits a syncer leak poorly — a swallowed load hangs, it does not return
partly-wrong data.

## Open question (not answerable from this tree)

Does a pipeline flush reset `req_set` / `cap_trans_id` in the load and store syncers
(`capstone_dyn_unit.anvil:521-522`)? If not, any capability access abandoned between
`send cap_load_ri.init(...)` (`:302`) and its `req`/`res` pair (`:343-345`) leaves an
8-value comparator armed that will match and consume an unrelated later load. This cannot be
answered from our tree: only the `.anvil` is present, no generated Verilog.

---

## Rebuilding the four domains

The `.dom` files are not tracked (~1.5 MB each, 6 MB total — each is a full SQLite build).
Rebuild from this tree:

    export SQLITE_SUPPORT_OPT_LEVEL=-O1
    for S in 18 20 21 22; do
      OUT_DIR=/tmp/capstone/sqlite-s$S DOMAIN_EXTRA_DEFS="-DCAPSTONE_SQLITE_STAGE=$S" \
        bash capstone/benchmarks/sqlite/build-sqlite-silicon.sh
    done

    stage 18 -> variant A   WEDGES
    stage 20 -> variant B   returns 4, expected 16
    stage 21 -> variant C   returns 16 (correct)
    stage 22 -> variant D   returns 16 (correct)

`strline_struct_repro.c` in this directory is the extracted, standalone form of those four
variants; the live versions are `run_sqlite_staged()` in
`capstone/benchmarks/sqlite/sqlite_capstone_domain.c`.
