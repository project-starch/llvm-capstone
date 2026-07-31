# Current recommended next step

## 2026-07-31 — SQLite on silicon: down to ONE instruction, and it is ours

### Where it actually is

**The domain runs on the board through both share entries into its run entry, then stalls
in `strlen`. No rows.** Position, verbatim from the run-scoped capture (`SQ:` markers are
`sqlite_host.c`'s FIFO-safe markers, ≤16 bytes so they always escape):

```
SQ: A/dom-ok ... SQ: D/mapped
SQ: E/share1 ... SHA6, ECSZ        <- share entry 1 returns
SQ: F/share2 ... SHA6, ECSZ        <- share entry 2 returns
SQ: G/enter
<nothing>
```

Everything up to `G/enter` is new since 2026-07-29 and is solid. What is left is a single
instruction.

### The stall, measured on the CURRENT build (`a41c6a6a`)

`probe_sqlite_wedge.py` with `PROBE_STEPI=1`, pc identical across three `stepi`:

```
pc = 0x81f3cc78  ->  image VA 0x14cc78     (VA = 0x10000 + (pc - 0x81E00000))
ra = 0x81e06b2c  ->  image VA 0x16b2c = sqlite3Strlen30
mcause = 0
a0 = 0x0
```

VA `0x14cc78` is inside `strlen`:

```
14cc6c: movc          a2, a0        <- a0 linear? then a0 := cnull HERE
14cc70: movc          a1, a2        <- a2 still live -> a2 := cnull
14cc74: lbu           a3, 0x0(a2)
14cc78: cincoffsetimm a2, a2, 0x1   <-- FROZEN. operand is not a capability
```

**`movc` is a MOVE, not a copy.** `capstone_flu_unit.anvil:6-27`: when `rd != rs1` and the
source is not `CAP_TYPE_NONLIN`, it writes `cnull` to the SOURCE. The measured `a0 = 0x0`
is exactly what `movc a2, a0` leaves behind if `a0` was linear. So the string capability
reaching `strlen` is **linear**, and register allocation emitted `movc` for a copy whose
source stays live.

Independent support: the previous build (`ad0aca1f`, `-O0`, completely different codegen —
the pointer round-tripped through a stack slot instead) froze at VA `0x14d884`, which is
the *semantically identical* instruction: the capability-cursor increment in `strlen`.
Two unrelated instruction sequences, same failure point.

Also: **no passing silicon rung calls `strlen`** — zero references across all 20 ladder
domains. `strlen` had never executed on this board before SQLite.

**Caveat, do not skip.** In that same dump `a1 = 0xca11ab1ebadcab1e` and
`mstatus = 0xca00000000`. That constant is the AXI **error-slave** response
(`axi_err_slv.sv:25`), so part of the register read went to an unmapped address and is
junk. `a0 = 0x0` is consistent with the mechanism but wants a second reading before it
carries weight on its own.

### Next step

1. Establish whether `cincoffsetimm rd, rs, 0` is a **non-destructive** capability copy.
   RTL `CINCOFFSETIMM` (`capstone_flu_unit.anvil:48-68`) returns its source unchanged via
   `create_result_pack(..., rs1, rd)`, which reads like a copy that does not consume. If
   that holds on both RTL and QEMU it is the natural replacement for `movc` in
   `copyPhysReg` for capability copies. **Not verified — and if it is true, ask why**,
   because duplicating a linear capability is what linearity exists to prevent. That
   question is architectural, not a lowering detail.
2. Depending on the answer, the fix is either a codegen change (non-destructive copy) or a
   `delin` at the point the string capability enters `strlen`.

C-14's existing `-capstone-fix-destructive-copies` does **not** cover this: it only
rewrites `movc` → `ADDI` when the source is provably a scalar integer, and leaves
capability copies alone by design. The gap is that there is no non-destructive capability
copy to emit.

### Ruled out this session — do not re-investigate

* **`-O0` codegen shape.** Rebuilding the string primitives at `-O1` (loop has no
  `ldc`/`stc` at all) changed the stall not at all. Knob kept as
  `SQLITE_SUPPORT_OPT_LEVEL`, default off.
* **Shadow-tag store→load race.** The AXI adapter interlocks: a load needing a tag read
  enters `TAG_WAIT` and holds until every pending tag write takes its B-response
  (`wt_axi_adapter.sv:406-427`). The decoupling is real but ordered. **The drafted
  board-owner question about a tag drain/fence is therefore ANSWERED IN-TREE — do not
  send it.**
* **"`ldc` of a linear capability writes cnull back to memory."** Not what this RTL does
  (`capstone_dyn_unit.anvil:296-352` is a plain load after its checks) and not what our
  QEMU does (`trans_capstone.c.inc:146`). "A double `ldc` consumes a linear cap" is not
  available as an explanation here.
* **The rev-node pool.** R-12 was confirmed on hardware and then removed by construction:
  string merging took SQLite from 1059 carves to 179. Measured at the stall: `head = 219`,
  `OVERFLOW = 0`. It is not the blocker. (Widening the pool needs a licence and a third
  party — last resort, and no longer needed.)

### Known-separate, logged, not fixed

* **No i128 `SELECT_CC` pattern on RV64.** `cond ? capA : capB` aborts ISel with "Cannot
  select" at `-O1`; both `Select_GPRCAP_Using_CC_GPR` matcher entries are guarded by
  `!is64Bit()`. Two-line reproducer and the failed fix attempt are in
  `history/31-07-2026_14-00-00_o1-strlen-refuted-and-i128-selectcc-gap.md`. Blocks
  `-O1`/`-O2` for any purecap code with a pointer select — which matters for performance
  numbers — but not for SQLite, which does not need to be optimised to run.
* `floatdidf_ng.o` emits an orphaned second `.capstone_gp_initdesc` (count=3) whose slots
  are never carved → silent wrong doubles.
* `caplifive.dts:35` gives Linux the full 1 GiB with no `reserved-memory` node while RTL
  reserves `0xBC3C_0000+`.

### Board driver contract

Env `FPGA_URL` **and** `FPGA_FW` (absolute path to the ~17.4 MB
`.../platform/fpga/ariane/firmware/fw_payload.bin`, NOT the 569 KB
`build/images/fw_jump.bin`). Wait on the printed `RUN_DONE`/`PROBE_DONE` +
`BOARD_RELEASED` sentinels — **never `pgrep -f`**, which matches the polling loop itself.
Drivers exit via `hard_exit()`. Full contract: `ref/HOW-TO-LAUNCH-ON-FPGA.md`.

### Build traps that are still live

1. `run-sqlite-silicon.sh:19` and `stage-sqlite-in-rootfs.sh:38` **rebuild the domain
   unconditionally**; a knob passed as a command prefix is discarded. EXPORT it and check
   the artifact hash CHANGED before believing any negative result.
2. QEMU `[CAPSTONE]` output goes to the harness `--log-file`, never the console, and the
   log is opened `"w"` so each run truncates it. Copy before re-running.
3. `run-sqlite-silicon.sh` copies from the hardcoded `sqlite-silicon/` directory, not from
   `OUT_DIR` — setting `OUT_DIR` alone makes it stage the *previous* domain.
