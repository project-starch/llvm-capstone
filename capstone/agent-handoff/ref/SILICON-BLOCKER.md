# The silicon blocker — everything known

**Living document.** Update it whenever a claim is added, refuted, or measured. Every entry
must say how it is known: MEASURED (board), SOURCE (quoted file:line), or INFERRED.
Last updated: 2026-08-01.

---

## 1. The blocker in one paragraph

SQLite does not run on the FPGA. The failure is inside `sqlite3RegisterBuiltinFunctions`:
staged probes show stages 0/1/7/8/9 returning `rc=0` (entry+return, `sqlite3_config(HEAP)`,
MutexInit, MallocInit/memsys5, PcacheInitialize all work), while **stage 10 wedges in 3/3
separate boots** — the only wedge in this campaign established across multiple boots rather
than from a single sample. A "wedge" means the domain emits no marker and the board session
must be torn down.

## 2. Root cause

**NOT FOUND.** No mechanism has survived measurement. Do not present any of the below as the
cause.

## 3. Refuted BY MEASUREMENT — do not revive without new evidence

| hypothesis | how it died |
|---|---|
| `cincoffset` consumes its source | SOURCE: `capstone_flu_unit.anvil:43,:62` return `rs1` unchanged |
| `STC` clears its source register | SOURCE: `capstone_dyn_unit.anvil:427` returns `rs2_v` unchanged |
| carve / rev-node pool exhaustion at entry | MEASURED: 183 carves vs ~1000 budget |
| `LDC` consumes its memory slot | MEASURED: stage 57/58 = 7 (two reads, both non-NULL and equal) |
| the SHA5 wedge is self-inflicted | MEASURED: UNGUARDED `wd51` returned `0xB1`, unchanged |
| array identity ("the Nth array is broken") | MEASURED: `wd60/61/62`, one shared array, only the multi-walk shape failed |
| granule/carve-base misalignment is the cause | MEASURED: `ga60 = 0xC1`, identical with granule-aligned glue |
| "the first data-dependent walk fails" | MEASURED: `wd66 = 2` inverts it; `wd71` bare walk passes 3/3 |
| store ordering / missing fence | MEASURED: `fence rw,rw` before `domain_main` changed nothing |
| binary layout | MEASURED: passing and failing binaries have identical carves, symbol vaddr, and the SAME 21 loop instructions |
| instruction placement | MEASURED: +24/+56 byte padding, identical failure |
| walk COUNT (1 ok / 2 partial / 3 wedge) | MEASURED: confounded — two of the four "3-walk wedges" never entered the domain, and `wd63` runs FOUR walks and RETURNS |
| the dyn unit is blocked on a rev-node query | MEASURED: `wrev=1`/`memwait=1` are ALSO set in the healthy control (`0xd5`); they are resting state |
| rev-node allocator exhaustion at the wedge | MEASURED: `head=413`, `overflow=0` (healthy: 222) |

## 4. Established and reproducible

* **Livelock, not a hang, for at least one probe.** Stage 51 returns `0xB1` — the domain runs
  and RETURNS. MEASURED.
* **The emitted pointers are correct.** `__capstone_cap_init` derives literals at
  `0x6da/0x6e0/0x6e6` — deltas of exactly 6 — across 1544 straight-line instructions with zero
  calls/branches; the one reused register is correctly spilled and reloaded. SOURCE
  (disassembly). Note: proves what is EMITTED, not runtime values.
* **`wd66` is a deterministic reproducer** (7 samples, all `2`): same element walked twice
  through the same pointer, first walk overruns, second terminates; the two loops are
  byte-identical (23 instructions each). MEASURED.
* **`wd71` is a deterministic control** (6+ samples, all `0x45`). Use it in every session.
* **Results are NOT always reproducible.** `wd63` returns `0x0E` and `0x0F` on identical
  back-to-back runs in one boot. Any single-sample conclusion is unsafe. MEASURED.

## 5. TWO wedge populations — never merge them again

MEASURED across every board log:

    sw=225   sw=255                        n    what fails
    0x84     0x98 = trap_seen=1 mcause=24  12   dies in REGION-SHARE, never enters the domain
    0x95     0x89 = trap_seen=1 mcause=9   13   dies INSIDE the domain (mcause 9 = stale entry ECALL)
    0xd5     0x8f                           1   HEALTHY (wd71 returned)

`mcause 24` is a real capability exception (`UNEXPECTED_OPERAND`, `capstone_unit.anvilh:289-291`;
cause `= 23 + code`, `cva6.sv:1357`). So **capability faults DO latch** — the instrument works,
it was pointed at the wrong family. Family A is a genuine fault taken with `mtvec = 0`, hence
silent. Family B latches no new trap.

Every "the blocker wedges N times" count written before 2026-08-01 mixes these.

## 6. Current hypotheses, ranked

1. **Family A (region-share) is a capability exception that is silent because `mtvec = 0`.**
   The monitor never writes `dom_seal[1]` (`sbi_capstone.c:760,782-784`) and slot 1 IS
   `{ctvec,mtvec}` (`csr_regfile.sv:399`). Getting its `mepc`/`mtval` would name the faulting
   instruction. SOURCE + MEASURED (mcause 24 latched).
2. **Capability compression aliasing.** Register capabilities hold compressed metadata whose
   bounds are rebuilt from the CURRENT cursor (`ariane_pkg.sv:692-693`), and `CINCOFFSET` does
   no representability check (`capstone_flu_unit.anvil:41-42`). Effective bounds can slide with
   the pointer at multiples of 2^(E+14). SOURCE + a reimplementation of the arithmetic;
   CONTESTED by one board reading (§7).
3. **Something in Family B that has not been named.** `wrev`/`memwait` are resting state, the
   dyn unit reports `dyn_rdy=1` (idle) at those wedges, and no new trap latches. Genuinely open.

## 7. CONTESTED — do not cite either side as settled

Does a 256-byte global's capability really span >= 1 MiB?

* MEASURED (stage 77, 2/2): `lcc` zimm 3/4 gave `end - start >= 1 MiB`.
* SOURCE: the carve is exactly 256 bytes (`start-gp-captable-interp.S:446-449`; `SPLIT` narrows
  the parent in the same instruction, `capstone_dyn_unit.anvil:140-144`), and ordinary `lbu` IS
  bounds-checked (`load_store_unit.sv:970-971`, cause 28).

Both attempts to settle it (`wd78`, `wd79`) WEDGED. **Settling test, no new instrumentation:**
rerun stage 76 with offset `1024*1024 + 512` instead of `1024*1024`. A fault confirms
compression aliasing; another `0x77` supports the over-grant reading.

## 8. Real defects found along the way (report separately; none is proven to be THIS bug)

* **No timeout/abort on rev-node queries.** `get_node_query_validity`
  (`capstone_dyn_unit.anvil:106-112`) is `send >> recv` with no abort; `get_rev_node`
  (`capstone_rev_node.anvil:36-41`) likewise blocks on `recv mem_ch.read_res`. Any unanswered
  query is an unrecoverable machine hang by construction.
* **`REVOKE_NODE` walks unbounded** — no visit limit, no cycle detection; only exits on
  `depth <= depth_bound`, and an invalid node does not stop it (`capstone_rev_node.anvil:13-34`).
  If it parks, every later query hangs.
* **The rev-node allocator wraps silently.** 10-bit bump allocator, no reclamation; overflow
  drives only a debug LED (`cva6.sv:1185,1652`).
* **Carve base granule misalignment.** idx 170 (`sqlite_heap`, 256 KB, granule 512),
  `base%g = 64`, `len%g = 0`. Simulation: granule-align OFF -> 1 unrepresentable carve, ON -> 0.
  The 2026-07-31 revert note had the failing END backwards. Knob `INTERP_GRANULE_ALIGN=1`.
* **QEMU is STRICTER than the silicon on ordinary loads.** QEMU keeps fat capabilities with
  exact bounds and checks ordinary loads (`trans_rvi.c.inc:286-292` -> `op_helper.c:1107`), so
  spatial violations that land on an alias boundary pass on silicon and trap in emulation. Also
  `RISCV_EXCP_CAP_OOB` (`cpu_bits.h:697`) is defined and never raised — QEMU's OOB `mcause` will
  not match the RTL's 28.
* **`mtvec = 0` in domains** means an in-domain fault has no handler and cannot print. Upstream
  design question; not to be patched unilaterally.

## 9. Instrument and method traps (all of these bit during this campaign)

1. **Never read a debug register only at the failure.** Read it at a SUCCESS first. Three of
   eight bits in `sw=225` are identical in healthy and wedged states; a "signature" seen at four
   wedges meant nothing.
2. **Never read `board-<tag>.log` for results** — it carries accumulated console scrollback.
   Only `PROBE_SCOPED_OUT` is valid.
3. **A wedge ends the session**, so a wedging domain CANNOT be repeated within one boot. Every
   wedging result is a single sample by construction. Repeat across boots, or build a probe that
   RETURNS a marker instead (the stage-51 watchdog is the model).
4. **A domain earns an early slot only if THAT EXACT BINARY has returned before.**
5. **Never wait on a process by name** — `pgrep -f <pattern>` matches the waiting command
   itself. Three deadlocks, ~50 minutes lost. Sequence steps in one script.
6. **`llvm-objdump --disassemble-symbols` silently truncates**; use `--start-address/--stop-address`
   and check the byte count against the symbol size.
7. **Every generated edit must assert its anchor matched** — a silent no-op `replace` produced a
   probe that did not compile.
8. **Stage N contains stage M for M < N** on the normal path; never order a superset before the
   subset it depends on.
9. Build probe batches with `build-stage-probes.sh` — it prints per-artifact hashes and a
   distinct-hash count, so a cached build cannot pass as fresh.

## 10. Next steps

1. **WORKAROUND (highest value for the deadline):** clamp `BUILTIN_LIMIT` in
   `build-sqlite-silicon.sh` and find the largest builtin count that still initialises. A
   minimal existence proof (CREATE/INSERT/SELECT on integers) needs very few builtins. If a
   small limit gets past the wedge, SQLite runs on silicon with a documented limitation.
2. Settle §7 with the `1024*1024 + 512` variant of stage 76.
3. Get Family A's `mepc`/`mtval` — it is a real, latching capability fault and would name the
   faulting instruction directly.
4. Re-take any pre-2026-08-01 conclusion that rests on a single sample.

---

## Workaround attempt 1: clamp builtin registration (2026-08-01)

`BUILTIN_LIMIT=<n>` in `build-sqlite-silicon.sh` clamps how many entries
`sqlite3RegisterBuiltinFunctions` processes. Built limits 1/8/24 at **stage 3** (through
`sqlite3_open`), run with the `wd71` control first:

    wd71   rc = 0x45    control OK
    bl1    WEDGED       (bl8/bl24 never ran -- a wedge ends the session)

**Do NOT read this as "one builtin entry reproduces the bug".** The build script's comment says
`limit=1 wedging -> the construct itself is broken`, but that shorthand assumes the probe is
SCOPED to that function. Stage 3 runs `sqlite3_initialize` AND `sqlite3_open`, i.e. stage 3 is a
superset of stage 10, so a stage-3 wedge at limit=1 is equally consistent with the failure being
somewhere later in `open` that clamping does not touch.

Scoped retest built: `BUILTIN_LIMIT=0` and `=1` at **stage 10**, which stops inside
`sqlite3RegisterBuiltinFunctions`:

* **limit 0 returns, limit 1 wedges** -> a SINGLE builtin entry is a minimal reproducer. That
  would be by far the smallest repro this campaign has produced.
* **limit 0 AND limit 1 both return** -> the builtin construct is fine at small counts and the
  earlier stage-3 wedge is later in `open`; re-bisect there, and the clamp is a viable
  workaround knob.
* **limit 0 wedges** -> the wedge is not in the builtin loop at all; stage 10's boundary is
  reached before any entry is processed, and the whole "RegisterBuiltinFunctions is the wedge
  point" framing needs re-checking.

### CORRECTION: `BUILTIN_LIMIT` was the WRONG KNOB (2026-08-01, MEASURED + SOURCE)

Scoped retest at stage 10: **`BUILTIN_LIMIT=0` WEDGES** (control `wd71` returned `0x45`).
Zero builtin entries processed, still wedged.

That looks like it exonerates the builtin path, and it does NOT. Reading the amalgamation at
the definition (`sqlite3-capstone.c:137217`):

    SQLITE_PRIVATE void sqlite3RegisterBuiltinFunctions(void){
      FuncDef capstoneBuiltinFunc[] = { ... ~72 entries ... };

The array is a **LOCAL** — `build-sqlite-capstone.sh` strips `static` — so it is constructed on
the STACK, straight-line, at run time. `BUILTIN_LIMIT` rewrites only the INSERTION loop bound
(`capstoneI<ArraySize(...)` -> `capstoneI<0`, `build-sqlite-silicon.sh:124-126`). **It never
reduces the construction.** At `limit=0` the full ~72-entry array is still built on the stack
before the (now empty) insertion loop runs.

So `limit=0` wedging is entirely consistent with the STRAIGHT-LINE CONSTRUCTION being the
culprit — which is exactly the R-14 shape (straight-line materialisation of distinct string
constants into a struct array wedges; the same data assigned in a loop, or as a flat pointer
array, is fine).

**Consequences:**
* The "~72 entries / scale effect" theory that motivated `BUILTIN_LIMIT` is untestable with
  that knob and remains neither confirmed nor refuted.
* Stage 10 remains the wedge point, and the suspect narrows to the array CONSTRUCTION rather
  than the hash insertion.
* `SQLITE_STATIC_BUILTINS=1` targets exactly this: it restores `static`, turning the run-time
  stack construction into a compile-time global initialised through `__capstone_cap_init`
  (machinery that already performs 394 capability-leaf stores successfully in this domain). It
  was dismissed earlier as "a regression that breaks even stage 0" — a SINGLE-SAMPLE verdict
  from the period when many single-sample verdicts in this campaign turned out wrong. **It is
  being re-tested at stage 0 and stage 10.**

### Workaround status

**Not yet available.** `SQLITE_STATIC_BUILTINS=1` was tried earlier and is a REGRESSION (it
breaks even stage 0). Builtin clamping has not yet produced a passing configuration. If the
scoped retest shows limit 0 passing, the next question is the largest limit that still passes,
and whether that leaves enough builtins for CREATE/INSERT/SELECT on integers.
