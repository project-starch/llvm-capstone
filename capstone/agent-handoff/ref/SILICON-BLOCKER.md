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

## 8b. RETRACTED — the "cap-init store threshold" was a FOUR-WAY CONFOUND

**The 1222..1263 threshold is WRONG. Do not cite it. Do not rebuild on it.**

Verified against the primary logs and the monitor source, not argued:

**(a) `pad200` never ran cap-init at all.** In `board-pad.log` its last marker is
`SQ: C/mkregion2SPLB:0000E006` — it died during REGION CREATION, with no `D/mapped`, no
`E/share1`, no `G/enter`. In `board-bis.log` the last domain (`SQ: id=4`) shows
`SHA5:00000004` and no following `G/enter`. In neither run did `__capstone_cap_init` execute a
single `stc`. A threshold in cap-init stores cannot be measured by a domain that never reaches
cap-init.

**(b) The failure site is a DELIBERATE MONITOR SPIN, not silicon.**
`sbi_capstone.c:494-504`:

        if (base + len == region_end) {
            if (base == region_base) {
                // matching region. We don't support this for now
                capstone_report(CAPSTONE_TAG_SPLB, CAPSTONE_ERR_SPLIT_EXACT);
                ... capstone_uart_flush();
                while(1);

`CAPSTONE_TAG_SPLB` = "split_out_cap: exact-fit region unsupported" (`:44`),
`CAPSTONE_ERR_SPLIT_EXACT = 0xe006` (`:58`). The predicate is a HOST MMAP ADDRESS COINCIDENCE —
the requested region exactly matching an existing one. It is an UNIMPLEMENTED CASE in our own
monitor, upstream of the domain and upstream of cap-init.

**(c) Store count does not determine the outcome.** Direct counterexample, measured on the
artifacts: `wd77`, `wd78`, `wd79` all have **exactly 1048** cap-init stores and are three
distinct binaries (md5 `9c9bd9c3…`, `fbaef16d…`, `c59b396a…`). `wd77` RETURNS; `wd78` and
`wd79` WEDGE. Same count, opposite outcomes.

**(d) The ladder confounded pad count with SEQUENCE POSITION.** `pad200` was always run LAST and
always had the highest domain id (`SQ: id=3`, then `id=4`). Monitor region ids grow monotonically
through a boot (`rgid=12 → 17 → 23 → 25 → 29`). No low-pad build was ever run last, and `pad200`
was never run first.

**What actually survives:** nothing about cap-init store counts. The `sb0` "corroboration" (1257
stores, wedges at entry) is the same confound — it also fails before the domain runs.

**Method failure to learn from:** the ladder looked like a controlled experiment because ONE
variable was varied deliberately. Three others varied with it (image size, sequence position,
host allocation state), and the outcome was never checked against the marker trail to confirm
the domain even reached the code under test. **Always confirm the failure SITE (marker trail)
before attributing a wedge to the thing the probe varies.**

---

## 8c. What the same investigation established instead (offline, verified)

* **RTL fixed-capacity structures are ELIMINATED as the mechanism.** Verified inventory:
  scoreboard 8, store buffer 4/4, `MaxOutstandingStores` 7, wbuf 8, AXI MetaFifo 4, dcache 2048
  lines, icache 1024, rev-node pool 1024, dom-switcher 67, TLB/PMP 16. A sweep of `core/` and
  `corev_apu/` finds NO depth in 1222..1263 and no byte bound in the corresponding range.
* **Tag memory is ELIMINATED.** `ariane_pkg.sv:586-590` and `wt_axi_adapter.sv:148-149`:
  `tag_addr = TAG_MEM_BASE + ((paddr - DATA_MEM_BASE) >> 4)` — a pure function of physical
  address, with no counter, allocator or high-water mark, so store COUNT cannot advance it.
  The arithmetic also shows the shadow tag region ends exactly at `CAP_REVNODE_MEM_BASE`
  (`0xBC3C_0000 + ((0xBC3C_0000-0x8000_0000)>>4) = 0xBFFF_C000`), abutting the node table with
  zero slack — it cannot overflow into it.
* **A real codegen shape change exists but is on the WRONG side of the boundary.**
  `__capstone_cap_init` grows `0x2c98 → 0x41f4` and `ldc` count `111 → 1413` between `pad150`
  and `pad175`, i.e. the register allocator collapses into per-leaf spill reloads. But `pad175`
  is already fully in the new shape and RETURNS, so it is not the wedge boundary. Its real
  consequence is that the ladder rungs are NOT one monotone family.
* **NEW, and independently actionable: `split_out_cap` cannot handle an exact-fit region.**
  That is a genuine unimplemented case in our monitor (`sbi_capstone.c:496-504`) that hangs the
  board whenever the host allocator happens to hand back an exactly-matching region. It is
  almost certainly responsible for a share of the "random" wedges in this campaign, and it is
  OURS to fix, not the hardware's.


(HISTORICAL, RETRACTED — kept only so the reasoning can be audited.) Stage 80 is an
entry-and-return domain that touches one array element and returns. The claim WAS that the only
variable across these builds is how many capability leaves `__capstone_cap_init` must store
(`CAPINIT_PAD=N` adds N initialised pointers). No SQLite code runs at all — no strings, no
walks, no hash tables, no allocator traffic.

    domain    cap_init stores   result
    wd71            1048        rc = 0x45   (independent control)
    pad1            1015        rc = 0x61   RETURNS
    pad120          1134        rc = 0x61   RETURNS
    pad200          1263        WEDGED
    pad260          1381        (never ran -- the wedge ended the session)

**There is a threshold between 1222 and 1263 capability stores in cap-init.** (CONFIRMED in a
second, independent boot — see the bisection below.)

    pad120   1134   rc=0x61   PASSES
    pad150   1184   rc=0x61   PASSES
    pad175   1222   rc=0x61   PASSES
    pad200   1263   WEDGED    (wedged in TWO separate boots)

`pad200` wedging twice, in different sessions, clears the single-sample caveat: this is the
only wedge in the campaign reproduced across boots other than stage 10 itself.

Cross-check: `sb0` (STATIC_BUILTINS at stage 0) has **1257** stores and wedges AT ENTRY — inside
the same band, from a completely different source change. Two independent routes to the same
region.

This is the first mechanism in the campaign with a NUMBER attached and no SQLite logic in the
path, which also makes it the first that could be handed over as a hardware-side reproducer.

**Caveats, stated plainly:**
* `pad200`'s wedge is a SINGLE sample — a wedge ends the session, so it cannot be repeated
  within a boot. Confirm across separate boots before quoting the number.
* The bound is wide (1134..1263). Narrow it before reporting.
* It does NOT explain the in-domain (`0x95`) family on its own: `b10n0` wedges at stage 10 with
  only 1017 stores, below the passing 1134. So either there are two mechanisms, or store count
  is a proxy for something else (total leaf bytes, a specific leaf, an address pattern).
* 1024 — the rev-node pool size — is NOT the boundary: 1134 stores passes.

**Next:** bisect 1134..1263 with intermediate pads, repeat `pad200` across boots, and then ask
what is exhausted at that count. Candidates: a fixed-depth structure in the store path, a
tag-cache capacity, or total bytes rather than store count (vary leaf SIZE at constant count to
separate those two).

CORRECTION on that last point: with this probe design, store count and capability BYTES are
proportional by construction — every cap-init leaf is one 16-byte capability store — so they
cannot be separated by varying the pad. What the ladder DOES isolate is stores from CARVES: the
pad is a single array, i.e. ONE extra carve carrying N extra stores, and the carve count is
therefore constant across the ladder. The threshold is in the store count, not the number of
globals.

## 8d. Session close-out 2026-08-01 (post-reflash)

**Board reflashed and verified.** The resident NV bitstream read as `None` and the board came up
on a STOCK OpenPiton+Ariane design — no Capstone bitstream resident. Reflashed to
`working-caplifive-captype-fixed.bit` (90 s), verified by re-reading rather than trusting the
flash call, and it persisted across reconnect. **Check the resident bitstream at the START of a
session; do not assume continuity from the previous one.** The runners' hard-stop would have
caught it, but only after a wasted build.

**Post-reflash health is good:** `wd71` returns `0x45`.

**`wd66`'s decode is STILL UNVALIDATED.** `wd81` — the probe built to return the two raw guard
values instead of a bitmap — WEDGED, so the question stands: `wd66 = 2` has been read as "first
walk overran, second succeeded" on the strength of a bitmap decode nobody has checked, and
`rc = 2` is equally consistent with a clobbered accumulator. **Do not cite the first-walk anomaly
as established.** `wd81` differs from `wd66` only in trivial ways (loops back-to-back, clamp at
the end instead of a conditional between them) and wedges where `wd66` returns — another
instance of the unexplained build-to-build sensitivity.

**The SQLite blocker is NOT the SPLB monitor spin.** Every stage-10 run reaches `SQ: G/enter`
and then wedges in-domain, so it is a genuine Family-B failure. SPLB is a SEPARATE, ours-to-fix
defect (`sbi_capstone.c:496-504`, exact-fit region unsupported -> `while(1)`) that corrupted the
`pad200` results and plausibly other "random" wedges in this campaign.

**Operational error to avoid repeating:** a cleanup glob `sb*.dom`, written for the `sb0`/`sb10`
probes, also matched **`sbi.dom`** — a package-installed domain. Deleting those desyncs
buildroot's stamps and previously caused six consecutive boot failures. Recovered by clearing
`.stamp_target_installed` for `capstone-sbi-domain` and `capstone-test-domains` and rebuilding.
**Enumerate probe names explicitly; never prefix-glob in the staged tree.**

## 8e. MINIMAL REPRODUCERS (copy-pasteable)

All builds go through `build-stage-probes.sh`, which prints per-artifact hashes and a
distinct-hash count so a silently-cached build cannot pass as fresh. Common preamble:

    cd <REPO-ROOT>            # the llvm-capstone checkout
    source capstone/tests/capstone-test-env.sh
    export FPGA_URL="$(cat ~/.claude-c/secrets/fpga-console-url)"
    export FPGA_FW="$PWD/capstone/caplifive-system/sw/buildroot/build/build/opensbi-custom/build/platform/fpga/ariane/firmware/fw_payload.bin"

**Before any run:** confirm the resident bitstream is `working-caplifive-captype-fixed.bit`.
It was found to be `None` (stock OpenPiton+Ariane) at the start of a session.

### R1 — THE BLOCKER: stage 10 wedges (3/3 separate boots)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 10
    # stage into the initramfs, rebuild firmware, then:
    export SQLITE_STAGE_DOMS="/test-domains/wd71.dom,/test-domains/wd10.dom"
    export SQLITE_STAGE_TIMEOUT=180 PROBE_SCOPED_OUT=/tmp/capstone/r1.txt
    python3 capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py

Expected: `wd71` -> `rc=0x45`; `wd10` -> WEDGED, having printed `SQ: G/enter` (it DOES enter the
domain). Source: `sqlite_capstone_domain.c`, stage 10 = `sqlite3MallocInit()` +
`sqlite3RegisterBuiltinFunctions()`. Stage 9 (`MallocInit` + `PcacheInitialize`) returns `rc=0`.

### R2 — the deterministic wrong-answer reproducer (7 samples, all `rc=2`)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 66
    export SQLITE_STAGE_DOMS="/test-domains/wd71.dom,/test-domains/wd66.dom,/test-domains/wd66.dom"

Stage 66 walks `capstone_probe_lit[1]` ("rtrim") TWICE through the same pointer and returns a
2-bit map. Observed `rc=2` on every sample. **The decode is UNVALIDATED** — `2` reads as "first
walk overran, second terminated", but is equally consistent with a clobbered accumulator. The
two loops were verified byte-identical (23 instructions each, `0x36994` / `0x36a40`).
`wd81`, built to return the raw guard values instead, WEDGES.

### R3 — the health control (6+ samples, all `rc=0x45`)

    PROBE_DEST=/tmp/capstone/repro bash capstone/benchmarks/sqlite/build-stage-probes.sh 71

Stage 71 walks the same element ONCE and returns `0x40 | index-of-NUL` = `0x45`. Put it FIRST in
every batch; a value other than `0x45` means the session is bad, not the experiment.

### R4 — SPLB: our monitor hangs on an exact-fit region (OURS TO FIX)

Not yet reduced to a deterministic trigger — it fires when the host allocator returns a region
exactly matching an existing one, which is layout-dependent. Signature in the UART log:

    SQ: C/mkregion2SPLB:0000E006

Source: `sbi_capstone.c:494-504` — `if (base + len == region_end) { if (base == region_base) {
... while(1); } }`, tagged `CAPSTONE_TAG_SPLB` ("split_out_cap: exact-fit region unsupported",
`:44`) with `CAPSTONE_ERR_SPLIT_EXACT = 0xe006` (`:58`). Any wedge whose last marker is
`SPLB:0000E006` is THIS, not silicon — check for it before attributing a wedge to hardware.

### R5 — the wedge-triage procedure (do this for EVERY wedge)

1. Find the last UART marker for that domain. `SPLB:0000E006` -> R4, ours. `SHA5` with no
   `SHA6` -> region-share (Family A). `SQ: G/enter` with no `SQ: H/return` -> in-domain
   (Family B).
2. Read the debug mux at the wedge — the runner does this automatically (`sw=255`, `224`, `225`,
   `249`-`254`) and clears the trap latch before each domain.
3. **Compare against the HEALTHY values, never against other wedges:** healthy `sw=225 = 0xd5`,
   `sw=224 = 0xff`. `wrev`/`memwait` are SET in the healthy state — they are resting values and
   mean nothing on their own.
4. `sw=255` bit7 = trap_seen, bits[6:0] = mcause. `24` = a real capability exception
   (`UNEXPECTED_OPERAND`); `9` = a stale ECALL from domain entry, i.e. no new trap.

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

### Workaround attempt 2: `SQLITE_STATIC_BUILTINS=1` — CONFIRMED REGRESSION (MEASURED)

The patch was verified to apply before spending board time (`static FuncDef aBuiltinFunc[] = {`
present, `capstoneBuiltinFunc` gone — an initial grep for the OLD name returned 0 and would have
read as "patch failed" if trusted).

    wd71  rc = 0x45   control OK
    sb0   WEDGED      STATIC_BUILTINS at STAGE 0

**Stage 0 is entry-and-immediate-return** — the domain runs no SQLite code at all. So restoring
the array to a compile-time `static` breaks the domain BEFORE any code executes; the only thing
that changed is what `__capstone_cap_init` must materialise. The earlier "breaks even stage 0"
verdict is CONFIRMED, not a single-sample error. This workaround is closed.

### The convergence worth chasing

Both ways of materialising the same ~72-entry `FuncDef` array fail:

* as a LOCAL (straight-line stack construction) -> stage 10 wedges;
* as a `static` (cap-init leaves) -> stage 0 wedges, i.e. even earlier.

Measured cap-init cost:

    build             cap_init size   capability stores (stc)   outcome
    sb0   (static)        16248                1257             WEDGES AT ENTRY
    b10n0 (clamp 0)       10768                1017             wedges at stage 10
    wd71  (probe)         10768                1048             RETURNS

Carve counts are ~equal (181/181/182) — the array is ONE global, so it adds one carve, not 72.
Store count alone does NOT predict the outcome (1017 wedges, 1048 returns), so there is no
simple monotonic threshold across different failure points.

**But `sb0` is the cleanest signal in the campaign:** it wedges at ENTRY, where the only work is
cap-init, with ~20% more capability stores than the largest known-good build. That makes
"cap-init fails somewhere above ~1048 stores" a sharp, cheap hypothesis — and unlike the
in-domain wedges it involves no SQLite logic at all.

**Next test (bisection, entry-time only):** build domains whose cap-init store count is varied
between ~1048 and ~1257 (e.g. by adding N dummy initialised globals holding capability leaves to
a stage-0 domain) and find the threshold. Stage 0 is the ideal vehicle: it returns immediately,
so any wedge is attributable to cap-init and nothing else. If a threshold exists, it is a
concrete number to hand over, and it would also explain the region-share (Family A) failures,
which happen before `domain_main` for the same reason.

### Workaround status

**Not yet available.** `SQLITE_STATIC_BUILTINS=1` was tried earlier and is a REGRESSION (it
breaks even stage 0). Builtin clamping has not yet produced a passing configuration. If the
scoped retest shows limit 0 passing, the next question is the largest limit that still passes,
and whether that leaves enough builtins for CREATE/INSERT/SELECT on integers.
