> **Agent-B private state** — this is Agent-B's copy on branch `capstone-bootstrap-b`
> (clone `/home/alexey/dev/llvm-capstone-b`). Do NOT edit `current-state.md` (Agent-A's
> single-writer base file). Seeded from A's `current-state.md` at Agent-B bring-up (2026-07-08).

# Agent-B delta (2026-07-13, task 015) — CHERI corpus baseline (paper "Task 1")

**The empirical CHERI column of the security table now exists.** Built the full
CHERI-RISC-V **purecap** stack out-of-tree via cheribuild (`~/cheri-ws`, nothing
in `llvm/`): CHERI-LLVM clang 17, CheriBSD purecap FreeBSD 15-CURRENT
(`CHERI_CAPREVOKE`), `qemu-system-riscv64cheri` 7.1.0. Ran each corpus `before.c`
purecap under three revocation configs (each confirmed with
`malloc_revoke_enabled()`):

- **spatial** (revoke off): blocks **3/15** — only the null-derefs (rows 12/13/14,
  synchronous capability-tag fault). All 12 temporal defects run to completion.
- **temporal** (async, the CheriBSD default): blocks **4/15** — the 3 null-derefs
  plus the double-free (row 11, caught by the **allocator's abort**, not a sweep).
  **All 10 use-after-free rows still MISS at the contract point** — the dangling
  capability is revoked only by a later stop-the-world sweep, never synchronously
  at `step`/`reset`/`finalize`/`close`. **This is the paper's thesis, confirmed.**
- **eager** (revoke on every free, non-default + expensive): blocks all 15 shim
  rows — but the **real diesel reuse-not-free case (`row3_reuse`) is blocked by NO
  config** (stale-but-allocated: the cap stays tagged and in-bounds). The clean
  "CHERI-can't" headline.

**Methodology note the paper MUST carry:** upstream SQLite does not run purecap in
a standalone binary here (`SIGBUS`/`BUS_ADRALN` in `sqlite3_open` at
`THREADSAFE=0`; deadlock at `THREADSAFE=1`; both vanilla and CheriBSD-patched
`contrib/sqlite3` 3.46.1). Each shim is therefore compiled **VERBATIM** against a
minimal SQLite-lifecycle harness (`mock-sqlite/`) that reproduces the exact
alloc/free/callback/invalidation events (harness runs clean purecap:
`sanity_mock`=0, real-sqlite sanity=138), built **`-O0`** so the UB dangling
access is emitted. **Additive, measurement only: no `llvm/` change, no
`capstone-qemu` change, no gitlink bump** — only `capstone/tests/cheri-baseline/`.
RESULTS `tests/cheri-baseline/RESULTS.md`; trail
`history/13-07-2026_20-45-00_cheri-corpus-test.md`. No *our-QEMU* rootfs lock.

---

# Agent-B delta (2026-07-13, task 014) — borrow-path cost measured (paper deliverable 2)

**The paper's deferred deliverable 2 now has a citable number.** Vehicle (a),
**confirmed with the human at the gate**: a dynamic-instruction-count proxy on
the QEMU functional model (not cycle-accurate timing — that needs an FPGA/gem5
vehicle not in the tree). Added an emulator readout **`csrdicount`** to
`capstone-qemu` (reads `icount_get_raw()`; under `-icount` one tick == one
retired instruction) and a new `tests/runtime-qemu/borrow-cost-probe/` that, in
one domain, measures three variants of the same borrow-one-word boundary op:

- **raw pointer** = 2 instr/op (O(1));
- **capability borrow** (`mrev`+`delin`+access+`revoke`, inline in-domain) =
  **6 instr/op**, i.e. **+4 over raw**, **payload-independent** (O(1));
- **defensive copy** = `2 + payload/8` instr (O(payload)): 34 at 256 B, 130 at
  1024 B.

So borrow **stays close to raw (3.0×, +4 instr)** and is **well below copy
(5.7×–21.7× cheaper at 256 B–1024 B, gap grows with payload)**. Dynamic counts
match the `-O2` disassembly exactly. **Label any paper number a functional-model
instruction-count proxy.** Full note: `tests/runtime-qemu/borrow-cost-probe/
RESULTS.md`; trail: `history/13-07-2026_16-52-21_borrow-path-perf.md`.
**Additive; `capstone-qemu` gitlink bumped (my lane, submodule pushed first); no
`llvm/` change; A's repros/allocators/monitor untouched.**

---

# Agent-B delta (2026-07-13, task 013) — row-12 provenance citation-mismatch resolved

**The corpus's last known provenance liability is closed.** A's ledger flagged
row 12 (expo-sqlite null-deref) for citing PR #34992 — a statement-*leak* fix, not
the `SQLITE_BUSY`-ignored null-deref the row describes. A thorough upstream search
(expo, cordova-sqlite-storage, better-sqlite3, node-sqlite3, PHP, go-sqlite3, SQLite
CVE DB) found **no** binding that ignores `SQLITE_BUSY`, nulls its C `sqlite3 *`, and
derefs it in C. **Outcome B (documented model) + citation re-anchor:** the citation
now points at the real crash report **expo #34990** (a Java `NullPointerException`
on close-while-busy, the artifact #34992 fixes); row 12 stays honestly **△ modeled**
because #34990's fault is a managed-language NPE, not a C SQLite-pointer null-deref —
same category as rows 7 and 13. The C repro is a faithful behavioral translation of
that crash. The false "All links verified" exposure is removed for row 12; tally
unchanged (4 modeled: 7, 12, 13, 16). **Docs-only, additive:** no `llvm/`, no
`capstone/capstone-qemu` gitlink bump, no code touched (Capstone lit / allocators /
rows 3/7 unaffected by construction), no QEMU boot. Edited row 12's own
`PROVENANCE.md`/`README.md`/`NOTE.md` + the shared `PROVENANCE-LEDGER.md`;
`stage2-mapping.md` unchanged (never claimed row 12 literal). Note
`history/13-07-2026_01-16-03_row12-provenance.md`. On `capstone-bootstrap-b`, pushed.

---

# Agent-B delta (2026-07-10, task 012) — HIERARCHICAL family literal across 3 languages

**The HIERARCHICAL-REVOKE family now has THREE literal real-SQLite repros in three
language bindings, not one.** Task-010 left row 7 (CPython) as the sole literal H;
this task adds row 9 (Ruby, sqlite3-ruby #49) and row 5 (PHP, bug #69971) — the
same intra-domain sub-arena cascade (`revoke_on_free_hier_alloc.h`) catching two
more distinct real CVEs. Generalization breadth, not a new mechanism.

- **row 9 (Ruby)** `run-sqlite-row9.sh` + `sqlite_row9_domain.c`: Database owns a
  child statement; teardown revokes the sub-arena; the child's 2nd `sqlite3_finalize`
  faults on the revoked `sqlite3_stmt`. 5/5.
- **row 5 (PHP)** `run-sqlite-row5.sh` + `sqlite_row5_domain.c`: faithful #69971
  wrong destruction order; Zend wrapper objects live on the surviving global arena
  so only SQLite's connection+statement are revoked; the statement free handler's
  `sqlite3_finalize(stmt->stmt)` faults INSIDE finalize on SQLite's own handle
  (same PC as row 9). 5/5.
- Both: control + fault (**cause 24**, inside the always-`-O0` `sqlite3_finalize`)
  at -O0 and -O2, plus a **real-SQLite sibling-scoping** check (`-D*_SIBLING`, -O2):
  revoke connection A, step connection B → SQLITE_ROW, B survives. New vs task-010,
  which proved scoping only at the primitive level.
- **Fidelity correction:** row 5's first cut faulted on the revoked PHP wrapper
  (cause 25 but the wrong object); moved the Zend wrappers to the surviving arena so
  the fault lands on the `sqlite3_stmt`. See the history note.
- **Additive only:** no `llvm/`, no `capstone/capstone-qemu` (no gitlink bump),
  `revoke_on_free_hier_alloc.h` / `sqlite_row7_domain.c` / A's files untouched.
  Regressions: Capstone lit **39/39**; shared allocator unmodified so row 7 +
  Phase-0 `hier-revoke-probe` unaffected. QEMU rootfs lock claimed + released.
  Note `history/10-07-2026_19-40-00_hier-family-literal.md`; `stage2-mapping.md` H
  literal column lists rows 7/9/5; row 5/9 README "after" sections added. Pushed to
  `origin/capstone-bootstrap-b`.

---

# Agent-B delta (2026-07-10, task 010) — a LITERAL real-SQLite pair per shape

**Every in-scope Stage-2 shape now has a faithful, literal real-SQLite matched
pair, not just a mechanism probe.** Before: only row 3 (task-008 B2) was literal;
16 rows were mechanism probes. Now 5 rows (one per class) are literal real-SQLite
repros — the same program as each `before.c`, real SQLite linked, host = ASan /
Capstone = capability fault on SQLite's own handle, with a no-fault control:

- **L — row 11** (double-finalize): `run-sqlite-row11.sh`. 2nd `sqlite3_finalize`
  derefs the revoked stmt (rof allocation freed by the 1st finalize). 4/4 -O0/-O2,
  cause 24 (stmt spills across the finalize call; control disambiguates).
- **U — row 14** (uninit connection): `run-sqlite-row14.sh`. `db` is a genuine
  UNINIT cap; pre-open read faults cause 26 (self-proving, opt-independent);
  correct control opens with real `sqlite3_open` first. 4/4.
- **H — row 7** (cursor dealloc): `run-sqlite-row7.sh` + `revoke_on_free_hier_alloc.h`
  + Phase-0 `tests/runtime-qemu/hier-revoke-probe` (6/6). Per-connection sub-arenas:
  statement allocations are SPLIT descendants of the connection's MREV'd node;
  close revokes the subtree, `sqlite3_step` on the child faults; a sibling
  connection survives (scoping). **Mechanism answer: YES, intra-domain, no monitor
  change.** 4/4 real-SQLite (cause 24; cause-25 self-proving reached in the primitive).
- **S — row 2** (UDF context UAF): `run-sqlite-row2.sh`. SQLite invokes a registered
  UDF whose freed `pApp` context is revoked; the callback faults. 4/4, cause 24.
  **Residual: the callback-CONTEXT UAF is faithful single-domain; the SEAL PROPER
  (domain-crossing entry) needs a second callback domain — reported, not built.**
- **R — row 19**: skipped (documented) — SQLite reuses the column-text storage in
  place on the next step, no free event to hook; row 3 already covers R.

**Additive-only: no `llvm/`, no `capstone/capstone-qemu` (no gitlink bump), shared
`revoke_on_free_alloc.h` unmodified.** Regressions: Capstone lit 39/39; the 5 pairs
(18 boots) + Phase-0 hier probe (6 boots) green. Note:
`history/10-07-2026_16-30-00_faithful-matched-pairs-per-shape.md`; `stage2-mapping.md`
gains a "Literal real-SQLite repro" column.

---

# Agent-B delta (2026-07-10, task 009)

**The SQLite Stage-2 corpus is complete — 17/17 in-scope rows validated on RTL
(task 009).** Rows 11 (LINEAR, double-free) and 14 (UNINIT, use-before-init) were
the last two deferred rows; both are now mechanism probes on RTL at `-O0`/`-O1`/`-O2`,
**21/21 green**, at `capstone/tests/runtime-qemu/linear-uninit-corpus-probe/`.
Note: `history/10-07-2026_02-30-00_linear-uninit-rows-closed.md` (supersedes the
2026-07-08 deferral note). `stage2-mapping.md`'s validation table is updated.

The 2026-07-08 deferral was correct then and is stale now — both of its blockers were
removed by B-lane work. **Neither escape hatch it scoped out was needed:** no
`share_uninit_region` monitor op, no gated `start.S` change, no firmware rebuild.

1. **Row 14's derivation is one missing `csdelin`.** `cap_rev_tree_revoke()` returns
   `retain_data` = "every node I invalidated was already delinearised", and
   `helper_csrevoke` (op_helper.c:709) turns that into the retained handle's type:
   all-non-linear → `LIN` (cursor at base); **still linear → `UNINIT`** (cursor at
   end, the form `csinit` asserts on). The held-cap probe always `delin`s its arena,
   which is why its revoke yields LIN. Omit the `delin` and `revoke(mrev(arena))`
   hands the domain a real UNINIT capability over its own grant.
2. **Row 14 needed an emulator fix, and it is load-bearing.**
   `_helper_access_with_cap` never consulted the capability **type**: an UNINIT cap
   was denied only incidentally, by bounds (its cursor sits at `end`). **A load at a
   negative offset was in bounds and returned the stale byte** — the disclosure
   UNINIT exists to prevent. Per spec, no load form permits type 3
   (`existing-insn.adoc` "Load Instructions", `mem-access-insn.adoc` `[#load-cap]`:
   types 0/1/5 only, else `Unexpected capability type (26)`); stores *are* permitted
   (that is how an owner initialises memory) and are deliberately untouched.
   `RISCV_EXCP_UNEXP_CAP_TYPE = 0x1a` was already defined and never raised. Now loads
   through UNINIT raise cause **26**, which is self-proving. Verified by removing the
   check and rebuilding: `uninit_negative_offset_fault` **returned `0x14110000`
   instead of faulting**.
3. **Row 11 rests on `csdrop` (task 002), and `csdrop` consumes a HANDLE, not
   memory.** It clears one register's tag; it does not revoke a lineage or free
   bytes. The row is "linearity leaves no second capability, so the second
   `finalize` has nothing to hand back" — not "the allocator refuses a second free".
   `linear_drop_sibling_ok` pins that: after dropping the carved statement, the
   connection it came from still works and the host mmap reads back the post-drop
   write. Both fault probes assert cause 24, so `linear_no_drop_ok` is their control;
   the deref fault and the double-`drop` fault are told apart by their log lines.

Unlike the held-cap probe, **no expected cause moves with `-O`**: the UNINIT handle
keeps its tag and a valid revocation node, and the dropped handle stays untagged
through a spill/reload.

Submodule `capstone-qemu` bumped (the UNINIT-load check + a `DROP requires
capability` diagnostic). No `start.S`, monitor, `capstone-c`, buildroot or LLVM
change.

---

# Agent-B delta (2026-07-09)

**Row3 Option B proven on the real delivery path (task 007).** One domain, a real
monitor-granted `REV_TRANSFERRED` linear region, an intra-domain `MREV` →
`REVOKE`, and the cached alias faults. 8 probes × `-O0`/`-O1`/`-O2`, **24/24
green**, at `capstone/tests/runtime-qemu/intra-domain-mrev-revoke-probe/`. This is
the faithful single-domain artifact the Q1 fork needs: (a) real monitor-delivered
linear cap, (b) intra-domain instruction-level revoke, (c) cached alias faults
with the **asserted** cause, (d) provenance enforced — the arena has no ambient
second path (`held_ambient_miss` proves a forged address traps).

Three things A must read before integrating:

1. **Receive protocol:** no entry-stub glue, no `lcc` index. The delivered
   capability *is* `domain_main`'s first argument on the `REGION_SHARE` entry;
   `my_first_domain/start.S` already surfaces it. The existing runtime probes miss
   it only because they are `.smode` payloads under the `sbi.dom` scaffold (cap
   lands in the scaffold's `regions[]`; S-mode uses ambient `cpmp`) built with
   Buildroot gcc, which has no capability builtins. Use `.dom` + `create_dom(p, NULL)`.
2. **After a domain revokes a granted region, the host must not touch it** —
   `swap_cpmp()` reloads the monitor's stale `regions[]` duplicate untagged and
   `cap_base()`'s `lcc` **aborts QEMU**. Robustness gap, not a mechanism failure;
   fixes are in the monitor (A) and the emulator (B), neither applied.
3. **`MREV` of SQLite's own `memsys5` pointer is not reachable** (step-3 verdict).
   Pointing memsys5 at a granted linear arena works; revoking its allocations does
   not, because `&zPool[k]` is a `cincoffset` that inherits the pool's rev node,
   type and bounds, and `SPLIT` — the only fresh-node derivation — has no inverse.
   Scaffold for the pragmatic path: `probe_linear_arena.h`.

No submodule change, no gitlink bump, no shared file touched.

**Codegen defects C1 + C2 fixed (task 006).** Two separate commits on
`capstone-bootstrap-b`, LLVM/clang tree only — no submodule change, no gitlink
bump.

- **C1 — the SQLite gate is open.** `CC_Capstone_FastCC` now has the `MVT::i128`
  capability case (allocating from `getFastCCArgGPRs(ABI)`), so a non-inlined
  `static` function taking or returning a pointer compiles at `-O1`/`-O2`
  instead of hitting `llvm_unreachable`. Test:
  `llvm/test/CodeGen/Capstone/fastcc-capability-args.ll`.
- **C2 — `MREV`/`DELIN` are no longer modelled as pure.** Both are
  `IntrInaccessibleMemOnly` at the IR level and `hasSideEffects = 1` at the
  MachineInstr level; selection moved to `INTRINSIC_W_CHAIN`; `MREV`'s `$rd` is
  `GPRNoX0`. Test:
  `llvm/test/CodeGen/Capstone/cap-mrev-delin-side-effects.ll`.

Two things worth carrying forward. **The intrinsic attribute alone was not
enough**: `Const`/`IntrNoMem` caused the IR-level CSE of two `MREV`s, but a
*separate* `hasSideEffects = 0` on the MachineInstr caused machine-level DCE of
an unused one. A fix at one layer would have looked correct in `opt -S` and still
lost the instruction in the asm. **And keeping a dead `MREV` alive exposed a
latent emulator gap**: regalloc then picks `x0` for the unused result, and
`helper_csmrev` writes `env->gpr[rd]` with no `rd == 0` guard, so `mrev zero, rs1`
would clobber the hardwired-zero register. Closed compiler-side with `GPRNoX0`; a
hand-written `.insn` could still reach it, so `op_helper.c` deserves the guard.

C3 (LINEAR cap consumed when passed by value) was **not** changed: it is correct
linear semantics, and the Option B recipe already handles it with an explicit
`delin`.

Regression with the rebuilt compiler: authority suite **32/32**,
capstone-mrev-codegen probes **9/9**, Capstone lit **38/38**.

**Intra-domain MREV/REVOKE codegen spike (task 005) — the Option B unknown is
answered.** Compiled domain C at `-O2` **does** route a deref through the held
capability, so a later `REVOKE` faults it: verified on a register-held alias
(cause 25), an alias passed across the C ABI into a non-inlined callee (cause
25), and an alias spilled to memory and reloaded (cause 24). The `-O2` asm shows
the post-revoke load still based on the gencap register — no ambient
re-materialisation. Also: **`MREV` retains its linear source** (`helper_csmrev`
never nulls `rs1`), so one linear grant mints many nested revocation handles; and
**a linear sub-cap is `MREV`-able via `SPLIT`** (fresh rev node ⇒ siblings revoke
independently), while `SHRINK`/`SHRINKTO` copy `rev_node_id` and so cannot carve
independently-revocable sub-buffers.

Nine probes landed at `capstone/capstone-qemu/tests/capstone-mrev-codegen/`
(firmware-free `csdebuggencap` mint; driver asserts the **exact** fault cause,
not just "a fault"). GREEN 9/9. Submodule `e0cd45de`→`fd4bc0c0` (pushed);
superproject gitlink bumped on `capstone-bootstrap-b`.

Three codegen defects found, all in B's lane, **none blocking the Option B
mechanism**: **C1** `CC_Capstone_FastCC` has no `MVT::i128` case, so any `-O1+`
domain TU with a non-inlined `static` function taking a pointer ICEs clang
(`CapstoneISelLowering.cpp:23820`) — this *does* block the SQLite integration;
~10-line fix, deliberately not applied (shared LLVM tree A also builds).
**C2** `cap_mrev`/`cap_delin` are marked `Const`/`IntrNoMem` but mutate the
revocation tree, so at `-O2` an unused `MREV` is DCE'd and two `MREV`s of the
same value CSE into one. **C3** only `NONLIN` is copyable, so `movc`/`cincoffset`
null a LINEAR source register: passing a LINEAR cap by value consumes it,
silently and with no diagnostic.

**Option B recipe (for A):** `arena = <linear grant>; R = mrev(arena);
alias = delin(arena); …use alias freely…; revoke(R)`. The `delin` is required by
C3 and is safe — it only clears the node's `linear` flag, so the alias stays
revocable and `REVOKE` hands back a reusable `LIN` handle. The arena must be
reachable **only** through the tracked capability; an ambient/`gp`-reachable
buffer has a second, un-revocable path (`mrev_ambient_miss` returns instead of
faulting). Note:
`history/09-07-2026_20-42-10_intra-domain-mrev-codegen-spike.md`.

## Previous (task 004)

**Revoke probes landed in-submodule + csdrop commit durability (task 004).** The
four task-003 revoke probes (`revoke_mem_alias`/`reg_alias`/`unrelated_ok`/
`mem_control` + `csrevoke_probe.h`) now live in the capstone-qemu submodule's own
test tree (`tests/capstone-revoke-probes/`) with a `run-revoke-probes.sh` driver
(reuses the sibling `capstone/tests/runtime-qemu` harness) and a README recording
the mechanism, the reproduction table, and the **provenance constraint** (a region
reached via an SBI `REGION_QUERY` mapping is not a tracked descendant of the
revocable cap, so it must be delivered through the tracked linear cap). Confirmed
on binary `2e6a67d1`: mem-alias → cause-24 (reload untags), reg-alias → cause-25
(live rev-node invalid), unrelated OK 0x22130033, control OK 0x2214005E. Submodule
probe commit `e0cd45de` (on top of `2e6a67d1`); superproject gitlink bumped
`2e6a67d1`→`e0cd45de`. Driver green 4/4. **Durability CONFIRMED:** both are on the
shared remotes (superproject `d2e9dd43`, `capstone-qemu` `e0cd45de`), verified by a
fresh clone outside this working clone; nothing outstanding to push. Agent-B now
has its own PAT and pushes its own branches (`capstone-bootstrap-b` only).

---

**`csrevoke` memory-alias sweep VALIDATED (task 003, outcome a).** Confirmed the
load-bearing BORROW-REVOKE property: `csrevoke` invalidates **memory-resident**
copies of a revoked capability, not just the register operand. Mechanism is lazy
revocation on a shared rev-tree node; `rev_node_id` survives `cap_compress`/
`cap_uncompress` (bits 33-63) and a reloaded revoked cap comes back untagged
(`helper_reg_set_cap_compressed`), so its deref cause-24 faults. Proven
firmware-free by hand-minting a LINEAR cap via `csdebuggencap` (`.insn`) +
`mrev`/`revoke` builtins; 4 QEMU probes (mem-alias FAULT, reg-alias FAULT,
unrelated OK 0x22130033, mem-control OK 0x2214005E). **No code change** (sweep
already works). The row 3/13/18/19 revoke vehicle is proven at the emulator layer;
the only remaining gate is A's `start.S` linear authority, which must route the
region **through the tracked linear cap** (not an SBI-query mapping — the earlier
R-probe "NO-TRAP-GAP" was that orthogonal provenance issue). Note:
`history/09-07-2026_15-33-23_csrevoke-memory-alias-sweep-validated.md`.

---

**`csdrop` (DROP) implemented in `capstone-qemu`** — the LINEAR / Stage-2 row-11
QEMU-lane unblock (task `agentB-002`). The emulator previously had no `csdrop`, so
`__builtin_capstone_cap_drop`'s `drop` mnemonic decoded as an illegal instruction;
now it invalidates a capability (clears rs1's tag), so a later use faults cleanly
(cause 24, "Cap mem access requires capability") rather than trapping as illegal.
Spec-faithful and type-agnostic (DROP has no LIN-only restriction). Submodule bump
`cf541a1f`→`2e6a67d1`; superproject gitlink bumped on `capstone-bootstrap-b`.
Validated under QEMU (control ok + fault cause-24), no regressions. Full note:
`history/09-07-2026_13-28-31_csdrop-implemented-row11-qemu-unblock.md`. Row-11 full
domain demo still needs A's gated linear-authority `start.S`. (Prior: C1 subobject
v1 arrays-only was merged to canonical by A, ff to `c4758de`.)

---

# Current Capstone state

Minimal snapshot. Read first in every session.

## SQLite in-memory bring-up

SQLite 3.53.3 compiles, links, **and runs end to end** as a
`capstone64-unknown-elf` pure-capability domain using memsys5 over the static
arena and the runtime-initialized SQLite VFS skeleton. `run-sqlite-memory.sh`
executes `CREATE TABLE` / `INSERT` / `SELECT` and the domain returns correct rows
(`row name=alpha value=11 / beta=22 / gamma=33`, `__CAPSTONE_SQLITE_MEMORY_PASSED__`).
The pinned fetch/build/run workflow is in `capstone/benchmarks/sqlite/README.md`.

**Bring-up is complete — all 8 gaps resolved:**
- Gaps 1–2 (compiler): `CapstoneCapGlobalInit` recurses nested global aggregates
  (#71); clang memcpy-from-private-template of cap aggregates handled (#72).
- Gaps 3–4 (QEMU): untagged `ldc`/`stc` made bit-preserving over the full 128-bit
  word, enabling a tag-preserving `memcpy` (#73/#74).
- Gap 5 (compiler ISel): `cscincoffset` int+ptr operand order (#79).
- Gap 6 (SQLite alignment): 16-align `sqlite3NestedParse`'s `saveBuf` so the
  tag-preserving `memcpy` fast path carries Parse-tail caps (#80).
- Gap 7 (compiler): materialize interior-pointer capability globals
  (`&global[N]`) — `sqlite3aLTb/aEQb/aGTb` (#81).
- Gap 8 (SQLite alignment): 16-align the `BtCursor` embedded by `allocateCursor`
  (#82).

Full per-gap detail in `history/` (dated notes) and
`design/sqlite-gap6-memcpy-tag-preservation-proposal.md`. Follow-ups: the SQLite
8-byte-alignment class (gaps 6/8) may surface more instances under wider workloads.

**In-domain cap-fault delivery — abort retired (2026-07-03).** QEMU no longer
aborts on an in-domain capability fault: `riscv_cpu_do_interrupt`'s
`assert(env->priv < PRV_C)` is replaced (for `env->priv == PRV_C`) by a clean halt
— a structured `[CAPSTONE] domain halted by capability fault: cause=…` line then
`fflush`+`exit(0)`. This preserves the domain's serial output (`abort()` didn't
flush stdio — the gaps 8/9 "no serial output" cause) and turns a SIGABRT into a
named halt. The monitor host-trap path (`priv < PRV_C`) is unchanged. Validated:
full authority suite all-PASS, SQLite base+extended PASS, no abort in logs. Step A
proved the `ctvec` horizontal-trap path can't deliver this (a domain installs no
`ctvec`). **Return-to-host** delivery (domain terminates, host continues) is the
remaining, monitor-side step — see
`design/domain-fault-delivery-proposal.md` + `history/03-07-2026_00-00-03_*`.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, and
  `run-nullblk-split-rmmod.sh`
- `run-hostcall-all.sh`, `run-nullblk-all.sh`, and `run-all-beebs.sh` provide
  aggregate gates for reproducible full reruns; keep individual wrappers as the
  diagnostic entry points. The HostCall, `null_blk`, and full BEEBS aggregates
  have passed end to end; BEEBS has also passed with `RUN_ALL_BEEBS_JOBS=4`.
  `run-all-beebs.sh` is serial by default
  (`RUN_ALL_BEEBS_JOBS=1`) and has opt-in isolated parallelism via
  `RUN_ALL_BEEBS_JOBS=N`. It keeps child output in per-benchmark logs by default
  and prints compact pass/fail lines; set `RUN_ALL_BEEBS_VERBOSE=1` for streamed
  child output. It retries structured QEMU infra flakes before benchmark
  execution twice by default (`RUN_ALL_BEEBS_BOOT_RETRIES=0` disables this) and
  caps aggregate boot-to-login waits at 90 seconds by default
  (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`), but does not retry benchmark marker failures.
- QEMU runtime smoke tests use snapshot mode, so repeated runs do not mutate `rootfs.ext2`
- Buildroot getty is pinned to `ttyS0`, avoiding intermittent boot-to-login hangs through `/dev/console`
- QEMU runtime smoke tests force `-smp 1`, avoiding intermittent boot stalls under the current OpenSBI/QEMU setup
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` - third BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` - fourth BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` - fifth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-prime.sh` - sixth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` - seventh BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` - eighth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` - ninth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cover.sh` - tenth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-duff.sh` - eleventh BEEBS benchmark runs
  end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` - twelfth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` - thirteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` - fourteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-strstr.sh` - fifteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ndes.sh` - sixteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraybinsearch.sh` - seventeenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-queue.sh` - eighteenth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listinsertsort.sh` - nineteenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listsort.sh` - twentieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-expint.sh` - twenty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-compress.sh` - twenty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-md5.sh` - twenty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh` - twenty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-matmult.sh` - twenty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc32.sh` - twenty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-mergesort.sh` - twenty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-stringsearch1.sh` - twenty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bs.sh` - twenty-ninth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fir.sh` - thirtieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-lcdnum.sh` - thirty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ns.sh` - thirty-second BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ud.sh` - thirty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nsichneu.sh` - thirty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraysort.sh` - thirty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayheapsort.sh` - thirty-sixth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayquicksort.sh` - thirty-seventh
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-dllist.sh` - thirty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-hashtable.sh` - thirty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-aes.sh` - fortieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-picojpeg.sh` - forty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-sha256.sh` - forty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-huffbench.sh` - forty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-rijndael.sh` - forty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc.sh` - forty-fifth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-statemate.sh` - forty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-arcfour.sh` - forty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-des.sh` - forty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-mont64.sh` - forty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-dijkstra.sh` - fiftieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-stack.sh` - fifty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-vector.sh` - fifty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-edn.sh` - fifty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-string.sh` - fifty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-qrduino.sh` - fifty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh` - fifty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-miniz.sh` - fifty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-slre.sh` - fifty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-wikisort.sh` - fifty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh` - sixtieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-compress.sh` - sixty-first BEEBS
  benchmark runs end to end and validates its adapted LZW-state checksum marker
- `capstone/benchmarks/beebs/run-beebs-cubic.sh` - sixty-second BEEBS
  benchmark runs end to end with the soft-float/libm runtime and root oracle
- `capstone/benchmarks/beebs/run-beebs-sqrt.sh` - sixty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ludcmp.sh` - sixty-fourth BEEBS
  benchmark runs end to end with the local const-array source workaround
- `capstone/benchmarks/beebs/run-beebs-minver.sh` - sixty-fifth BEEBS
  benchmark runs end to end and validates its adapted matrix checksum marker
- `capstone/benchmarks/beebs/run-beebs-frac.sh` - sixty-sixth BEEBS
  benchmark runs end to end with shared soft-float/libm support
- `capstone/benchmarks/beebs/run-beebs-st.sh` - sixty-seventh BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-nbody.sh` - sixty-eighth BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-qsort.sh` - sixty-ninth BEEBS
  benchmark runs end to end with a widened 1-indexed array and sorted-region hash
- `capstone/benchmarks/beebs/run-beebs-qurt.sh` - seventieth BEEBS benchmark
  runs end to end and validates all three quadratic root cases
- `capstone/benchmarks/beebs/run-beebs-select.sh` - seventy-first BEEBS
  benchmark runs end to end with a widened 1-indexed array and return-value oracle
- `capstone/benchmarks/beebs/run-beebs-newlib-sqrt.sh` - seventy-second BEEBS
  benchmark; self-contained `__ieee754_sqrtf`, upstream exact verifier with
  `exp[]` moved to `static const` (Bug #9), soft-float builtins only
- `capstone/benchmarks/beebs/run-beebs-newlib-exp.sh` - seventy-third BEEBS
  benchmark; self-contained `__ieee754_expf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-log.sh` - seventy-fourth BEEBS
  benchmark; self-contained `__ieee754_logf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-mod.sh` - seventy-fifth BEEBS
  benchmark; self-contained `__ieee754_fmodf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-stb_perlin.sh` - seventy-sixth BEEBS
  benchmark; 3-D Perlin noise, self-contained oracle (`benchmark()` compares a
  10x10 plane against a `static const` table and returns 0 on full match);
  only external dep is `floor`, added to the shared soft-float libm
- `capstone/benchmarks/beebs/run-beebs-matmult-float.sh` - seventy-seventh BEEBS
  benchmark; `matmult` source built `-DMATMULT_FLOAT` (float[10][10]), soft-float
  builtins only, FNV-1a checksum of the global `ResultArray` vs a host reference
  (`--gc-sections` drops the dead `values_match`/`frexpf`/`fabsf`)
- `capstone/benchmarks/beebs/run-beebs-whetstone.sh` - seventy-eighth BEEBS
  benchmark; classic Whetstone over the shared libm (added `atan`); built
  `-DPRINTOUT` with a capturing `POUT` that FNV-folds every module's outputs,
  compared exactly to a same-libm host reference

Most BEEBS correctness-marker wrappers now share `beebs_simple_domain.c` and
`beebs_simple_host.c`. Keep separate per-benchmark domain/host files only when
the marker ABI or host behavior is genuinely different; currently the older
`fac`, `fibcall`, and `insertsort` wrappers keep custom markers.

Most Capstone-specific benchmark source adaptations live in explicit `.c` files
under `capstone/benchmarks/beebs/adapted/`; shell scripts generally orchestrate
fetch/build/link/run rather than embedding C source. Full-replacement adapted
files (bubblesort, prime, cnt, duff, janne_complex, tarai, levenshtein,
recursion) are compiled directly. Prefix/tail files (crc32) and tail-append
files (strstr, insertsort, jfdctint, fdct, aha-compress, nettle-md5,
nettle-cast128, nettle-arcfour, nettle-des) are concatenated with the stripped
upstream source at build time. `huffbench` uses checked-in adapted C snippets
for its freestanding prefix and RNG replacement. `aha-mont64` uses a checked-in
rewrite helper for constant hoisting. `ndes` uses a checked-in rewrite helper
for pointer-based aggregate passing and explicit table delinearization.
`ctl-string`, `qrduino`, `miniz`, `slre`, and `trio-sscanf` are generated as
scratch sources under `$CAPSTONE_TMP_ROOT/beebs-build` because their adaptations
are local include/stub/allocation/verifier rewrites rather than reusable
replacement translation units.  `slre` additionally uses a checked-in tail file
(`adapted/beebs_slre_capstone_tail.c`) to avoid the `char *regexes[]` global
pointer array that would require caprelocs.  `wikisort` uses a checked-in tail
file to keep the upstream prefix while replacing the Range/sort/test tail.
`trio-sscanf` strips hosted includes, builds with `TRIO_SSCANF`,
`TRIO_EMBED_STRING`, float/file/dynamic-string features disabled, a minimal set
of embedded `triostr` helpers, and checked-in freestanding libc stubs.
`compress`, `cubic`, `minver`, `qsort`, `qurt`, and `select` use adapted
oracle tails because the upstream verifiers return `-1`. FP benchmarks use
compiler-rt soft-float builtins and, where needed, the shared
`adapted/beebs_softfloat_libm.c` domain libm.

`build-beebs-simple-capstone-common.sh` now supports `BEEBS_EXTRA_DEFINES`
(array of `-D` defines, e.g. `BEEBS_EXTRA_DEFINES=(QUICK_SORT)`),
`BEEBS_STRIP_FROM_REGEX` plus `BEEBS_ADAPTED_TAIL_SRC` for single-source
tail-replacement adaptations, and includes `-fno-jump-tables` unconditionally
(jump tables use raw integer addresses which fault on Capstone since loads
require capabilities).

## Resolved blocker

The 2026-06-09/10 split `null_blk` unload blocker is resolved. The hang was
diagnosed as lost timer progress after split-domain activity: QEMU traces showed
that the final timer H-interrupt was taken while `mie.MTIP` was disabled, after
which OpenSBI did not reprogram the timer and RCU/percpu-ref progress stopped.

The fix is in `capstone/capstone-qemu`:

- Capstone H-interrupt selection in `riscv_cpu_local_irq_pending()` now considers
  only interrupts enabled by `env->mie`.
- `rmw_mie64()` calls `riscv_cpu_check_interrupts()` after `mie` changes so a
  pending H-interrupt becomes deliverable when software reenables it.

The split null_blk package also keeps the safer fixes found during investigation:
metadata is borrowed per domain call instead of permanently shared, and
`null_validate_conf()` copies back only validated scalar configuration fields.

All temporary Linux/OpenSBI/QEMU trace and printk diagnostics were removed before
the verified run.

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Three remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

The `va_list` capability-tag-loss backend bug is fixed and validated: `va_start`/
`va_arg`/`va_copy` now lower with capability ops (`stc`/`ldc`, 16-byte `cincoffset`
stride). The CoreMark `ee_printf_asm.S` trampoline is removed — `ee_printf` uses a
standard C `va_list` and CoreMark still validates. This unblocks the `va_list`
prerequisite for `trio`.

The `sub i128` pointer-decrement backend blocker is fixed and validated:
`ptr - integer` and `ptr + (-offset)` now lower through `cincoffset` with a
negated XLEN offset.

The `sub i128` pointer-difference backend blocker is also fixed and validated:
`ptr - ptr` now lowers by extracting both capability cursors with `lcc ..., 2`,
subtracting the XLEN cursor values, and sign-extending the integer result back
through the `i128` carrier when needed. `ctl-string` is the proof benchmark.

Stack-passed capability arguments are fixed: a function with >8 args whose extra
args are pointers had its stack-slot address computed with an integer `ISD::ADD`
(→ `addi`, tag-stripping), delivering the callee an untagged capability.
`CapstoneTargetLowering::LowerCall` now uses a capability `CIncOffset` for the
slot address (test `stack-cap-arg.ll`; repro `tests/runtime-qemu/stack-cap-arg-repro/`).
This unblocked RV8 `norx` and is the same class as the `va_list` fix.

The i128 non-vector-shift assertion (Bug #3) is fixed (`lowerScalarI128Shift`
general constant-shift fallback). **Capability globals are now auto-tagged**: the
`CapstoneCapGlobalInit` ModulePass synthesizes a per-module `__capstone_cap_init`
(called from `my_first_domain/start.S` before `domain_main`) that materializes
initialized capability globals in place at runtime — a tag cannot live in the
static image. Validated via `static-cap-typed-load-repro` + lit
`static-cap-global-init.ll`. Design:
`design/capability-globals-init-decision.md`.

## Capability granularity & provenance (C1/C2 — paper track)

After the three benchmark suites completed, work pivoted to the paper's security
contributions. **An external audit (2026-06-29,
`history/29-06-2026_15-08-22_granularity-provenance-audit.md`) reviewed this whole
direction; its findings are folded in below — read it before paper-facing work.**
Current state on `capstone-bootstrap`:

- **Bounds model** (`design/capability-bounds-model.md`): the narrowing op is
  **`SHRINK`** (`int_capstone_cap_shrink`); `SPLIT`/`SHRINKTO` exist in the ISA
  but are unwired. **Audit correction:** the `<4 KiB exact / grain-above`
  representability rule is **spec-derived, NOT measured** — this QEMU keeps exact
  fat bounds in a side table (`cm_map`) and restores them on load, so observable
  `SHRINK` is **exact at all sizes**. Un-narrowed bounds are segment-granular
  (single `PT_LOAD` ≈ whole image).

- **C1 object-granularity narrowing — INITIAL SLICES (not a spatial-safety
  theorem; broad `gp`/`sp` roots remain, permissions stay RWX):**
  - **Globals** — `selectLGA` (`CapstoneISelDAGToDAG.cpp`) narrows each sized data
    global to `[&g, &g+sizeof(g))`. Flag `-capstone-shrink-globals` (**default on**);
    functions / unsized externs not narrowed.
  - **Heap** — NOT a libc policy: only **two benchmark-local allocators**
    (`rv8_malloc.c`, dtoa `malloc_beebs`) `cap_shrink` returns; trio left
    un-narrowed (its `realloc` over-reads); CoreMark uses stack storage. Do not
    call this "heap default-on."
  - **Stack** — fixed stack objects narrowed to `[&obj, &obj+size)` via the
    shared `narrowToFrameObjectBounds` helper, now covering **both** the
    bare-`FrameIndex` address **and** interior pointers / load-store bases
    (`materializeFrameIndexAddrBase`), flag `-capstone-shrink-stack`
    (**still default off** pending the empirical default-on matrix). Not yet:
    varargs save-area, dynamic `alloca` (variable-size + spill slots excluded by
    design). Object- not subobject-granularity.
  - Validation is **functional only**: **CoreMark ✓, RV8 7/7 ✓, BEEBS 82/82 ✓**
    with global+heap on; stack-on smoke = CoreMark + 9 stack-heavy BEEBS ✓. Found
    a **real OOB bug**: rijndael wrote 8 bytes through a `char r[4]` (patched).
    **Code-size overhead measured across all 90 domains (CoreMark + 7 RV8 + 82
    BEEBS, 2026-07-01):** globals narrowing costs a near-constant **~15.6 bytes
    per narrowed global**; as % text, **median 1.83%, mean 4.17%, range 0%
    (no sized globals) – 46% (`statemate`, generated WCET tables)**; no
    correctness regression — matrix + full table in
    `design/c1-coverage-matrix-and-overhead.md`. **Runtime/cycle overhead still
    NOT measured** (functional QEMU, no cycle-accurate path) — don't claim it.
  - **Negative pointer difference fixed:** exact signed element scaling now
    restores `srai` after narrowing the i128 pointer-difference carrier to XLEN;
    genuine logical shifts remain `srli`. Positive and negative runtime probes
    pass, including `low - high == -7`.

- **Provenance/authority evidence suite** (`capstone/tests/capstone-authority/`,
  `run-authority-suite.sh`): 20 domains pinning runtime behavior (source + asm +
  QEMU trap/no-trap vs an oracle). forge/ptr→int→ptr **tag-fault**; global/heap/
  stack edge/index `_oob` **bounds-fault**; positive/negative pointer differences
  and last-valid-byte controls pass. A struct-field over-read is
  **no-trap-today**, confirming the subobject-bounds gap. The additive opt matrix
  passes all 12 eligible domains at `-O1/-O2/-O3`; 8 assembly-verified O0-only
  probes are explicitly skipped. Runtime fact:
  a domain-mode capability fault currently **aborts the QEMU model** (a
  `riscv_cpu_do_interrupt` assertion) after emitting the diagnostic.

- **Regression tests:** lit `cap-shrink-globals.ll`, `cap-shrink-stack.ll`
  (on/off A/B), `ptr-diff-signed.ll`, and updated
  `static-cap-global-init.ll`. Full Capstone lit suite green (32 tests).

- **C2 (provenance verifier) — REDESIGNED (v2, 2026-07-01), awaiting reviewer
  sign-off before implementing.** The audit found v1 (`UNKNOWN`-accepting,
  opcode-only) was a hygiene checker, not a proof. The redesign in
  `design/c2-provenance-verifier-proposal.md` §"Design (v2)" folds in all three
  fixes: no permissive `UNKNOWN` (`ROOT`/`CAP`/`INT`/`TAINTED` lattice, TAINTED-as-
  authority flagged), IR→MIR intent + calling-convention arg/return seeding,
  precise per-opcode transfer functions (LDC propagates memory tag; tied-operand
  ops inherit+validate; integer-as-base is a fault not a forge), two separated
  properties (P1 non-forging / P2 preservation), and a small hand-proved formal
  model with the corpus as validation. v1 retained in the doc for history. Do NOT
  implement until the reviewer signs off on v2.

- **Audit's strategic reframing (for the reviewer):** object bounds re-derive
  CHERI; Capstone's novelty is linearity/revocation/`SPLIT`/**root-elimination**.
  Proposed stronger frame: **provenance + attenuation + root-elimination** (trusted
  `SPLIT` removes the ambient broad root from application code). A
  research-direction decision, not yet acted on.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
