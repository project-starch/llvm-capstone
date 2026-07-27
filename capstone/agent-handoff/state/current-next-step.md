# Current recommended next step

## 2026-07-27 — deadline triage: the paper's eval section

Deadline is **end-July 2026**. Weigh everything against *"does this change a number
or a claim in the paper?"* The perf table rests on **3 measured rungs**; shipping 3
rungs with correct caveats beats 5 rungs with an unexplained divergence.

**The one open gate: the gp-captable silicon divergence** (`state/current-state.md`
"Latest (2026-07-27)" for the full status and the list of dead hypotheses).

Highest-value work now, in order:

1. **Compiler-side, no board needed — look at `core_init_matrix` at −O0.** #66 put
   `coremark_matrix`'s fault inside that one ~40-line function. Both surviving
   candidates run through the **gp-delivered block capability** (the seeding loop
   writes `A[]` and `B[]` through it), and an extra capability store in a loop is the
   *already-confirmed* miscompute trigger from the 2026-07-26 bisect. Check whether
   our codegen keeps a **live capability across the loop** where the RTL's shrink-off
   / store-hazard workaround assumes otherwise.
2. **The genuinely open deliverable is unchanged: is our emitted code ISA-legal and
   QEMU-correct?** If yes, this is an **RTL divergence** to hand to the board owner
   with a minimal repro — a *paper-acceptable* outcome (documented hardware
   limitation), not a failure. Keep the board-owner message short and human.
3. **Do not assume the two hanging rungs share a mechanism.** `matmult_int` has no
   data-dependent bound; `coremark_matrix` is built around one. They may be two bugs.
   `matmult_int`'s remaining lever is a **phase bisect** needing no mechanism guess:
   return after the init loops, then after the `mm_cell` loops, then after the FNV
   fold (2–3 boots).
4. **Board probe #67** (split `core_init_matrix` — return `N` *before* the seeding
   loop, one boot, separates the two candidates) is **B's**. The board is one shared
   physical resource, **serialized across lanes** — confirm a window before any
   hardware run.
5. **Lift the 4 KiB code window** (task #62) — one hardcoded number in
   `link-gpfree.ld`, QEMU-validated at 16 KiB and 32 KiB. It is what full CoreMark +
   Dhrystone need, and the only realistic way to close the **missing pointer-chasing
   axis** (capabilities are 16 B vs 8, so the current set likely *understates*
   overhead).

**Fallback, already written:** if the divergence does not resolve in time, report the
3 rungs + the `ref/fpga-silicon-measurements-for-paper.md` §5 caveat list, and state
the 2 blocked rungs as a documented hardware limitation. **Do not let this blocker
hold the whole eval section hostage.**

Keystone unblocks tracked elsewhere, both from the board owner's two answers:
**monitor rebuild** with `caplifive-system`'s pinned `capstone-c` (`bugfix`@`508342a`,
not our tree's `master`@`8cda52c`) — `plans/monitor-regen-audit-task-B.md`; and
**large-RO delivery via host userspace**, not the monitor —
`plans/sqlite-on-silicon-scoping.md`.

---

## Superseded (2026-07-22) — gp-free domain bring-up, `C_GEN_CAP` blocker

**RESOLVED 2026-07-23 — kept for provenance.** The blocker below (the monitor's `gp`
mint used `C_GEN_CAP`, a QEMU-only DEBUG instruction not implemented on the RTL) was
root-caused and **fixed on silicon**: a capstone-c-style **data-authority `gp` +
cap-table** probe (`start-gpfree-captable.S` — derive `gp` from sp/cscratch in-glue,
globals in DATA memory, access via `ldc gp[0]`; no code split, no `gp` memory
round-trip) **passes on captype-fixed CVA6**, retval `554745961` — the first
globals-using domain to run on the board. The monitor was left **unchanged**. See
memory `history/22-07-2026_18-05-00_...` entry in `MEMORY.md` for the full trail.

Original note follows.

DONE on QEMU (functional, silicon-faithful): a real globals-using app runs
correctly in a pure-cap domain **gp-free** with the fabrication OFF and the
**monitor** delivering `gp` via cscratch. Compiler committed (`88054a14`); monitor
+ QEMU edits are local submodule experiments. Board owner ratified the approach.

**Silicon smoke ATTEMPTED 2026-07-22 — firmware/boot chain FIXED, domain run BLOCKED
by a definitive root cause. FULL KB:
`history/22-07-2026_18-05-00_gp-free-silicon-smoke-firmware-fixed-createdomain-hangs.md`.**
- Working now: board boots Linux to a shell with our gp-delivery monitor
  (CAPLIFIVE-ARIANE fw, embedded FDT+kernel), `/dev/capstone` present, controller +
  domain transfer+verify. Firmware-build recipe in memory
  `project_fpga_fw_payload_build_recipe`.
- BLOCKER (since resolved, see above): the monitor's `gp` mint uses **`C_GEN_CAP`
  (custom-2 funct7 0x40) — a QEMU-only DEBUG instr (`helper_csdebuggencap`), NOT
  implemented on the RTL** (decoder `default: ;`). It fabricates a cap from
  (base,end), which HW capability monotonicity forbids. On silicon it no-ops →
  garbage cap → `stc` fault → M-mode hangs in `capstone_error`=`while(1)`. Compiler
  side was fine; the domain never ran.

## Active track (2026-07-15): the paper's PERFORMANCE story — mostly DONE

The active track since the 2026-07-13 reframe is **performance**, not C1/C2.
The C1/C2 section below is paused reference material.

- **DONE:** the full CHERI-vs-Capstone temporal-safety perf comparison
  (QEMU-to-QEMU, microbench + BST tree, both sides), the paper perf tables
  (`evaluation.tex` §`sec:eval-perf-compare`), the `-O2` capability-select ICE
  fix, and the `-O2` tree re-measurement (revoke-at-free **+5 instr/op O(1)**,
  matching the microbench). See `state/current-state.md` "Latest (2026-07-15)".
- **STANDING NEXT STEP — Capstone RTL cycle-accurate number.** The QEMU result is
  a functional-model instruction-count proxy; the goal is "on real hardware we
  are obviously faster" layered on top. RTL borrow-cost port is staged in
  `tests/rtl-smoke/`; the temporal-overhead run is the follow-on. **Human-in-the-
  loop** (browser GUI, agent can't drive the board) and **POSTPONED** pending
  the board owner's answer on whether it can be automated.
- **Infra:** `capstone/tests/run-nightly.sh` is the new one-shot build+test+report
  driver (serial QEMU suites, report to `/tmp/capstone/`).

## Paused track: capability granularity + provenance (the paper, C1/C2)

The three benchmark suites are **complete** (CoreMark ✓, BEEBS 82/82 ✓, RV8 7/7 ✓;
only C++ `bigint` deferred). Work has pivoted to the first paper's security
contributions. Status + where things are: `state/current-state.md` (the C1/C2
section) and the design docs below.

- **C1 granularity — INITIAL SLICES (functionally validated, not measured).**
  `SHRINK` for **globals** (`-capstone-shrink-globals`, default on); a **real
  reusing heap allocator** (vendored **umm_malloc** behind the `cap_heap.c`
  narrow/re-widen shim; RV8 7/7 incl. free/realloc-heavy; dtoa still a bump);
  **stack** gated spike (default off; fixed objects incl. interior/load-store
  bases as of 2026-07-01). Tests: `capstone/tests/capstone-authority/`
  (28/28 at canonical `-O0`, incl. `heap_free_reuse` + `heap_coalesce` on the real
  umm allocator; eligible probes pass at `-O1/-O2/-O3`) + lit
  `cap-shrink-{globals,stack}.ll`. The suite now measures the residual subobject
  gap directly; broad `gp`/`sp` roots and RWX perms remain. **#78 phase-1
  (spatial) done + shipped. Phase-2 (revoke-on-free temporal safety) is
  SUSPENDED (2026-07-06)** — a de-risking spike found three blockers (per-alloc
  revocation vs coalescing → needs a slab allocator; `mrev` needs LINEAR; the
  arena's linear authority is not `gp`, a ~592 B small-data pointer). It is a
  real multi-step project, not a quick add-on; resume plan in
  `design/heap-temporal-safety-revoke-on-free-proposal.md` (status: SUSPENDED),
  evidence in `history/06-07-2026_18-00-00_...`.
- **C2 provenance verifier — PROPOSED, revise before implementing.** The audit
  found it is a hygiene checker, not a proof (see the doc's audit-response banner).
  **Do not implement verbatim**; needs typed-MIR redesign + reviewer sign-off.

**An external audit (2026-06-29,
`history/29-06-2026_15-08-22_granularity-provenance-audit.md`) reviewed this whole
direction — read it first.** Its recommended order (adopted here):

1. Negative pointer difference (`srli` on signed exact scaling) — **fixed**;
   signed/unsigned lit coverage + positive/negative runtime probes pass.
2. Bounds-model doc corrected (QEMU keeps exact fat bounds; representability not
   measured) — **done**. **Recommended next:** decide the intended real 128-bit
   `SHRINK` semantics.
3. Rewrite the C1 claim as a **coverage matrix**; **measure overhead** —
   **DONE (2026-07-01):** `design/c1-coverage-matrix-and-overhead.md` (coverage
   matrix + full code-size table over all 90 domains: ~15.6 B/narrowed-global,
   median 1.83% / mean 4.17% / range 0–46% text, no correctness cost). Residual:
   runtime/cycle overhead unmeasured (functional QEMU) — needs a cycle-accurate
   or instrumented-instruction path.
4. Decide the **`uintptr_t` / address-only ABI** (currently an accidental 64-bit
   authority-losing middle ground).
5. **Revise** the C2 proposal to a strict typed-MIR invariant + small formal model
   — **revision DONE (2026-07-01):** `design/c2-provenance-verifier-proposal.md`
   §"Design (v2)". **Then implement** — gated on reviewer sign-off on v2.
6. Stack coverage toward default-on (task #77). **Increment 1 DONE (2026-07-01):**
   interior pointers + load/store bases through fixed stack objects now narrow
   too, via a shared `narrowToFrameObjectBounds` helper called from both
   `ISD::FrameIndex` and `materializeFrameIndexAddrBase`
   (`CapstoneISelDAGToDAG.cpp`); lit `cap-shrink-stack.ll` extended
   (`cap_slot`/`field_store`). **Increment 2 DONE (2026-07-03,
   `history/03-07-2026_00-00-05_dynamic-alloca-stack-narrowing.md`):** dynamic
   (runtime-sized) allocas now narrow — `lowerDYNAMIC_STACKALLOC` shrinks the
   returned pointer to `[cursor, cursor+alignedSize)` while `sp`/X2 stays broad;
   lit `cap-shrink-dynalloca.ll`; runtime probes `stack_dynalloca_{inbounds,oob}`
   (in-bounds ok / OOB bounds-fault); **varargs save-area already covered** via the
   fixed-object path. Also **fixed a pre-existing orthogonal limitation**:
   `lowerDynamicAllocaSizeToXLen` now materializes memory-sourced (`-O0`) alloca
   sizes into an XLen register instead of erroring
   (`Unsupported dynamic alloca size expression`) — a general fix that unblocks
   `-O0` dynamic allocas. Full Capstone lit **36/36**, authority suite green.
   **DEFAULT NOW ON (2026-07-03).** `-capstone-shrink-stack` was flipped to
   `cl::init(true)` after **two independent full default-on regression matrices**
   (this session, HEAD `099a55b22fbf`) agreed the `-O0` suite is clean: lit
   **36/36** (with the default actually flipped), authority **26/26** (incl.
   `stack_dynalloca_{inbounds,oob}`), CoreMark, RV8 **7/7**, BEEBS **82/82** incl.
   rijndael, and **zero shrink-specific regressions at `-O1/-O2/-O3`** (flag on/off
   byte-identical). Lit fallout from the flip was 3 orthogonal tests
   (`dynamic-alloca`/`i128-xlen-lowering`/`ptr-arith`) pinned to
   `-capstone-shrink-stack=false`; the two dedicated shrink tests' `NOSHRINK`
   arms now pass `=false` explicitly. Pass `-capstone-shrink-stack=false` to
   recover un-narrowed stack bounds. — The earlier default-on empirical matrix was
   **DONE (2026-07-02,
   `/tmp/capstone/stack-shrink-default-on-results.md`)**. At the canonical `-O0`:
   authority 23/23, CoreMark all levels, RV8 7/7, BEEBS was **81/82** — the single
   regression **rijndael** is now **TRIAGED + FIXED (2026-07-03,
   `history/03-07-2026_00-00-04_rijndael-stack-shrink-oob-triage-and-fix.md`)**:
   verdict = **genuine over-read the narrowing caught (feature working)**, not a
   too-tight bound. Root cause: `aes.h` `typedef unsigned long word` (8 bytes on
   rv64) with the comment "must be a 32-bit storage unit" → `word_in(in_blk+12)` =
   `*(unsigned long*)(in_blk+12)` = 8-byte load at +12 of a 16-byte AES block.
   Fixed in `build-beebs-rijndael-capstone.sh` (patch `word` → `unsigned int`);
   rijndael now **PASS** both default and `-shrink-stack` (correctness marker
   validated, no fault). The fix is rijndael-isolated → `-O0` stack-shrink BEEBS is
   now effectively **82/82**.
   The `-O1/-O2/-O3` mass failures are **pre-existing, not stack-shrink-specific**
   (i128 `xor`/`or` ISel gap, fp128 materialize, `cscincoffset` assert — RV8 is
   0/7 at `-O1+` with or without shrink), so they are not a clean signal.
   Path to default-on (all **DONE**): ~~(a) resolve the rijndael `-O0` case~~;
   ~~(b) varargs save-area + dynamic `alloca` increments~~; ~~a full clean
   default-on matrix~~ → **default flipped on 2026-07-03**. Residual C1 gaps now:
   **subobject** bounds and **inter-procedural** provenance (spill slots +
   variable-size objects remain excluded by design).
7. trio size-aware `realloc` + one canonical bounded allocator.
8. Separate RX code / RW data, tighten perms, constrain function caps.
9. **Root elimination via trusted `SPLIT`** — the likely Capstone-specific
   contribution (reviewer decision; reframes the paper as
   **provenance + attenuation + root-elimination**).

## Parallel prerequisite: SQLite runtime — QEMU FIX LANDED, re-enable memcpy next

Compiler gaps 1–2 are fixed (recursive nested-aggregate cap-global init, task
#71; clang memcpy-from-private-template of cap aggregates, task #72). Gaps 3–4
were a single QEMU limitation (untagged `ldc`/`stc` zeroed the high 64 bits, so
no in-domain `memcpy` preserved both plain data and capability tags). **The QEMU
author confirmed this is a bug and that the intended model is full 128-bit
round-trip; the fix is now implemented (2026-07-01)** on `capstone-qemu` branch
`fix/untagged-ldc-stc-128bit-preservation` (Option A: widen the untagged register
path with `scalar_hi`). Validated by the new authority probe
`untagged_cap_roundtrip` (both 64-bit halves survive) + full authority suite
green. Details: `design/untagged-cap-loadstore-preservation-proposal.md` §8.

**SQLite gaps 3–4 CLEARED (2026-07-01).** The tag-preserving `memcpy`/`memmove`
is re-enabled (`beebs_freestanding_string.c`, shared by SQLite + BEEBS; copies the
pointer-aligned middle via `ldc`/`stc`, byte head/tail). Validated: BEEBS 82/82,
200k-case libc fuzz, and authority probes `tagged_cap_memcpy_aligned` (a tagged
cap survives `memcpy`) + `tagged_cap_memcpy_misaligned` (documents the alignment
limit). SQLite now runs **past** the data-corruption gap.

**SQLite gap 5 — FIXED (2026-07-01).** The `helper_cscincoffset: rs1_v->tag`
abort was an `int + ptr` commutative-add operand-ordering bug: `cscincoffset` got
an untagged **integer** as the capability base. Fixed in `selectCIncOffset`
(`CapstoneISelDAGToDAG.cpp`) — the shared chokepoint for both the raw `ISD::ADD`
i128 and `CapstoneISD::CIncOffset` ISel paths — by applying the same
predicate-based swap `CapstoneTargetLowering::LowerADD` uses
(`isCapstoneIntegerOffset` / `isCapstoneCapabilityValue`, now shared via
`CapstoneISelLowering.h`) so the capability is always the base. Lit **34/34** (new
`cap-cincoffset-base.ll`); the cscincoffset assertion is gone.

**SQLite gap 6 — FIXED 2026-07-03 (Option 1: 16-align `saveBuf`).** Step-0
diagnostic (pc-gated at `memcpy+0x1fc`, correlating each strip with its source
byte-load + memcpy caller) resolved it to **Case A**: the primary loss is a genuine
tagged, 16-aligned capability byte-copied to a *relatively-misaligned* destination —
`sqlite3NestedParse`'s `char saveBuf[PARSE_TAIL_SZ]` (the Parse-tail save/restore
buffer) placed at a 12-mod-16 slot. Because the relative misalignment is constant,
no memcpy change (Option 2) can preserve the tag; the fix is layout. Landed:
`build-sqlite-capstone.sh` `sed`-patches `saveBuf` to `__attribute__((aligned(16)))`
(+ verification grep) so both save/restore copies use `memcpy`'s aligned `ldc`/`stc`
fast path. **SQLite now runs past `sqlite3DeleteTable`.** No regression risk to
BEEBS/RV8/CoreMark (SQLite-amalgamation-only; shared `memcpy` + QEMU untouched).
Authority probe `tagged_cap_saverestore_aligned_buf` added (round-trips a tagged
cap through a 16-aligned `char[]`; oracle `ok`); full authority suite green.
Detail: `history/02-07-2026_00-00-00_sqlite-gap5-fix-and-gap6-investigation.md`
("Gap 6 FIXED"); design `design/sqlite-gap6-memcpy-tag-preservation-proposal.md`
(IMPLEMENTED).

**SQLite gap 7 — FIXED 2026-07-03 (compiler, `CapstoneCapGlobalInit`).** The
`cscincoffset` untagged-base assert was in `sqlite3VdbeExec` doing
`sqlite3aEQb[opcode]`. `sqlite3aLTb/aEQb/aGTb` are global pointers initialized to
an **interior address of another global** (`&sqlite3UpperToLower[256(+6/+12)-OP_Ne]`).
`needsMaterialization` used `stripPointerCasts()` (strips only zero-index GEPs), so
these interior-pointer globals were never tag-materialized → loaded untagged. Fix:
peel the ConstantExpr GEP/cast chain inbounds-agnostically (SQLite's index is only
in-bounds via appended array bytes, so clang doesn't mark the GEP `inbounds`).
Validated: lit `static-cap-global-init-interior.ll` + Capstone lit 35/35; authority
25/0; CoreMark/RV8 7-7/BEEBS 82-82 (shared-pass gate; failures seen were pre-boot
infra flakes, pass standalone). Detail:
`history/03-07-2026_00-00-00_sqlite-gap7-interior-pointer-cap-global-init.md`.

**SQLite gap 8 — FIXED 2026-07-03; SQLite now runs END TO END.** The unaligned cap
`stc` was in `allocateCursor`, which embeds a cap-bearing `BtCursor` at
`SZ_VDBECURSOR(nField)` — only 8-aligned (`ROUND8` + `(N+1)*sizeof(u64)`). Fix:
`build-sqlite-capstone.sh` `sed`-patches the `BtCursor` offset (and its allocation
size) to 16-align. SQLite-source only (no shared code/QEMU/compiler change), so the
benchmark gate is unaffected. **Confirmed with pristine QEMU:** the domain emits
`row name=alpha value=11 / beta=22 / gamma=33` and `__CAPSTONE_SQLITE_MEMORY_PASSED__`
— CREATE/INSERT/SELECT all correct. **Gaps 1–8 all resolved; SQLite bring-up is
complete.** Detail:
`history/03-07-2026_00-00-01_sqlite-gap8-unaligned-cursor-and-full-pass.md`.

**Follow-ups (not blockers):** (1) QEMU aborts (`riscv_cpu_do_interrupt` assert
`env->priv < PRV_C`) when an in-domain cap fault is raised — cap faults aren't
deliverable in PRV_C; worth fixing so faults are clean/catchable. (2) The alignment
gaps (6 `saveBuf`, 8 `BtCursor`) are a class — SQLite hand-packs pointer-bearing
regions at 8-aligned offsets; wider workloads may surface more, each a localized
16-align patch.

**Workload hardening (2026-07-03).** Extended the domain test to a richer SQL
workload (transaction, secondary INDEX, bound prepared inserts, REAL column,
UPDATE/DELETE, aggregates+sorter, ORDER BY, JOIN, GROUP BY, string funcs) — all
**pass** on the capability machine (`__CAPSTONE_SQLITE_EXTENDED_PASSED__`). One new
gap (9) surfaced and is a client-API artifact, not a core cap gap: the build's
`SQLITE_TRANSIENT`→function patch is applied only in the amalgamation, so the
public `sqlite3.h` sentinel (`-1`) isn't recognized by the core; a client
`bind_*(SQLITE_TRANSIENT)` gets `-1` stored as a destructor and later called
(`cjalr` on `-1`). Worked around (`SQLITE_STATIC` for persistent buffers) +
documented; proper fix = make the core accept `-1` too, or fix the clang
constant-eval crash so no sentinel substitution is needed. Detail:
`history/03-07-2026_00-00-02_sqlite-workload-hardening-and-gap9-transient.md`.

--- superseded gap-6 investigation notes (kept for provenance) ---

SQLite faulted with `Cap mem access requires capability` in
`sqlite3DeleteTable` on an untagged `Table*` (`0x102247f50`). A storage-slot-keyed
QEMU trace (keyed by value **and** the granules it is stored into; since reverted,
submodule clean) settled it from reliable helper-argument data (tag/addr/size —
`env->pc` is lazily synced here and useless for pinning the store): **every `stc`
of the pointer is tagged (89/89)**; the value is read back **untagged** at exactly
two **stack** granules. The transition granule is stored **tagged**, then cleared
by **byte-wise scalar stores (`size=1`)**, then read **untagged**; DeleteTable's
own slot then receives it already-untagged and faults (its slot is otherwise
clean: `stc`→scalar-load→`ldc`). **Mechanism: a live tagged capability in a 16-byte
stack granule has its tag stripped by a byte-wise memory copy**
(`memcpy`/`memmove`/small struct-copy lowered to `sb`/`sh`/`sw`) — the sub-16-byte
scalar-copy path that bypasses `ldc`/`stc` (same class as
`tagged_cap_memcpy_misaligned`). NOT the heap `HashElem` (superseded), allocator,
or value-motion. **EXACT site pinned (2026-07-02):** threading a translate-time pc
(`ctx->base.pc_next`) through `helper_remove_cap_mem_map`, plus the correct load
base `0x101ff0000` (the earlier `…6000` was off, hence prior incoherent symbols),
identifies the stripping store as the domain freestanding **`memcpy`'s byte-copy
loop** (`memcpy+0x1fc`, `sb`) copying the 16-byte-aligned `Table*` **byte-by-byte**
because src/dst are **relatively misaligned mod 16** — so `memcpy`'s own
tag-preserving `ldc`/`stc` fast path (22 ops in its body) can't apply. Caller is in
the `sqlite3NestedParse` region. **Fix is primarily runtime-library** (smaller than
feared): make `memcpy`/`memmove` preserve tags for any 16-byte-aligned cap granule
inside a relatively-misaligned copy (query/repair via `ldc`/`stc`), and/or 16-align
cap-bearing structs. **Fix proposal written:
`design/sqlite-gap6-memcpy-tag-preservation-proposal.md`** (awaiting review) — step 0
is one diagnostic (capture the culprit `memcpy(dst,src,n)` + source granule tag
state to decide "under-aligned cap struct" vs "stale dest tag"), then Option 1
(16-align the cap-bearing struct, reuse `memcpy`'s existing fast path) and/or
Option 2 (tag-aware `memcpy`), plus a new authority probe
`tagged_cap_memcpy_relmisaligned`. General finding in
`design/research-decisions-log.md`. Full detail:
`history/02-07-2026_00-00-00_sqlite-gap5-fix-and-gap6-investigation.md`.

The other queued `capstone-qemu` item, revocation enforcement (task #70,
`design/revocation-enforcement-proposal.md`), has its **enforcement half wired on
the memory-access path** (`capstone_cap_revoked` in `_helper_access_with_cap`
raising `RISCV_EXCP_INVALID_CAP`, plus the reload-untag in
`helper_reg_set_cap_compressed`; gated by `CAPSTONE_REVOCATION_ENFORCE`, default
on). **RECORDING FIX LANDED + VALIDATED (2026-07-03; QEMU submodule
`8b6a47f322` on `capstone-bootstrap`, parent pointer bumped).** The
**recording** side (`cap_rev_tree_revoke`, `cap_rev_tree.c`) had a self-comparison
loop guard (`_CAP_REV_NODE(tree, node_id).depth > depth` with `depth =
node_id.depth`, always false) that recorded nothing; fixed to test the walked
node's depth (`cur`), so revoke now invalidates the junior subtree. Revocation now
**bites**, validated three ways: **record** (junior subtree invalidated),
**enforce** (revoke-matrix: revoked cap reloads untagged, use-after-revoke store
dropped), **re-share** (payload-revoke with a non-linear `REV_DEFAULT` borrow:
revoke→re-share→round 2→**success**). No effect on non-revocation workloads
(`cap_rev_tree_revoke` runs only on an explicit `csrevoke`). The runtime author
greenlit the experiment (spec §8 model confirmed). **#70 NOW FULLY END-TO-END
(2026-07-03): both follow-ons resolved + validated**
(`history/03-07-2026_00-00-07_step-b-clean-in-domain-fault-delivery.md`).
**(b) clean fault delivery:** a caught use-after-revoke now **terminates the
domain and returns to the caller** (sentinel `0x0FA017ED`) instead of the monitor
spinning — fixed in `sbi.dom`'s `swap_cpmp`/`handle_exception`
(`fault_return_from_domain`, reusing the `DOM_RETURN` unwind);
`run-revoke-matrix-probe.sh` PASS (both cases). **(a) linear re-share:** revoking
a linear `REV_BORROWED` borrow leaves the handle UNINIT; the ISA gap was that
`revoke` left `cursor==base` while `csinit` needs `cursor==end` and `scc` refuses
UNINIT. Fixed in QEMU `helper_csrevoke` (UNINIT → `cursor=end`) + `csinit`-before-
`mrev` in `shared_region_annotated` (both `sbi_capstone.c` copies);
`run-hostcall-all.sh` green (12/12, was red across the linear-re-share probes once
recording became active). Authority suite still `__CAPSTONE_AUTHORITY_SUITE_PASSED__`.
**Key architecture note:** two monitors — the **OpenSBI firmware** (`fw_jump.elf`,
submodule `components/opensbi`) handles lender **SBI ecalls** (region
share/revoke/`mrev`); **`sbi.dom`** (submodule `package/.../capstone-sbi`) handles
the borrower's **in-domain faults**. Fix each in the copy that runs the path.
(a) is an experimental revocation-semantics choice worth a one-line author
confirm. Changes are **uncommitted** (nested-submodule chain: capstone-qemu +
opensbi + capstone-sbi + buildroot pointer + parent). Prior dormant→recording
trail: `history/03-07-2026_00-00-06_revocation-70-verify-still-dormant.md`.

The earlier benchmark milestones (RV8, BEEBS, backend fixes) are retained below as
reference/history.

---

## RV8 suite — complete (3rd of 3: CoreMark ✓, BEEBS ✓, RV8 ✓)

The RV8 benchmark suite (`https://github.com/michaeljclark/rv8-bench`) is stood up
under `capstone/benchmarks/rv8/` (split layout like CoreMark: `fetch-rv8.sh` →
`/tmp/capstone/rv8-src`, pinned commit; no submodule). Shared adapted runtime:
`adapted/rv8_capstone_preamble.h`, `rv8_malloc.c` (16-aligned bump allocator),
`rv8_stubs.c` (no-op gettimeofday/printf/exit), reusing the BEEBS freestanding
libc + `beebs_simple_domain.c` harness.

**All 7 RV8 C benchmarks PASS**: dhrystone, qsort, sha512, aes, primes, norx,
miniz (`run-rv8-<name>.sh` → `__RV8_<NAME>_PASSED__`). Each adapts the hosted
program to the domain (strip hosted includes + `-include` preamble, stub
gettimeofday/printf/exit, 16-aligned bump `malloc` with size-tracking `realloc`,
reduce workloads, self-contained oracles). norx required a **backend fix**
(stack-passed capability args delivered untagged — see backend-work note above).
miniz reduced to core compress/uncompress with an enlarged arena. **Only
`bigint` (C++) remains, deferred** — assessed (2026-06-24) as a full "C++ on the
domain" bring-up, not a benchmark adaptation: (1) no C++ STL for capstone64
(`<vector>`/`<string>`/`<iostream>` absent; bigint's `Nat` is `std::vector`-backed
+ `std::cout`); (2) the backend crashes on `new <type>` expressions (`APInt::zext`
assertion; repro `capstone/tests/cxx-new-expr-crash.cc`). See
`capstone/benchmarks/rv8/README.md`. So the three benchmark suites (CoreMark,
BEEBS, RV8) are complete for C; SQLite / real software is the next stage.

## Current BEEBS milestone - 82 benchmarks validated (suite complete)

### Recent backend work (2026-06-24): Bug #3 fixed; capability globals tagged

`Bug #3` (i128 non-vector-shift legalization assertion) is **fixed in the
backend** — `lowerScalarI128Shift` now has a general constant-shift fallback for
operands the narrowing helper can't recognize (notably the `ashr/lshr i128` a
pointer-difference `(p-q)/sizeof(T)` lowers to). Validated by a domain probe + the
`matmult-int` repro + new lit coverage in `i128-xlen-lowering.ll`.

**Capability-global tagging is resolved** (constructor-codegen). A capability tag
cannot live in a static ELF image, so initialized capability globals (pointer
tables, string tables like `dtoa`'s `char *nums[]`, function-pointer tables)
loaded **untagged** and faulted on first use. New IR ModulePass
`llvm/lib/Target/Capstone/CapstoneCapGlobalInit.cpp` synthesizes a per-module
`__capstone_cap_init` that stores each capability-global element in place at
runtime (isel lowers to a tagged `cincoffset gp`+`delin`+`stc`);
each initializer has internal linkage and is registered through the
PC-relative `.capstone_cap_init` table that
`capstone/my_first_domain/start.S` iterates before `domain_main`. Empty tables
are a no-op for domains with no capability globals. Validated end-to-end: the
three previously faulting `static-cap-typed-load-repro` domains now pass
unchanged (string-struct, array=`nums[]` shape, function-pointer); Capstone lit
tests; `bs` unaffected. Decision + implementation:
`capstone/agent-handoff/design/capability-globals-init-decision.md`; test
`static-cap-global-init.ll`.

Multi-module update (2026-06-24): the per-module `__capstone_cap_init` + GCT
markers were strong symbols that collided in multi-module links (regressed
CoreMark: duplicate-symbol link error). Fixed by making each initializer
**internal** and registering it via a **PC-relative `.capstone_cap_init` offset
table** (`emitCapGlobalInitTableEntry`) that `start.S` iterates before
`domain_main` (absolute `.init_array` is unusable — the domain processes no
load-time relocations). GCT begin/end markers made weak. **CoreMark links + runs
again** ("Correct operation validated"); repro / RV8 7/7 / BEEBS `bs` / 29 lit
all green. Details: `design/capability-globals-init-decision.md`.

This resolved `dtoa` **blocker #1** (untagged `nums[]`). `dtoa` is now
**RESOLVED** end-to-end (`run-beebs-dtoa.sh` → `__BEEBS_DTOA_PASSED__`, oracle
267945, upstream `benchmark`/`verify` unchanged): blocker #2 (arena 16-byte
alignment for the 16-byte `Bigint.next` capability) was fixed in
`build-beebs-dtoa-capstone.sh` via `-DOmit_Private_Memory` + a 16-aligned
`heap[]` + 16-byte-rounded `malloc_beebs` (integer rounding, no pointer forging).
`dtoa` was the last deferred BEEBS benchmark → **81**. See
`plans/beebs-deferred-benchmarks.md` §3 (shift fix) and §15 (`dtoa`, RESOLVED).

### Benchmarks

82 BEEBS benchmarks now pass end-to-end and the suite is **effectively
complete**: the only upstream `src/` dirs without a runner are `matmult-int`
(byte-identical to `matmult`, already built `-DMATMULT_INT`) and `trio` (the
shared trio library that `trio-sscanf`/`trio-snprintf` both build from — not a
standalone benchmark).

The most recent additions are `dtoa` (81st) and `trio-snprintf` (82nd).
`trio-snprintf` (`run-beebs-trio-snprintf.sh`) builds the shared `src/trio`
library `-DTRIO_SNPRINTF` with `TRIO_FEATURE_FLOAT=0` (integer formatter only, no
long-double/fp128), reusing the `trio-sscanf` preamble/stubs. Its upstream
`verify` is -1, so an adapted oracle (`adapted/beebs_trio_snprintf_tail.c`) runs
the five `trio_snprintf` conversions and checks each formatted string exactly
(`"123"`, `"123"`, `"  123"`, `"0007b"`, `"   10"`). The `va_list` + capability-
global fixes from this session made it build/run with no new backend work.

The prior milestone addition was `janne_complex`
(`run-beebs-janne_complex.sh`) — a trivial integer WCET
benchmark (nested data-dependent loops). It is fully self-contained: integer
only, includes only `support.h`, and its upstream `verify_benchmark` returns
`r == 1` (which `complex()` always yields), so it needs no soft-float, no libm,
no string lib, no adapted tail, and no host reference. The three wrappers just
delegate to `build-beebs-simple-{capstone,host}-common.sh` /
`run-beebs-simple-common.sh` (same minimal pattern as `bs`). No compiler change.

The prior addition was `fasta`
(`run-beebs-fasta.sh`) — the first of the libc-frontier benchmarks. Upstream
`fasta` discards all output and `verify_benchmark` returns -1, so the adapted
tail (`adapted/beebs_fasta_capstone_tail.c`) keeps the deterministic generator
core (`myrandom` LCG + `accumulate_probabilities`) and reimplements the two
consumers (`repeat_fasta`/`random_fasta`) to fold every generated character into
an FNV-1a checksum, compared exactly to a same-source host reference
(`0x24d70971e2d6dc0f`; `myrandom`'s f32 ops are correctly-rounded on both host
hardware float at `-ffp-contract=off` and target compiler-rt soft-float, so the
character stream is bit-identical). It introduced the shared freestanding
string/mem library `adapted/beebs_freestanding_string.c`
(`memcpy/memmove/memset/strlen/strcmp/strcpy` — the "pure computation" slice of
libc, locally implemented, the string counterpart to `beebs_softfloat_libm.c`;
`-ffunction-sections`/`--gc-sections` drops the unreferenced routines) and added
`floatdisf`/`floatundisf` to the shared soft-float builtin set. The host-gcc
recompute matches the reference bit-for-bit, confirming the generator is
compiler-independent. No compiler change.

The prior additions were
`matmult-float` and `whetstone` (`run-beebs-{matmult-float,whetstone}.sh`),
which complete the soft-float/libm-only FP class. Both reuse the soft-float
builtins (+ shared libm) with no compiler change, and both use the proven
"reference computed from the same source + same soft-float math, compared
exactly" pattern (IEEE float/double ops are bit-identical between host hardware
float at `-ffp-contract=off` and target compiler-rt soft-float).

- `matmult-float`: the same source as `matmult` built `-DMATMULT_FLOAT`
  (UPPERLIMIT 10, float[10][10]); soft-float builtins only (no libm). The adapted
  tail replaces the upstream local-`exp[][]` verifier (Bug #3/#9) with an FNV-1a
  checksum of the global `ResultArray` read as a flat byte stream (oracle
  `0xbdbace3d315e67a4`). Built `-ffunction-sections`/`--gc-sections` so the dead
  upstream `values_match` (which would pull in `frexpf`/`fabsf`) is dropped.
- `whetstone`: needed `atan` added to the shared `adapted/beebs_softfloat_libm.c`
  (fdlibm port, ~1.6e-16, validated by the self-test). Upstream `verify` is -1
  and the per-module results flow only through `POUT` (gated on `PRINTOUT`), so
  the domain is built `-DPRINTOUT`, the upstream printf `POUT` definition block
  is stripped, and the adapted tail's capturing `POUT` folds every module's four
  doubles into an FNV checksum compared (exact) to a same-libm host reference
  (`0x2f975c4609a1bfbb`).

The prior addition was
`stb_perlin` (`run-beebs-stb_perlin.sh`), a 3-D Perlin-noise benchmark. Its
oracle is self-contained: `benchmark()` computes a 10x10 noise plane and
compares every value against a `static const float expected[10][10]` global
(in `.rodata`, so no Bug #9), returning 0 iff all 100 match exactly. The adapted
tail just checks `res == 0`. Its only external dependency is `floor`, newly
added to the shared `adapted/beebs_softfloat_libm.c` (bit-exact, validated by
the libm self-test); everything else is the existing soft-float builtins. Built
`-ffp-contract=off`; host (gcc -O0 -ffp-contract=off) and target match the
embedded table bit-for-bit. No compiler change. Note: `matmult-int`'s upstream
source is byte-identical to `matmult/matmult.c`, which `run-beebs-matmult.sh`
already builds with `-DMATMULT_INT`, so it is effectively already covered.

The prior step added the four
`newlib-*` single-precision math benchmarks `newlib-sqrt`, `newlib-exp`,
`newlib-log`, `newlib-mod` (`run-beebs-newlib-{sqrt,exp,log,mod}.sh`). Each
`src/newlib-*/ef_*.c` is **self-contained** — it ships its own routine
(`__ieee754_sqrtf`/`expf`/`logf`/`fmodf`, integer bit-manipulation plus
non-contracted float arithmetic) with no libm/libc calls — so they reuse only
the soft-float builtins (`build-beebs-softfloat-common.sh`); no libm object, no
compiler change. Built with `-ffp-contract=off` so no FMA contraction can
diverge from the soft-float reference. `newlib-sqrt` keeps the upstream exact
`==` verifier (its `exp[]` is moved to `static const` to avoid Bug #9; the
correctly-rounded `__ieee754_sqrtf` is bit-identical to the embedded newlib
values); `newlib-exp/log/mod` have upstream `verify_benchmark == -1`, so each
gets an oracle tail that captures all five calls and exact-bit-compares them
against a host reference (`gcc -O0 -ffp-contract=off` over the same source).

The prior additions `qsort`,
`qurt`, and `select` (`run-beebs-{qsort,qurt,select}.sh`) — FP benchmarks needing
only the soft-float builtins (no libm; they ship their own helpers or use only
float compares), each with an adapted oracle tail (upstream verifiers return -1):
`qsort` widens `arr` to [21] and checks monotonicity plus a host-reference hash
over the sorted 1-indexed region; `qurt` captures and checks all three known
quadratic root cases (tolerance — it uses its own approximate sqrt);
`select` widens `arr` to [21] (fixing a latent 1-indexed over-read) and compares
the captured k-th return against a host reference. No compiler change.

The prior batch `frac`/`st`/`nbody` (`run-beebs-{frac,st,nbody}.sh`) reuses the
shared libm; `st`/`nbody` drove the correctly-rounded `sqrt`.

Two reusability changes: the libm is now the neutrally-named, shared
`adapted/beebs_softfloat_libm.c` (was `beebs_cubic_libm.c`), and its `sqrt` is now
**correctly-rounded** (Newton seed + exact two-product residual + round-to-
nearest-even; bit-exact vs the host over 230M values). The correctly-rounded sqrt
is required by benchmarks that compare results for **exact** equality (`st`,
`nbody`); `frac` needs only `fabs`. `cubic` re-verified after the sqrt change.
`ludcmp`/`minver` (prior additions) reuse the soft-float builtins only.

A runtime trace (instrumenting `helper_cscincoffset`, since reverted) showed the
earlier `ludcmp` `cscincoffset rs1->tag` crash was **not** a `cincoffset`
operand/canonicalization bug: the matrix algorithm runs fine. It is the
documented **Bug #9** (a `verify_benchmark` *local* const-initialized array,
`float exp_a[8][9]={...}`, lowered to a `memcpy` from `.rodata` into a stack array
whose destination capability comes back untagged). Workaround (source, no compiler
change): mark `exp_a`/`exp_b`/`exp_x` `static const` so they live in `.rodata`
(no stack copy, no `memcpy`) — same class of fix as mergesort / nettle-*. `minver`
needed only a correctness oracle (its upstream verify returns -1): an FNV-1a
checksum of the inverted matrix `a_i` + `det` vs a native float reference.
The **Bug #9 backend root cause** (untagged stack dest in a rodata→stack copy)
remains an open, deferrable backend task — see `plans/beebs-deferred-benchmarks.md`.

`sqrt` (63rd) is validated with `run-beebs-sqrt.sh` — the first FP benchmark to
*reuse* the soft-float runtime (no compiler change). It needs no libm (it ships its own
`sqrtfcn`) and has a real `verify_benchmark`; the only new infrastructure is the
shared `build-beebs-softfloat-common.sh` helper, which compiles the compiler-rt
float+double soft-float builtin set and is now also sourced by `cubic`.

`cubic` (62nd) is the **first floating-point benchmark**, validated with
`run-beebs-cubic.sh`.

`cubic` required standing up a soft-float + libm runtime (see
`design/capstone-softfloat-libm.md`). Two backend changes:
(1) `CapstoneSystemLibrary` in `RuntimeLibcalls.td` registers the runtime
libcall-name table (FP libcalls previously aborted at `TargetLowering.cpp:189`
with "unsupported library call operation" because the table was empty);
(2) a pre-legalize `ISD::ConstantFP` DAG combine in `CapstoneISelLowering.cpp`
loads fp128 constants from the constant pool (`ldc`) instead of softening them
into an unforgeable 128-bit capability immediate. The genuine capability-forge
guard (`inttoptr` of a wide integer) is unchanged (`cap-constants-invalid.ll`).
Runtime: `SolveCubic`'s `long double` is reduced to `double` (documented source
adaptation — avoids fp128 quad soft-float, which would also need an i128
non-vector-shift backend fix); doubles use compiler-rt soft-float builtins; a
compact self-contained `adapted/beebs_softfloat_libm.c` provides
`fabs/sqrt/exp/log/pow/sin/cos/acos` (validated <1e-12 vs system libm). Verified
against the exact mathematical roots {2, 2.5, 6} and {2.5}.

`compress` (61st) is validated with `run-beebs-compress.sh`: pure-integer, no
compiler change, FNV-1a checksum of the LZW work product
(`in_count`/`out_count`/`free_ent` + `htab`/`codetab`) vs a native LP64 host
reference. Its historically documented "backend crash" was already stale.

`compress` no longer crashes the backend (the historically documented
"pre-existing backend crash" was resolved by intervening backend fixes); it is
a pure-integer source adaptation with no compiler change. Its upstream
`verify_benchmark` returns -1 ("no verification") and this BEEBS variant never
calls `output()`, so `comp_text_buffer`/`bytes_out` stay empty. The adapted tail
(`adapted/beebs_compress_capstone_tail.c`) instead checksums the LZW work
product (`in_count`/`out_count`/`free_ent` + `htab`/`codetab`) with FNV-1a
against a native LP64 host reference — exercising capability-mode array indexing
as a real correctness gate.

`run-all-beebs.sh` now has low-token aggregate output: child wrapper output goes
to `$CAPSTONE_TMP_ROOT/run-all-beebs/*.attempt-N.log`, while the aggregate prints
compact pass/fail lines. It is serial by default, with opt-in isolated
parallelism via `RUN_ALL_BEEBS_JOBS=N`; each attempt gets its own build/share
workspace under the aggregate log directory. It retries only structured QEMU
infra flakes before benchmark execution twice by default and caps aggregate
boot-to-login waits at 90 seconds (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`) so QEMU boot
flakes fail fast into that retry; real marker failures still stop immediately.

## Recent root fixes

Narrow truncating stores from i128 carrier to capability-addressed memory
are fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles `MemVT = i32/i16/i8`
  truncating stores by emitting `SW`/`SH`/`SB` respectively.  The large-offset
  CIncOffset decomposition is also extended to cover SW/SH/SB.
- This arises when a pointer-difference result (i64 in an i128 any_extend carrier)
  is stored into a narrower integer field (`int len = ptr1 - ptr2`).
- `slre` is the proof benchmark.

The large-offset capability load/store backend blocker is fixed:

- `selectLDC_STC` in `CapstoneISelDAGToDAG.cpp` now handles constant offsets
  > 2047 for `ldc`/`stc` by emitting `CIncOffset(base, offset)` then
  `ldc/stc rd, 0(adjusted)`.
- `sglib-rbtree` was the proof benchmark: its iterator struct pushes
  `equalto` and `subcomparator` past the 2047-byte immediate range.

Pointer arithmetic fixes now cover:

- `ptr - integer` and `ptr + (-offset)` as `cincoffset base, -offset`.
- True `ptr - ptr` by extracting both capability cursors with `lcc ..., 2`,
  subtracting the XLEN cursor values, and sign-extending back to the `i128`
  carrier when needed.
- C pointer subtraction in Clang CodeGen truncates the result to C `ptrdiff_t`
  when the target pointer integer type is wider. This avoids ICmp type
  mismatches when comparing pointer differences on Capstone.
- InstCombine `or disjoint` pointer-carrier additions are lowered as capability
  offset arithmetic when one operand is a known capability base and the other is
  a known integer offset.

Constant-pool lowering for large scalar constants is fixed:

- `lowerConstant` now emits `LOAD:i64(LGA:i128(TargetConstantPool))` directly
  when a large i64 constant is placed in the constant pool, so the final load
  has a capability base rather than a raw integer constant-pool address.
- `lowerConstantPool` returns the capability address (`LGA:i128`) like
  `lowerGlobalAddress`.

`qrduino` and `miniz` both needed benchmark-local source adaptations for static
string pointer data: keep string literals as arrays so benchmark code derives
capabilities from the global symbol rather than loading untagged raw pointers
from data.

`miniz` additionally uses generated scratch sources under
`$CAPSTONE_TMP_ROOT/beebs-build`, strips hosted includes, provides inline libc
stubs, expands/aligned its bump heap, and rounds allocations to 16 bytes.

Verified gates for this milestone:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv clang/test/CodeGen/cap-ptr-compare.c
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-beebs-ctl-string.sh
bash capstone/benchmarks/beebs/run-beebs-qrduino.sh
bash capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh
bash capstone/benchmarks/beebs/run-beebs-miniz.sh
bash capstone/benchmarks/beebs/run-beebs-slre.sh
bash capstone/benchmarks/beebs/run-beebs-wikisort.sh
bash capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh
```

## Remaining viable targets

No clean-add BEEBS target is known. Pick remaining targets only if you are ready
to fix a root issue or carry an invasive source adaptation.

Good next investigations:

- `trio`/`trio-snprintf`: the `va_list` capability storage/copying blocker is now
  **fixed** in the backend (`va_start`/`va_arg`/`va_copy` lower with `stc`/`ldc`
  and a 16-byte `cincoffset` stride; see `plans/backend-compiler-fixes.md`).
  `trio-sscanf` is validated with an embedded/minimal string-helper build.
  Full `trio` and `trio-snprintf` still need a deliberate soft-float/complex
  format-lib strategy; `trio-snprintf` also has `verify_benchmark = -1`, so do
  not add it as a normal correctness gate without changing the verifier story.
- FP-blocked benchmarks: require a deliberate soft-float/libcall strategy for
  Capstone, not one-off wrappers.

## Blocked (do not retry without root fix)

### FP-blocked: needs in-domain libm + libc (dtoa)

- `cubic`: **RESOLVED** (first FP benchmark). The runtime libcall-name table is
  now registered and an in-domain soft-float + libm runtime exists; see the
  milestone note above and `design/capstone-softfloat-libm.md`.
- `sqrt`: **RESOLVED** — pure soft-float (own `sqrtfcn`, no libm), real verify.
  Reuses `build-beebs-softfloat-common.sh`. `run-beebs-sqrt.sh`.
- `ludcmp`, `minver`: **RESOLVED** (Bug #9 source workaround / oracle, see the
  milestone note above). The earlier "cincoffset bug" hypothesis was disproven by
  a runtime trace; the matrix algorithm's `cincoffset`s are correct.
- `dtoa`: now compiles (libcall names resolve), but the bare-metal domain still
  lacks the libm/libc it needs — `log`/`floor`/`ceil` plus `malloc`,
  `memcpy`/`memmove`/`memset`, `strcpy`/`strlen`, `errno`, and freestanding
  `float.h`/`fenv.h`/`locale` shims (89 KB FP↔decimal library). The `cubic`
  soft-float runtime is reusable; `dtoa` mainly adds the libc surface. Larger
  follow-on. See `plans/beebs-deferred-benchmarks.md` (Bug #14).
- The soft-float runtime continues to unblock the remaining FP benchmarks below
  at the *compile* level; each still needs its libm closure linked + a
  correctness oracle (and exact-comparison verifiers need the correctly-rounded
  `sqrt`, now in place).

### Remaining uncovered benchmarks — the libc/format frontier only

The soft-float/libm-only FP class is now **complete** (`matmult-float` and
`whetstone` were the last two; `whetstone` is exact via the same-libm reference,
not a tolerance oracle). `matmult-int`/`matmult-float` source is byte-identical
to `matmult/matmult.c` (built `-DMATMULT_INT`/`-DMATMULT_FLOAT` respectively).
What remains are heavier, libc-dependent benchmarks, each its own effort:

- `fasta` - needs libc (`memcpy`/`strlen`/`malloc`-ish).
- `trio`, `trio-snprintf` - float / complex format lib (`trio-sscanf` is the
  validated proof wrapper; `trio-snprintf` also has `verify_benchmark = -1`).
- `dtoa` - heavy libc (`malloc`/`errno`/`float.h`/`fenv.h`/`locale`) + libm.

Plus the **Bug #9 backend root fix** (removes the `static const` source
workaround class across many benchmarks).

## Regression gate for backend/lowering/ABI changes

For non-trivial backend, lowering, ABI, or broad benchmark-runtime changes, do
not treat the change as fully validated until this full gate passes:

```bash
source capstone/tests/capstone-test-env.sh
"$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone
bash capstone/tests/runtime-qemu/run-coremark.sh
bash capstone/benchmarks/beebs/run-all-beebs.sh
```

Smaller BEEBS subsets are still useful for narrow wrapper/doc changes and quick
pre-commit smoke checks, but they are not the full backend validation gate.

For runtime/HostCall changes, use `capstone/tests/runtime-qemu/run-hostcall-all.sh`
as the normal proof gate. For OpenSBI/kernel/module changes, use
`capstone/tests/runtime-qemu/run-nullblk-all.sh`. Individual wrappers remain the
right entry points for focused reruns and diagnosis.

## Known backend limitations (document when encountered)

- **memcpy/memmove/memset libcall**: the Capstone backend crashes with null
  symbol name when generating calls to these. Always provide inline stubs
  instead.
- **cincoffset commutative bug**: fixed in lowerADD (isIntegerOffset now covers
  scaled-index GEP; isCapabilityValue distinguishes genuine ldc loads from
  sextloads). edn was the last benchmark blocked by this.

## What not to regress

Do not delete `capstone/caplifive-buildroot/build/local.mk` - its absence
silently switches the image to stock OpenSBI and breaks all runtime proofs.
