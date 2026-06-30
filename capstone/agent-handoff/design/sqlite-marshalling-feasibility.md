# Feasibility grounding: capability-mediated SQLite boundary marshalling

*Status: GROUNDING NOTE for the marshalling research direction (the collaborator
discussion summary in `../history/SqliteProposal.{pdf,tex}`). Not a commitment to
the direction — it maps that proposal onto the **actual** state of this tree so
the proposal can be reviewed against ground truth before we invest. Written after
a source-level inventory of the LLVM/clang toolchain and the Capstone QEMU on
`capstone-bootstrap` (2026-06-29). Per the "propose before big directions" norm,
this is the doc to review first.*

## What the proposal needs

The proposal (Pointer-Safe Marshalling at the Host–SQLite Boundary) argues that
Capstone can make the host↔SQLite lifetime contract safe-by-construction without
copying, using four mechanisms: **linear capabilities** (exclusive, move-only
borrows), **revocation capabilities** (explicit revoke at borrow end),
**hierarchical/senior revocation** (a `close` cascades to derived handles), and
**sealed/uninitialised** capabilities (safe callback domain-switch; safe reclaim /
use-before-init). Its two evaluation deliverables are (1) a binding co-designed
with the engine that mints and revokes capabilities, and (2) a measurement that
the capability-mediated borrow stays near raw-pointer cost and below the copy
(`SQLITE_TRANSIENT`) baseline.

The headline question for *us* is not "is the idea sound" but "**what already
exists in this tree, and what is the real gap?**" The answer (refined by the M0
spike, below): substantially more *plumbing* is wired than expected — the
primitive stack, the REV/LIN/UNINIT lifecycle, and the kernel-module borrow/revoke
ABI all work — **but the central safe-fail guarantee does not hold yet**: M0
showed that revocation is currently *recorded but not enforced*, so a
use-after-revoke still succeeds. The real gap is QEMU revocation enforcement, not
the toolchain.

## Ground truth 1 — the full primitive stack is wired, C → QEMU

Every primitive the proposal relies on is plumbed end-to-end (verified by tracing
each layer):

| Primitive | clang builtin | LLVM intrinsic | ISel (`CapstoneISelDAGToDAG.cpp`) | QEMU helper (`op_helper.c`) |
|-----------|---------------|----------------|-----------------------------------|------------------------------|
| revoke | `__builtin_capstone_cap_revoke` | `int_capstone_cap_revoke` | `selectRevoke` | `helper_csrevoke` (703) |
| mint revocation | `__builtin_capstone_cap_mrev` | `…_mrev` | `selectMrev` | `helper_csmrev` (714) |
| init (→linear) | `__builtin_capstone_cap_init` | `…_init` | `selectInit` | `helper_csinit` (839) |
| seal | `__builtin_capstone_cap_seal` | `…_seal` | `selectSeal` | `helper_csseal` (861) |
| delin (lin→nonlin) | `__builtin_capstone_cap_delin` | `…_delin` | (wired) | `helper_csdelin` (821) |
| tighten (perms) | `__builtin_capstone_cap_tighten` | `…_tighten` | `selectTighten` | `helper_cstighten` (793) |
| drop | `__builtin_capstone_cap_drop` | `…_drop` | `selectDrop` | (invalidates reg) |
| split | — (no builtin) | — | — | `helper_cssplit` (769) |
| shrink | `__builtin_capstone_cap_shrink` | `…_shrink` | `selectShrink` | `helper_csshrink` (729) |

Builtin signatures (`clang/include/clang/Basic/BuiltinsCapstone.td`): all are
`void*(void* cap, …)`. So linear/revocation/seal are reachable from C **today** —
no new instruction wiring is required to prototype. (`SPLIT` is the exception:
helper exists, but no intrinsic/builtin — root-elimination experiments would need
it wired.)

## Ground truth 2 — the capability type model and how a *linear* cap originates

`capstone-qemu/target/riscv/cap.h:26` — `CapType ∈ {LIN, NONLIN, REV, UNINIT,
SEALED, SEALEDRET}`; `captype_is_copyable` is true **only for NONLIN**, so LIN is
move-only at the hardware level (the proposal's exclusivity holds by construction).

The lifecycle the proposal's L+R mechanism requires:

```
UNINIT --csinit(offset)--> LIN --csmrev--> REV(retained by lender)
                            |                 |
                       delegated to        csrevoke  ==> sweeps rev-tree,
                       borrower             clears tag of every derived cap
                                            (use-after-revoke => tag fault)
```

- `helper_csmrev` asserts the input is `CAP_TYPE_LIN` and returns a `REV` cap
  tracked in `env->cr_tree` (the revocation tree).
- `helper_csrevoke` calls `cap_rev_tree_revoke`, invalidating derived caps
  (`tag = false`), so a later dereference traps as a **tag fault**
  ("Cap mem access requires capability") — *not* a bounds fault.

**Critical origin question (the #1 risk):** `csinit` requires an **UNINIT** source,
and the *only* producer of UNINIT in QEMU is `csrevoke` itself (circular). Ordinary
compiler-emitted pointers are **NONLIN** — roots (`sp`/`gp`) are delinearized at
domain entry, and everything derived from them is NONLIN. So **a linear capability
cannot be conjured from ordinary C pointers by the compiler.** It must enter the
domain from somewhere that already holds UNINIT/LIN authority. That somewhere is:

## Ground truth 3 — the borrow/revoke *plumbing* exists via HostCall shared regions (but is not enforced — see M0)

This is the load-bearing discovery — with an important caveat the M0 spike
established. The host↔domain boundary the proposal needs *already exists* as the
**HostCall shared-region** mechanism, and the borrow/revoke *operations* run end
to end. **However, this only demonstrates that the operations are plumbed, not
that revocation is enforced** — the existing probes below confirm the calls
succeed and that re-share-requires-revoke, but none of them tests that a
use-after-revoke *faults*. M0 (below) does, and finds it does not. Read this
section as "the wiring is present," not "the guarantee holds":

- `tests/runtime-qemu/shared-region-probe/` — the host (S-mode runtime) shares a
  region with the domain via SBI ecalls (`SBI_EXT_CAPSTONE_REGION_QUERY/COUNT`);
  the domain accesses it by region id. This is exactly "lender mints a capability
  to a buffer and delegates it to the borrower."
- `run-hostcall-second-pending-payload-probe.sh` — reusing/re-sharing a borrowed
  payload region across rounds **without** revoking first reproduces a
  `helper_csmrev` assertion (`rs1_v->val.cap.type == CAP_TYPE_LIN` fails: the
  region is no longer LIN because it is still borrow-shared).
- `run-hostcall-second-pending-payload-revoke-probe.sh` — the same shape but the
  host calls `revoke_region(id)` before re-sharing **succeeds**.

The runtime/QEMU author has confirmed the intended rule (README, runtime-qemu):

> if a region is already borrow-shared, it must be revoked before anything else
> can be done to it, including borrow-sharing it again.

That rule matches the *shape* of the proposal's revocation contract (revoke at
the borrow's end before the buffer is reused), and the engine→host "borrow result,
revoke at next `step`/`reset`/`finalize`" pattern (Table 4 rows 1–8) maps onto the
shared-region borrow + `revoke_region` primitive. **But "revoke succeeds and
enables re-share" is not "use-after-revoke faults"** — that stronger property is
what M0 tests and what is currently missing (see below).

## Ground truth 4 — existing SQLite scaffolding

- `tests/runtime-qemu/sqlite-vfs-skeleton/` + `build-sqlite-vfs-skeleton.sh`
  already build a Capstone-compiled SQLite-facing VFS skeleton domain against the
  **official SQLite 3.53.1 amalgamation** (auto-fetched from sqlite.org), at `-O2`.
  (The in-progress benchmark bring-up should reconcile with this; see Open items.)

## What this re-frames in the milestones

Because the borrow/revoke *operations* run (but are not enforced), **M0 is not
"can we revoke" — it is "does use-after-revoke actually fault?"** The answer (M0
result, below) is **no, not yet**, so the revised staging puts enforcement first:

- **M0 — characterize + harden the existing borrow/revoke.** Lift the
  shared-region revoke probe into a named, oracle-classified test (the way the
  authority suite classifies traps): borrow→revoke→use-after-revoke must produce a
  deterministic **tag fault**, and a senior/hierarchical revoke must cascade to a
  derived sub-region. This is the spike; it reuses existing assets.

  **M0 result (2026-06-29) — done; surfaced a load-bearing GAP.** Built and ran
  `tests/runtime-qemu/borrow-revoke-uaf-probe/` (lender lends a region as a
  revocable borrow, revokes it between two domain calls; the borrower caches the
  delegated pointer in round 1 and dereferences it in round 2). The borrow and
  the `revoke_region()` both succeed, **but the use-after-revoke is NOT trapped**:
  the round-2 store lands and the lender observes the stage-2 sentinel
  (`0x2222…`). This **contradicts the proposal's central "a subsequent
  dereference then faults" claim in the current runtime configuration.** Two
  incidental findings: (a) only the `REV_BORROWED` annotation (0x1) establishes a
  revocable relationship — `REV_SHARED` (0x2) makes `revoke_region` assert in
  `helper_csrevoke` (`type == CAP_TYPE_REV` fails); (b) this is a distinct issue
  from the known `helper_csmrev` `CAP_TYPE_LIN` assertion on re-share-without-revoke.

  **Root cause (CONFIRMED, 2026-06-30) — revocation is recorded but not enforced.**
  `cap_rev_tree_revoke` (`cap_rev_tree.c:81`) sets the descendant rev-tree nodes'
  `valid = false`. But that `valid` bit is **never read to gate a capability use**:
  it is written on node creation, asserted in `_cap_rev_tree_dup_node_before`, set
  false in `revoke`, and read nowhere else in `target/riscv`. The capability-use
  paths do not consult it — `_helper_access_with_cap` (`op_helper.c:943`) authorizes
  a load/store purely on the base register's cached `tag` bit plus a `cap_in_bounds`
  check, and the cap-load path `helper_reg_set_cap_compressed` (`op_helper.c:999`)
  derives the result tag from `cap_mem_map_query` (is a cap stored here?), not from
  rev-tree validity. `helper_csrevoke` also does not touch `cm_map` or scrub
  in-flight register/memory tags. **Therefore a revoked capability already
  materialized in a register or memory remains fully usable**, which is exactly the
  observed use-after-revoke. (This supersedes the earlier ambient-root/scalar
  hypothesis: authority demonstrably comes from the register cap's tag+bounds; the
  defect is that revocation is not checked on that path.)

  **Consequence for the direction.** The borrow/revoke *plumbing* works
  (mint-revocation, the REV/LIN/UNINIT lifecycle, the kernel-module borrow ABI),
  but the security *guarantee* does not hold in this QEMU build because enforcement
  is missing. Making revocation bite requires wiring enforcement into the
  capability-use path — either **lazy** (the access/load path checks the cap's
  rev-tree node `valid` bit; matches why the tree exists, adds a per-access lookup)
  or **eager** (`csrevoke` scrubs tags in registers + `cm_map`). This is the gating
  prerequisite for M1+, and it is a **QEMU model** change, not a compiler change.
- **M1 — one SQLite boundary group, end-to-end.** Take the single hottest group —
  `sqlite3_column_text` (engine→host borrow) — and express it as: engine `mrev`s a
  revocation cap for the row buffer, lends the linear cap to the host shim, and
  `revoke`s on the next `step`. Show a cached pointer (the diesel/rusqlite bug,
  Table 3 rows 2–3) faults.
- **M2 — reverse direction + callbacks.** `bind`/`result` (host→engine) and the
  callback/sealed path (the gh-142830 re-entrant free, row 1) via sealed caps.
- **M3 — measurement.** Borrow path vs `SQLITE_TRANSIENT` copy baseline
  (instruction count / cycles / memory traffic), the proposal's deliverable (2).

## Open risks / decisions (review these)

1. **Fault delivery is an abort, not a trappable fault (the safe-fail gap).** The
   `helper_csmrev` violation and post-revoke derefs currently surface as a QEMU
   **assertion / process abort** (`riscv_cpu_do_interrupt` assert; the authority
   runner already treats this abort as the terminal trap signal). The proposal's
   whole value is "safe-fail: a contract violation becomes a *deterministic
   trappable fault*, not a crash." Converting the abort into a guest-deliverable
   exception is likely the single most important systems prerequisite. (Already on
   the audit's lower-priority list as "architectural fault delivery.")
2. **Linear caps through the compiler.** LIN is move-only; it must survive the
   calling convention, spills/reloads, and PHIs without being copied (which would
   either fault or silently demote). The C2 provenance work is the natural checker
   for "a linear/borrowed cap is never duplicated or demoted before its required
   use" — provenance + linearity are the same dataflow question. We have **not**
   yet shown a LIN cap surviving a non-trivial function body; M1 will stress this.
3. **Origin of LIN authority is the runtime, not codegen.** Because the compiler
   cannot mint LIN from NONLIN, the lender side (the trusted host/runtime) is where
   `init`/`mrev` happen. The "binding co-designed with the engine" (deliverable 1)
   is therefore partly a *runtime/SBI* design, not purely a compiler feature. This
   is consistent with the shared-region mechanism but should be stated plainly.
4. **`SPLIT` / root-elimination is unwired.** If the paper leans on the audit's
   "root-elimination" framing, `SPLIT` needs an intrinsic+builtin first (helper
   exists). Not required for the borrow/revoke marshalling story; required for the
   stronger "ordinary code cannot bypass the broad root" claim.
5. **`NONLIN` fallback for shared reads.** The proposal concedes patterns needing
   concurrent shared reads must fall back to NONLIN (forfeiting clean revocation).
   Worth deciding which SQLite paths (if any) need this before claiming coverage.

## Pointers

- Proposal: `../history/SqliteProposal.{pdf,tex}`.
- Primitives in QEMU: `capstone-qemu/target/riscv/op_helper.c` (helpers),
  `cap.h` (type model), `insn_trans/trans_capstone.c.inc` (decode).
- Toolchain: `clang/lib/CodeGen/TargetBuiltins/Capstone.cpp` (builtins),
  `llvm/include/llvm/IR/IntrinsicsCapstone.td`, `CapstoneISelDAGToDAG.cpp`.
- Working borrow/revoke: `tests/runtime-qemu/shared-region-probe/`,
  `run-hostcall-second-pending-payload-revoke-probe.sh`, runtime-qemu `README.md`.
- SQLite scaffolding: `tests/runtime-qemu/sqlite-vfs-skeleton/`.
- Related: `c2-provenance-verifier-proposal.md` (linearity ≈ provenance dataflow),
  `capability-bounds-model.md` (the granularity/C1 supporting result).
