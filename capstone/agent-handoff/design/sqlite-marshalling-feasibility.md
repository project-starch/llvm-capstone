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
exists in this tree, and what is the real gap?**" The answer is: substantially
more is wired than expected, and the central borrow→revoke→fault behaviour already
works through one specific channel.

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

## Ground truth 3 — borrow/revoke already works via HostCall shared regions

This is the load-bearing discovery. The host↔domain boundary the proposal needs
*already exists* as the **HostCall shared-region** mechanism, and the
borrow→revoke contract is already demonstrated:

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

That rule **is** the proposal's revocation contract (revoke at the borrow's end
before the buffer is reused). So the engine→host "borrow result, revoke at next
`step`/`reset`/`finalize`" pattern (Table 4 rows 1–8) maps directly onto an
already-working primitive: shared-region borrow + `revoke_region`.

## Ground truth 4 — existing SQLite scaffolding

- `tests/runtime-qemu/sqlite-vfs-skeleton/` + `build-sqlite-vfs-skeleton.sh`
  already build a Capstone-compiled SQLite-facing VFS skeleton domain against the
  **official SQLite 3.53.1 amalgamation** (auto-fetched from sqlite.org), at `-O2`.
  (The in-progress benchmark bring-up should reconcile with this; see Open items.)

## What this re-frames in the milestones

Because borrow/revoke already works, **M0 is not "can we revoke" (done) — it is
"map the SQLite boundary onto the shared-region API and make the failure mode
safe-fail."** Revised staging:

- **M0 — characterize + harden the existing borrow/revoke.** Lift the
  shared-region revoke probe into a named, oracle-classified test (the way the
  authority suite classifies traps): borrow→revoke→use-after-revoke must produce a
  deterministic **tag fault**, and a senior/hierarchical revoke must cascade to a
  derived sub-region. This is the spike; it reuses existing assets.
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
