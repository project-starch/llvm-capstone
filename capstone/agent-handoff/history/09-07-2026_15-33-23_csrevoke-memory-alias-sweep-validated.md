# `csrevoke` memory-alias sweep — VALIDATED (outcome a)

**Date:** 2026-07-09
**Lane:** compiler/codegen + emulator
**Task:** `capstone/agent-handoff/tasks/agentB-003-validate-revoke-alias-sweep.md`
**Result:** **Outcome (a)** — the memory-alias sweep already works. No code change.

## The one question

Does `csrevoke` invalidate **memory-resident** copies of a revoked capability,
not just the register operand? The BORROW-REVOKE corpus rows (3/13/18/19) want a
literal single-domain "after" where a **cached pointer held in memory** faults on
use after revoke — so this property is load-bearing, and it gates whether A's
`start.S` linear-authority firmware cycle is worth spending.

**Answer: YES.** A memory-resident alias of a revoked cap faults on the next
dereference.

## Mechanism (code reading)

Revocation in this emulator is **lazy, keyed on a shared revocation-tree node**,
not an eager register/memory scan:

- Every cap carries a `rev_node_id` into `env->cr_tree.node_pool`
  (`cap_rev_tree.h`). A node has a `valid` bit.
- `helper_csmrev` (`op_helper.c`) requires a `CAP_TYPE_LIN` input and, via
  `cap_rev_tree_mrev`, inserts a revocation node **senior** to the linear node
  (depth-ordered version chain: `_cap_rev_tree_dup_node_before` + bump the
  linear node's depth). The returned `CAP_TYPE_REV` handle is the revoker.
- `helper_csrevoke` → `cap_rev_tree_revoke` walks the **junior run** (nodes
  deeper than the REV node) and sets each `valid = false`. All copies of the
  revoked cap share that (now invalid) node.
- `capstone_cap_revoked(env, cap)` (`op_helper.c:999`) checks
  `!cap_rev_tree_check_valid(tree, cap->rev_node_id)` on **every** dereference.

The memory coverage hinges on `rev_node_id` surviving a store→load, and it does:
`cap_compress` packs `revnode_id` into **bits 33-63** of the compressed 128-bit
word (`cap_compress.c` `DEF_OTHER_FIELD(revnode_id, 33, 63)`), and
`cap_uncompress` restores it (`out->rev_node_id = revnode_id_get(...)`). On
`ldc`, `helper_reg_set_cap_compressed` (`op_helper.c:1070`) reconstructs the cap
and, if `capstone_cap_revoked` is true, **clears the tag** — so the reloaded
memory alias is untagged and its next deref raises
`RISCV_EXCP_UNEXP_OP_TYPE` (**cause 24**, "Cap mem access requires capability").
A register-resident alias (no reload) instead trips the revoked check at
`op_helper.c:1022` (`RISCV_EXCP_INVALID_CAP`). Either way: a clean capability
fault.

The LLVM backend already spells the intent: `REVOKE` in `CapstoneInstrInfo.td` is
commented *"Revoke capability globally (Memory Sweep)"* with `mayLoad=1,
mayStore=1, isBarrier=1`.

## The experiment (firmware-free, in-lane)

`csmrev` needs a `CAP_TYPE_LIN` input, which a domain cannot mint (that is A's
gated `start.S` linear authority). To validate the sweep **without** firmware, I
hand-mint a linear cap with the QEMU debug op **`csdebuggencap`** (Func7=`1000000`
Func3=`001` opcode=`1011011`; `helper_csdebuggencap` sets `type=CAP_TYPE_LIN`,
RWX, a fresh lone rev-node, `tag=1`). It has no assembler mnemonic, so it is
emitted as a raw R-type via `.insn r 0x5b, 0x1, 0x40, rd, rs1, rs2`. `MREV`/
`REVOKE` use the real clang builtins `__builtin_capstone_cap_mrev` /
`__builtin_capstone_cap_revoke`. This is the exact analogue of the task-002
`csdrop` test that hand-minted an ordinary cap.

Four probe domains (built `-O0`, run under `qemu-system-riscv64` via the
`runtime-qemu` domain harness):

| Probe | Flow | Expected | Observed |
|---|---|---|---|
| `revoke_mem_alias` **(key)** | mint L → MREV → store L to `.bss` slot → write (ok) → REVOKE → **reload slot** → deref | fault | `Cap mem access requires capability`, **cause 24**, no retval |
| `revoke_reg_alias` | mint L → MREV → write (ok) → REVOKE → deref L | fault | cause 24, no retval |
| `revoke_unrelated_ok` | mint L + L2 → MREV L → REVOKE → deref **L2** | ok | retval `571670579` = `0x22130033` |
| `revoke_mem_control` | mint L → store to slot → reload → deref (**no REVOKE**) | ok | retval `571736158` = `0x2214005E` |

Reading across the four: the memory-resident alias faults **only** after REVOKE
(control passes), the register alias faults too, and an unrelated cap is
untouched (no over-broad sweep). Both fault probes show cause 24 because at `-O0`
each alias round-trips memory and is untagged on reload — i.e. the
reload-time-untag path is exactly what fires for memory aliases.

## Reconciling the earlier R-probe "NO-TRAP-GAP"

`runtime-qemu/borrow-revoke-uaf-probe` (2026-06-29) reported use-after-revoke
**not** trapping. That is a **different, orthogonal** issue: the borrower obtained
the region through an independent SBI `REGION_QUERY` mapping (an ambient/NONLIN
mapping that is **not a tracked descendant** of the lender's revocable cap), so
its `rev_node_id` was never in the revoked subtree. The QEMU sweep is correct;
the gap is provenance/plumbing in the monitor path (the borrower must reach the
region **only** through the delegated, tracked, linear cap). This experiment
isolates the clean case — genuine tracked aliases — and shows the sweep has teeth.

## Bottom line for A / the PI

- The QEMU **revoke vehicle is proven**: register *and* memory aliases of a
  revocable capability fault after `csrevoke`. The literal single-domain
  BORROW-REVOKE "after" (cached in-memory pointer faults post-revoke) is viable
  **at the emulator layer**.
- The **only remaining gate** for rows 3/13/18/19 is A's intra-domain linear
  authority (`start.S`/firmware) — and it must deliver the region to the domain
  **through the tracked linear cap**, not an independent SBI-query mapping (the
  R-probe's lesson), or the sweep will miss it exactly as observed there.
- **No `capstone-qemu` change was needed** (outcome a) — hence no gitlink bump.

## Test artifacts

The four probe sources + `csrevoke_probe.h` live in the session scratchpad (not
committed — they need A's `runtime-qemu` domain harness under `capstone/tests`,
outside this task's `capstone-qemu`-only scope). They can be promoted to
`capstone/tests/capstone-authority/domains/` as `revoke_*` probes with oracle
lines (`revoke_mem_alias`/`revoke_reg_alias` → bounds/tag-fault,
`revoke_unrelated_ok`/`revoke_mem_control` → ok), analogous to the `subobjfield_*`
and (proposed) `csdrop_*` sets — a cross-lane additive change to coordinate with A.
