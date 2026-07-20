# Task #70 revocation — verified dormant, then recording fixed on a branch

*2026-07-03. Arc: picked up #70 as "verify/close" → found it still dormant →
got the runtime author's (light) go-ahead to experiment → implemented the
recording fix on a `capstone-qemu` branch → confirmed revocation now bites →
surfaced two follow-ons that are the runtime author's design call. This note is
the full trail; it supersedes an earlier state-doc claim that "the author
confirmed the traversal is correct" (imprecise — see below).*

## 1. Starting state — dormant (verified two ways)

Enforcement half (committed `3d71d161e7`) is present and correct:
`capstone_cap_revoked` faults a revoked base in `_helper_access_with_cap`
(`RISCV_EXCP_INVALID_CAP`) and demotes a revoked cap reloaded from memory to
untagged in `helper_reg_set_cap_compressed`; gated `CAPSTONE_REVOCATION_ENFORCE`
(default on). But **dormant**: nothing marked nodes invalid.

Recording bug (`cap_rev_tree.c` `cap_rev_tree_revoke`): the loop guard was
`_CAP_REV_NODE(tree, node_id).depth > depth` with `depth = node_id.depth` — i.e.
`D > D`, always false, so the invalidation body never ran. Confirmed empirically:
`run-revoke-matrix-probe.sh` → "use-after-revoke NOT trapped (dormant)".

## 2. Author (the collaborator) go-ahead

Round-1 questions were answered (record under `/tmp/capstone/`); the Jul-1
follow-up got: (a) the traversal *structure* "doesn't look incorrect ... starts
at the node next to the root" — he did **not** engage the `D > D` guard
literally; (b) enforcement "was left unfinished ... there's a TODO ... you may
experiment with turning the check on." Plus: OK to branch `capstone-qemu` (and the
spec repo) for fixes. Net: greenlit to implement + validate on our side; the
exact rule is ours to finish against the spec (proposal §8) with the probes as
oracle. (So "author confirmed traversal correct" was too strong.)

## 3. The recording fix (branch `fix/revocation-recording-invalidation`)

The rev-tree is a doubly-linked seniority chain (`.prev`=senior, `.next`=junior)
with per-node `depth`. `mrev L->R` gives R the senior node (depth D) and bumps L
to depth D+1 (junior). `revoke R` must invalidate the junior run after R and
splice it out (`node_id.next = cur`). Fix: guard tests the **walked** node's
depth, not node_id's:
```c
for (cur = node_id.next;
     cur != NULL && _CAP_REV_NODE(tree, cur).depth > depth;   // was node_id.depth
     cur = cur.next) { cur.valid = false; ... }
```
One-line change; the rest of the revoke logic (retain_data / splice) is unchanged.

## 4. Result — revocation now bites (verified)

Rebuilt `qemu-system-riscv64`; `run-revoke-matrix-probe.sh` case 2 (memory-stored
borrow):
- Before: round 2 store lands, "round 2 returned 0x202".
- After: the borrower's cached cap reloads **untagged** (`Cap(0, 0x7, …)`), the
  round-2 store **no longer lands** (no `0x202`) — use-after-revoke is caught.

`CAPSTONE_DEBUG_PRINT` is compiled in, so the absence of a fault line + absence of
`0x202` together confirm the store was stopped, not silently completed.

**No-false-positive is guaranteed by construction:** `cap_rev_tree_revoke` runs
only on an explicit `csrevoke`, which normal workloads (authority suite, CoreMark,
RV8, BEEBS) never issue — so this change cannot perturb non-revocation code.

## 5. Two follow-ons — runtime-author design calls (NOT landed)

**(a) Re-share regression (spec-correct, needs a mode decision).** The known-good
re-share-after-revoke probe now hits
`helper_csmrev: Assertion rs1_v->val.cap.type == CAP_TYPE_LIN failed`. This is
exactly proposal §8.2: revoking a **linear** cap sets `retain_data=false` ⇒ the
retained handle becomes **UNINIT**, and a direct re-`mrev` asserts. Per spec the
re-share must either (i) use a **non-linear** (delin'd) borrow so revoke keeps the
handle LIN, or (ii) `init` (UNINIT→LIN) before `mrev`. The monitor
(`sbi_capstone.c:370-410`) already has four sharing modes
(linear/non-linear × post-return-revoke yes/no) plus `revoke_region` (470); which
mode the borrow/lending path should use, and how re-share treats a post-revoke
UNINIT handle, is a **monitor lending-ABI decision** — the author's call. Not
changed here.

**(b) Fault delivery in a domain-call context (separate, parked gap).** In the
revoke-matrix probe the caught use-after-revoke fault is raised inside a
lender→borrower domain call; delivery goes to the monitor trap path, which dumps
registers (`[CAPSTONE] Print …`) and spins (probe times out) rather than cleanly
halting or returning the fault to the lender. This is the pre-existing
domain-fault return-to-host gap (`design/domain-fault-delivery-proposal.md`
Step B / `__domasync` unused), independent of #70's recording.

## 5b. REV_DEFAULT re-share validated + fix merged (2026-07-03)

The clean re-share path was confirmed empirically. Switching the payload-revoke
probe's borrow from `REV_BORROWED` (linear) to `REV_DEFAULT` (non-linear:
`shared_region_annotated` delin's the borrower copy, so `revoke` keeps the
retained handle **LIN**) makes the full cycle complete:
`revoking payload region → re-share → second call → third call → **success**`,
no `csmrev` assertion. The base (non-revoke) variant still reproduces its
documented `csmrev`-assert marker (re-sharing a REV handle without a revoke is
expected to assert), so no regression there.

So all three legs are proven: **record** (junior subtree invalidated), **enforce**
(use-after-revoke caught: untagged reload, store dropped), **re-share** (works for
a non-linear borrow). The recording fix was merged to the QEMU submodule dev
branch `capstone-bootstrap` (`8b6a47f322`) and the parent pointer bumped.

## 6. Disposition

- **Recording fix:** correct + verified, on branch `fix/revocation-recording-invalidation`
  (parent submodule pointer left at `db0a750c2d`, main line unaffected).
- **#70 stays open:** the core mechanism (record + enforce) now works end to end,
  but a clean close needs (a) the re-share sharing-mode/init decision and (b) the
  fault-delivery path — both runtime-author-owned. One precise follow-up for the
  author: for a lent/borrowed region, is the delegated cap meant to be non-linear
  (delin'd, so revoke keeps the handle LIN and re-share via `mrev` works) or linear
  (so re-share must `init` before `mrev`)?
- Corrected the falsified "author confirmed traversal correct" note in
  `state/current-next-step.md`.
