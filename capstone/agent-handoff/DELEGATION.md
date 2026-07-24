# Delegating across agent lanes (A ↔ B)

Durable guide for how the two peer lanes split work. Read alongside `CLAUDE.md`
(canonical hard constraints) and `README.md`.

## Model
Two **peer Opus lanes**, equally capable, each starting a session cold (repo +
`CLAUDE.md` + memory only):
- **A (main lane)** → commits to `capstone-bootstrap`.
- **B lane** → commits to `capstone-bootstrap-b`.

B is **not a subagent** — it's a peer session the user runs. The `CLAUDE.md`
"never delegate to subagents" list refers to the built-in Explore / corpus-runner /
general-purpose subagents, **not** to B. Delegating to B is a peer hand-off.

## When to delegate (timing matters as much as task choice)
- **Hot context is the scarce asset.** The lane holding hot context on X should
  *keep* X and delegate work whose cost is largely *independent* of that context —
  otherwise the peer pays a big cold-start tax re-deriving what you already hold.
- **Don't delegate reflexively** the moment a lane frees up. Proceed on your
  critical path while hot; delegate at a **natural pause** — a genuine block, or the
  start of a long build/validate cycle — when the handoff write-up is cheapest and
  best-informed.
- A good delegated task is **(1) independent** (minimal file overlap / coordination),
  **(2) token-heavy** (grindy, iterative — the kind that would burn the busy lane's
  budget), and **(3) crisply bounded** (one clear deliverable + how to validate it).

## What to delegate vs. keep in-lane
- **Delegate:** independent breadth work, harness/build/test tooling, packaging,
  measurement — anything disjoint from the other lane's active files.
- **Keep in the owning lane (don't split across lanes):** the compiler/codegen +
  capability-ABI (splitting risks ABI incoherence + merge conflicts on the Capstone
  target files), all board/FPGA sessions (serialized, secret token, human-in-loop),
  the paper (single-author coherence), and anything touching real-person names.

## How to write the handoff (high autonomy)
The peer is as capable as you. Hand off the **goal + guardrails, not the steps** —
be free on method.
- **Guardrails = the cold-start context the peer can't re-derive:** the permanent
  rules (below), the exact build config, the deliverable + validation, **what's
  already solved** (so they don't redo it), and any file-overlap coordination.
- Make the task doc **self-contained** — the peer may read only it. Enumerate the
  permanent rules in-doc, point to `CLAUDE.md` + this file as canonical, and give
  the exact start command (branch + `git merge origin/capstone-bootstrap`).
- Suggest an approach, but say it's a suggestion; flag any path that would touch the
  other lane's files as "coordinate first."

## Branch hygiene
- A → `capstone-bootstrap`, B → `capstone-bootstrap-b`, both off the shared
  mainline. Sync by `git merge origin/capstone-bootstrap` — **never rewrite the
  other lane's pushed history.**
- Fold a finished side-branch back to mainline via **fast-forward** when it's a
  linear superset and gated/corpus-safe (flag-off codegen byte-identical, or the
  change corpus-validated). Retire the side branch after folding.

## Context management (compaction) — both lanes
Board-debug threads run long, so every lane manages context deliberately (canonical:
`CLAUDE.md` "Context & compaction").
- **Assess lightly** (not every turn) whether it's a safe moment to `/compact`, and
  **recommend it in one line** at a natural checkpoint **once the important state is
  already captured** in committed docs/memory. Do **not** recommend it mid-task,
  during active debugging, or while un-captured details still matter. Say briefly why
  the timing is safe (or why to postpone).
- **Never compact unilaterally** — you can only recommend; it's the user's call.
- When you recommend `/compact`, add a short **compaction brief**: what to **keep
  verbatim** (current task + exact next step, un-committed decisions/rationale, open
  blockers, live file paths/values in flight) vs. what's safe to **compress**
  (resolved sub-threads, tool-output noise, superseded approaches, anything already
  in docs/memory). The generic summarizer can't tell what's load-bearing — tell it.

## Permanent repository rules — every lane adopts these as its own
(Canonical source: `CLAUDE.md` "Hard constraints" + the memory feedback files.)
1. **Never mention any real person by name — anywhere.** PI / supervisor / board
   owner / collaborator → neutral roles ("the board owner", "the PI"). All
   committed/shared content. Permanent and absolute. (Upstream `lldb/`, `llvm/`
   files are not ours — leave their names alone.)
2. **No `Co-Authored-By:` lines**; no worker/agent identity in commit messages;
   **don't rewrite pushed history**; **commit only when the user asks.**
3. **Never commit debug/report/session-note files** (`*_DEBUG_CHECKPOINT.md`, etc.).
4. **Manager/PI-facing summaries → `/tmp/capstone/`**, never the repo.
5. **Serialize the QEMU suites** — the shared `rootfs.ext2` write-lock means never
   two matrix/QEMU runs in parallel (coordinate across lanes).
6. **`ninja -j90`** (~80% of 112 cores), **never `-j112`** (a parallel debug-link
   storm hangs the whole box — no SSH).
7. **No commits into submodule *source*** (`capstone-qemu` / opensbi / buildroot /
   system); submodule source stays uncommitted.
8. **Bug-fix / root-cause / audit notes → `history/`** (dated
   `DD-MM-YYYY_HH-MM-SS_name.md`), **not `design/`** (`design/` = architecture
   decisions only). Active plans → `plans/` (committed, portable).
9. Board/FPGA sessions, compiler/codegen + capability-ABI changes, subtle-correctness
   debugging, and the paper stay in the owning lane; never delegate them to subagents.
