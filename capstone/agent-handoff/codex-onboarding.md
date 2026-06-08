# Onboarding for OpenAI Codex (and similar coding agents)

Unlike Claude Code, Codex does not automatically read `CLAUDE.md` or any project-specific
configuration file. You must provide context explicitly at the start of each session.

## Steps to start a Codex session on this project

1. **Clone the repo** (if not already done):
   ```bash
   git clone -b capstone-bootstrap https://github.com/project-starch/llvm-capstone
   cd llvm-capstone
   git submodule update --init
   ```
   Sub-repositories (`caplifive-buildroot`, `capstone-qemu`, `capstone-c`, `capstone-spec`)
   are git submodules. Do not clone them manually.

2. **Open a Codex session** in the repo root.

3. **Paste the prompt template below** as the first message (the system/context prompt).

4. **Then describe your specific task** in a follow-up message.

## What to watch for

- If Codex tries to clone sub-repositories manually, correct it: they are git submodules,
  populated with `git submodule update --init`.
- If Codex proposes adding `Co-Authored-By:` lines to commits, instruct it not to.
- If Codex generates a `*_DEBUG_CHECKPOINT.md` or session notes file, do not commit it.

---

## Prompt template (paste verbatim as your first message)

```
I am working on the Capstone architecture LLVM fork.

Repository: https://github.com/project-starch/llvm-capstone (branch: capstone-bootstrap)

Sub-repositories are git submodules. Populate with:
  git submodule update --init
Do NOT clone them individually.

Read these files first before proposing any changes:
  capstone/agent-handoff/README.md
  capstone/agent-handoff/state/current-state.md
  capstone/agent-handoff/state/current-next-step.md

Working constraints:
1. Prefer the smallest meaningful next step toward the real goal.
2. Re-test every completed step at the affected layer.
3. Never add "Co-Authored-By" lines to commits.
4. Never commit *_DEBUG_CHECKPOINT.md, session notes, or colleague conversation files.
5. After a validated change: provide exact `git add` / `git commit` commands,
   with a short subject line + detailed body.
6. Keep `capstone/agent-handoff/` current when the validated baseline or workflow changes.
7. Active plans live in `capstone/agent-handoff/plans/` — nowhere else.
8. Manager-facing summaries go under /tmp/capstone/, not into the repo.

Build and test entry points (after: source capstone/tests/capstone-test-env.sh):
  bash capstone/tests/runtime-qemu/run-smoke.sh          # quick sanity check
  bash capstone/tests/runtime-qemu/run-coremark.sh       # CoreMark validation
  "$CAPSTONE_LLVM_LIT" -sv llvm/test/CodeGen/Capstone    # backend CodeGen tests

Current milestone: add the second BEEBS benchmark (`insertsort`) after the validated `fac` path.
Plan: capstone/agent-handoff/plans/benchmark-bringup.md

Backend workarounds (stable, do not remove or override):
  capstone/agent-handoff/plans/backend-compiler-fixes.md

Deeper reference files (load only when the task needs them) in folders:
  capstone/agent-handoff/ref/
  capstone/agent-handoff/design/
```
