# Agent-B setup — second Claude Code agent on the `-b` clone

Complete, copy-pasteable setup for bringing up a **second** Claude Code agent
(different Claude account) on the second clone, fully isolated from Agent-A.
Read `MULTI-AGENT-WORKFLOW.md` first for the *why*; this file is the *how*.

Assumptions (already done by you):
```
git clone https://github.com/project-starch/llvm-capstone /home/alexey/dev/llvm-capstone-b
cd /home/alexey/dev/llvm-capstone-b
git checkout -b capstone-bootstrap-b origin/capstone-bootstrap
git submodule update --init --recursive
```
Paths below assume Agent-A = `/home/alexey/dev/llvm-capstone`,
Agent-B = `/home/alexey/dev/llvm-capstone-b`.

---

## 1. A separate Claude account without clobbering Agent-A's login

Claude Code keeps **all** per-user state — auth credentials, settings, and the
project **memory** store — under one config directory (`~/.claude` by default,
containing `.credentials.json`). Two accounts on one Linux user therefore need
**two config directories**, selected with the `CLAUDE_CONFIG_DIR` environment
variable. This also gives Agent-B its own memory store automatically (memory is
keyed by config dir + project path), which is exactly what we want — the two
agents' memories stay separate; durable cross-agent facts go in committed
`agent-handoff/` docs instead.

```bash
# One-time: create B's config dir and log in with the SECOND account.
mkdir -p ~/.claude-b
CLAUDE_CONFIG_DIR=~/.claude-b claude          # then /login with account B, or:
CLAUDE_CONFIG_DIR=~/.claude-b claude login    # follow the browser/device flow
```

Agent-A's `~/.claude` is never touched. Verify the two are independent:
```bash
ls ~/.claude/.credentials.json ~/.claude-b/.credentials.json   # two distinct files
```

> Fallback if `CLAUDE_CONFIG_DIR` ever misbehaves: run Agent-B as a **separate
> Linux user** (`sudo useradd -m claude-b`), which gives a naturally separate
> `~/.claude`. Heavier, but bulletproof. The `CLAUDE_CONFIG_DIR` route is simpler
> and recommended.

---

## 2. Launcher script + alias (so you never forget the config dir)

Save as `~/.local/bin/claude-b` and `chmod +x`:
```bash
#!/usr/bin/env bash
# Launch the second Claude Code agent: its own account + its own clone.
export CLAUDE_CONFIG_DIR="$HOME/.claude-b"
export CAPSTONE_TMP_ROOT="/tmp/capstone-b"     # keep B's scratch off A's (see §4)
cd /home/alexey/dev/llvm-capstone-b || exit 1
exec claude "$@"
```
Then just run `claude-b`. (Optionally add `alias claude-a='cd /home/alexey/dev/llvm-capstone && claude'` for symmetry.)

---

## 3. Environment inside the B clone

The test env is **path-relative** — sourcing it from the B clone points every
`CAPSTONE_*` path at the B clone automatically (its own `clang`, its own
`rootfs.ext2`, its own QEMU). No edits needed:
```bash
cd /home/alexey/dev/llvm-capstone-b
source capstone/tests/capstone-test-env.sh
echo "$CAPSTONE_REPO_ROOT"     # => /home/alexey/dev/llvm-capstone-b
echo "$CAPSTONE_CLANG"         # => .../llvm-capstone-b/llvm/cmake-build-debug/bin/clang
```

Read-first set (same as A, from CLAUDE.md):
- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/state/current-state.md` → **but see §5 on per-agent state**
- `capstone/agent-handoff/MULTI-AGENT-WORKFLOW.md`
- `capstone/agent-handoff/COORDINATION.md`  (who owns what right now)

---

## 4. The `/tmp/capstone` collision — override it for B

`capstone-test-env.sh` defaults `CAPSTONE_TMP_ROOT=/tmp/capstone`, and CLAUDE.md
routes manager-facing summaries there. **Both clones would write the same dir.**
The launcher in §2 already exports `CAPSTONE_TMP_ROOT=/tmp/capstone-b`; if you
run B without the launcher, set it yourself:
```bash
export CAPSTONE_TMP_ROOT=/tmp/capstone-b
```
Claude Code's own scratchpad is session-keyed and already isolated — no action.

---

## 5. Build state — B starts with an empty 10 GB build dir

`llvm/cmake-build-debug` is gitignored, so the B clone has **no compiler yet**.
Two options:

- **B does compiler/codegen or firmware work → build in the B clone** (the safe,
  isolated path). Rebuild LLVM and buildroot in `-b` exactly as documented for A.
  Costs time + ~10 GB, but keeps artifacts and the 2 GB `rootfs.ext2` independent
  so both agents can build/test in parallel.
- **B does pure docs/analysis, no compiling → skip the build.** Optionally
  *read-only* share A's clang so smoke checks work:
  ```bash
  # From the B clone; symlink A's prebuilt build dir (READ-ONLY use only).
  ln -s /home/alexey/dev/llvm-capstone/llvm/cmake-build-debug \
        /home/alexey/dev/llvm-capstone-b/llvm/cmake-build-debug
  ```
  **Hazard:** if A later *rebuilds* clang, B sees the in-progress binary. Only do
  this if B never triggers an LLVM rebuild. When in doubt, build B's own.

> Never run B's QEMU matrix against A's `rootfs.ext2`. With separate clones this
> is automatic — just don't point `CAPSTONE_*` at the A tree.

---

## 6. Branch, submodules, shared files (summary — full rules in MULTI-AGENT-WORKFLOW.md)

- B commits on **`capstone-bootstrap-b`** only. Integrate by merging B→A at
  checkpoints; never co-commit to one branch.
- **One owning agent per submodule** to avoid gitlink-SHA conflicts. Check
  `COORDINATION.md` before bumping any submodule; log the bump after.
- **`state/current-state.md` / `current-next-step.md` are single-writer.** B keeps
  its own: `state/current-state.B.md`, `state/current-next-step.B.md` (A keeps the
  base files or renames to `.A.md`). Do not both write the base files.
- `history/` notes are timestamped/append-only — safe to share; prefix the slug
  with `agentB-` if two could land in the same second.

---

## 7. First-run verification checklist (run inside `claude-b`)

```bash
source capstone/tests/capstone-test-env.sh
[ "$CAPSTONE_REPO_ROOT" = /home/alexey/dev/llvm-capstone-b ] && echo "repo OK"
[ "$CAPSTONE_TMP_ROOT" = /tmp/capstone-b ] && echo "tmp OK"
git branch --show-current            # => capstone-bootstrap-b
git config --get remote.origin.url   # => .../project-starch/llvm-capstone
ls ~/.claude-b/.credentials.json     # B's own login exists
```
All five green ⇒ Agent-B is isolated and ready. Then pick up its task from
`COORDINATION.md` and its own `state/current-state.B.md`.
