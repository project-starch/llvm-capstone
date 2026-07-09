# COORDINATION — live ownership between the two agents

Single source of truth for **who owns what right now**. Update this file
*before* you start on a subsystem/submodule and *when you hand off*. Read it at
the start of every session. Rules behind it: `MULTI-AGENT-WORKFLOW.md`.
Setup for the second agent: `AGENT-B-SETUP.md`.

- **Agent-A** — clone `/home/alexey/dev/llvm-capstone`, branch `capstone-bootstrap`.
- **Agent-B** — clone `/home/alexey/dev/llvm-capstone-b`, branch `capstone-bootstrap-b`.

---

## Active ownership  _(edit before you start / on handoff)_

| Owner | Subsystem | Submodules / dirs held | Branch |
|---|---|---|---|
| Agent-A | (current: SQLite Stage-2 corpus, firmware/monitor) | `caplifive-buildroot` → `opensbi`, `capstone-sbi-domain/capstone-sbi`; `capstone/capstone-c`; `capstone/benchmarks/sqlite`; `capstone/tests` | `capstone-bootstrap` |
| Agent-B | compiler/codegen + emulator (C1 subobject-bounds proposal, design-only) | `llvm/` (in-tree) + `capstone/capstone-qemu` | `capstone-bootstrap-b` |

Suggested non-overlapping split (pick when B's task is set):
- Compiler/codegen (`llvm/`, in-tree) + `capstone/capstone-qemu` — one owner.
- Firmware/monitor (nested `caplifive-buildroot`/`opensbi`/`capstone-sbi`) + `capstone-c` — the other.
- `capstone/paper` — whoever is writing; low build-contention, coordinate edits here.

## Current position  _(update at EACH checkpoint — one line per agent; makes takeover read-and-go)_

One sentence each: what you're mid-doing, where, `branch@sha`, tested?, any
uncommitted WIP. This is the field the surviving agent reads to take over a lane
if the other hits a usage limit. Keep it honest and current.

| Agent | Current position |
|---|---|
| Agent-A | idle at `capstone-bootstrap` tip; no in-flight task; nothing uncommitted. |
| Agent-B | CHECKPOINT (2026-07-09): `csdrop` (DROP) implemented in `capstone-qemu` — the LINEAR / Stage-2 row-11 QEMU unblock. Submodule `capstone-qemu` bumped `cf541a1f`→`2e6a67d1` (branch `capstone-bootstrap-b`); superproject gitlink bumped on `capstone-bootstrap-b`. Pieces (all additive, in-lane): decode entry (Func3=001/Func7=0001011), `DEF_HELPER_2(csdrop)`, `trans_csdrop` (rs1-only, like `trans_csrevoke`), `helper_csdrop` (spec-faithful, type-agnostic — clears rs1 tag → next deref raises cause-24 UNEXP_OP_TYPE). Rebuilt `qemu-system-riscv64` (`CC=/usr/bin/gcc`, `-j 80`). VALIDATED under QEMU: control `csdrop_live` → ok (retval 0x220F005E); fault `csdrop_use_after` → read live cap ok, drop, re-read → "Cap mem access requires capability" cause=24, no retval (consume→use→clean fault, NOT illegal instruction). No regressions: canonical smoke + borrow-revoke UAF (R) both pass. Full note: `history/09-07-2026_13-28-31_csdrop-implemented-row11-qemu-unblock.md`. NOT pushed yet (awaiting go-ahead; superproject branch + submodule commit both need pushing for A to integrate). Row 11 full before→after domain demo still GATED ON A: needs intra-domain linear authority (sign-off-gated `start.S`/firmware, Blocker 1). Test probe sources in session scratchpad (not committed — would be a cross-lane add to A's `capstone/tests`). Prior C1 v1 (arrays-only subobject bounds) merged to canonical by A (ff to `c4758de`). PREVIOUS: IMPLEMENTED C1 subobject-bounds v1 (sign-off received). UNCOMMITTED WIP on disk in `capstone-bootstrap-b` (proposal `da765ce` committed): clang flag `-fcapstone-subobject-bounds` (LangOptions.def, Options.td), CGExpr.cpp `maybeNarrowSubobjectBounds` hook (v1=array fields only + union/FAM/last-array/incomplete refusals), lit `clang/test/CodeGen/capstone-subobject-bounds.c`, authority probes `capstone/tests/capstone-authority/domains/subobjfield_{overrun,inbounds,union_active,flexarray}.c` + oracle + build-script case. Reconfigured B's cmake tree cleanly per PI recipe: host compiler `/usr/bin/clang;/usr/bin/clang++` (NEVER a capstone-built clang), Debug/shared, `X86;RISCV;Capstone`, `clang;lld`, opt-tablegen, bindings off. Build capped at `-j 80` (~71% of 112 cores — never all CPUs) DONE (exit 0). COMPILER VALIDATION GREEN: driver forwards the flag, new lit PASS, clang-Capstone lit 7/7, llvm-Capstone backend lit 36/36, IR narrows to [field,field+sizeof) as intended. RUNTIME authority suite: human OK'd building the full stack in B (commit held until runtime probes pass). BUILT (standard host toolchain, `-j 80`, nice'd, serialized): capstone-qemu DONE (qemu-system-riscv64 v8.1.1 — note: had to force `CC=/usr/bin/gcc`; configure auto-picked a build-folder trunk clang from PATH that miscompiled op_helper.c). capstone-c DONE. buildroot DONE (fw_jump.elf/Image/rootfs.ext2 built). RUNTIME AUTHORITY VALIDATION GREEN (5/5): subobjfield_overrun→bounds-fault (the flip: field over-read now traps), subobjfield_{inbounds,union_active,flexarray}→ok (in-field works; union+FAM refusals hold, no false trap), subobject_overread (no flag)→no-trap-today (gap still in default compiler). Full increment validated: lit 7/7 + backend lit 36/36 + authority 5/5. Compiler changes ready to commit (awaiting human go-ahead). Compiler changes validated + uncommitted pending runtime pass. NOTE: buildroot build is Agent-A's firmware lane — running per explicit PI authorization; no firmware submodule gitlink bumped. Next: lit + authority-suite validation, then report for commit approval. NOTE cross-lane: added additive files under `capstone/tests` (A's held dir) per explicit PI task — see sync log. |

## Claimed / do-not-touch  _(hold list)_

| Path or submodule | Held by | Until |
|---|---|---|
| `capstone/agent-handoff/state/current-state.md` + `current-next-step.md` | Agent-A (base files) | ongoing — Agent-B uses `*.B.md` |
| _add entries as you claim exclusive edit rights_ | | |

## Submodule-bump log  _(append after every gitlink change — this is where conflicts hide)_

| Date | Agent | Submodule | Old→New SHA | Superproject commit |
|---|---|---|---|---|
| 2026-07-08 | Agent-A | (main repo) added `capstone/paper` @ db9142f (opt-in, update=none) | — → d4959767 | d4959767d27c |
| 2026-07-09 | Agent-B | `capstone/capstone-qemu` (implement `csdrop`/DROP — row-11 QEMU unblock) | cf541a1f → 2e6a67d1 | (superproject commit on `capstone-bootstrap-b`) |

## Sync log  _(append at each integration point)_

| Date | Action | Detail |
|---|---|---|
| 2026-07-08 | scaffolding | Added `MULTI-AGENT-WORKFLOW.md`, `AGENT-B-SETUP.md`, this file; paper submodule committed (`d4959767`). Push required for Agent-B's clone to pull them. |
| 2026-07-08 | Agent-B online | Agent-B came online in clone `/home/alexey/dev/llvm-capstone-b` on `capstone-bootstrap-b`. Verified isolation: `CAPSTONE_REPO_ROOT`/`CAPSTONE_CLANG` inside B clone, `CAPSTONE_TMP_ROOT=/tmp/capstone-b`, `CLAUDE_CONFIG_DIR=~/.claude-b` (own creds), remote=project-starch/llvm-capstone, clean tree. Created `state/current-{state,next-step}.B.md`. LLVM build + buildroot/rootfs not yet built (fresh clone). Awaiting task assignment; no submodule owned. |
| 2026-07-08 | Agent-B merge + lane claim | Merged `origin/capstone-bootstrap` (`39f68da`, agent-unavailable resilience + checkpoint protocol) into `capstone-bootstrap-b` (clean, no conflict). Claimed the **compiler/codegen + emulator** lane (`llvm/` in-tree + `capstone/capstone-qemu`); no firmware submodule touched. Task: design-only C1 subobject-bounds proposal (`design/c1-subobject-bounds-proposal.md`), gated for review before any implementation or full build. |
| 2026-07-09 | Agent-B csdrop shipped | Implemented `csdrop` (DROP) in `capstone-qemu` (row-11 LINEAR QEMU unblock, task `agentB-002`). Submodule bump `cf541a1f`→`2e6a67d1` on submodule branch `capstone-bootstrap-b`; superproject gitlink + this file + `state/*.B.md` + history note committed on `capstone-bootstrap-b`. Additive only (new opcode, previously illegal); no existing instruction changed; no firmware submodule touched. Validated under QEMU (control ok + fault=cause-24 clean cap-fault) with no regressions (smoke + borrow-revoke UAF pass). **Integration:** A merges B→canonical as with C1 v1 — but A also needs the `capstone-qemu` commit `2e6a67d1` reachable (push submodule branch to shared `project-starch/capstone-qemu`, or A fetches from B's clone). Row-11 full domain demo still needs A's gated linear-authority `start.S`. |
| 2026-07-09 | Agent-B C1 subobject v1 shipped | PI approved implementing. Landed `-fcapstone-subobject-bounds` (clang frontend, default off, arrays-only v1) + lit + authority `subobjfield_*` probes. Validated: clang lit 7/7, backend lit 36/36, runtime authority 5/5. **Cross-lane note:** added ADDITIVE files under `capstone/tests/capstone-authority/` (A's held dir) — new `subobjfield_*.c` domains + appended `oracle.tsv` lines + one additive case in `build-authority-suite.sh`; no existing probe/oracle line changed, no submodule gitlink bumped. Built the full runtime stack in the B clone (capstone-qemu, capstone-c, buildroot) with standard host toolchain (`/usr/bin/{gcc,clang++}`) at `-j 80` — did NOT modify any firmware submodule. No commits pushed; on `capstone-bootstrap-b` only. |

---

### Handoff etiquette
- Before bumping a submodule: check the **Claimed** list; if someone holds it, wait or coordinate here.
- After bumping: append to the **Submodule-bump log** in the same commit.
- Keep the branches close — merge B→A at checkpoints rather than letting gitlink SHAs diverge for weeks.
- Durable facts both agents need go in a committed `agent-handoff/` doc, **not** only in an agent's private memory (memory doesn't cross clones/accounts).

### Resilience / agent-unavailable (full protocol in `MULTI-AGENT-WORKFLOW.md`)
- **Commit + push small and often** (exact paths). Uncommitted WIP in a dead session is the only thing a usage-limit cutoff can actually lose.
- **Update your `state/*.md` and your Current-position line** at every checkpoint, so the other agent can take over your lane read-and-go.
- Lanes are independent: one agent hitting its limit never blocks the other's lane — it degrades two lanes to one, never to zero.
- If the integrator (A) is dark and B needs a merge: B keeps committing to `capstone-bootstrap-b`; integration waits or the human temporarily promotes B to merge B→canonical, and A reconciles submodule bumps from the log on return.
- The human can `git commit` a stalled clone's WIP at any time, even while that agent is dark.
