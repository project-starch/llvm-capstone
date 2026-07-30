# Capstone LLVM fork — Claude Code workspace

## Read first

Set up the environment, then read the minimal handoff set:

```bash
source capstone/tests/capstone-test-env.sh
```

- `capstone/agent-handoff/README.md`
- `capstone/agent-handoff/state/current-state.md`
- `capstone/agent-handoff/state/current-next-step.md`

New to the project? See `capstone/agent-handoff/ONBOARDING.md`.

## Hard constraints

- **Never mention any real person by name — anywhere.** No PI, supervisor,
  colleague, board owner, or collaborator names in commits, code, docs, reports,
  or any committed/shared content. Use neutral roles ("the board owner", "the
  collaborator", "the external collaborator", "the PI"). This is permanent and
  absolute. (Upstream `lldb/`, `llvm/` etc. files are not ours — leave their
  names alone.)
  - **This includes the commit-message SUBJECT line, and includes naming a
    collaborator even when merely *referring to* or *assigning work to* them**
    (e.g. "task for <name>" → "task for the external collaborator"). A name in a
    commit subject is the exact mistake made on 2026-07-25 (had to be amended +
    force-pushed) — do not repeat it.
  - **MANDATORY pre-commit AND pre-push check:** before every `git commit` and
    before every `git push`, scan the full commit message *and* the staged diff
    for personal names (grep the message and `git diff --cached` for known
    collaborator/PI names). Treat any hit as a release-blocking error — fix
    before committing. If a named commit was already pushed, amend/rewrite and
    force-push, and notify the other lanes (they must re-sync).
    Run it with the script, which also catches emails, attribution trailers,
    `user@host` build strings, debug files and the FPGA token:

        bash capstone/tests/precommit-scan.sh --msg <msgfile>    # exit 1 = BLOCKED

    This is a **script, not a subagent**, deliberately: a release gate on an absolute
    rule must be deterministic. Never delegate this check, and never weaken a pattern
    to make it pass. The exact-name list lives **outside the repo** at
    `~/.claude-c/secrets/name-denylist.txt` (mode 600) — putting the names in a
    committed file would itself break the rule. Keep that file populated; without it
    the script warns and runs only its name-independent heuristics.
- No `Co-Authored-By:` lines in commits.
- Never commit debug/report files (`*_DEBUG_CHECKPOINT.md`, session notes).
- Active plans live in `capstone/agent-handoff/plans/` (committed, portable across machines and agents).
- Manager-facing summaries go under `/tmp/capstone/`, not into the repo.
- **Never name people in committed files.** Do not write the PI, co-PI, any collaborator, or
  any individual's name (or personal build hostnames like `root@<name>`) into any file that
  gets committed. Refer to roles generically: "the project lead", "the collaborator", "the
  team". Names, meeting attributions, and questions/notes addressed to a person go under
  `/tmp/capstone/` only — never committed, never pushed. (Functional exceptions that are not
  name-drops: real dependency URLs in `.gitmodules` and published-paper citation URLs.)
- **Never commit or share the FPGA console link or token.** The board URL/host and its token
  are secret. Keep the full URL in an env var (`FPGA_URL`) only, for the duration of a run;
  in any committed text use the placeholder `<FPGA-CONSOLE-URL>`. Never echo it into a
  captured/committed log.
- **ASK BEFORE EDITING THE PAPER (`capstone/paper/`).** Never edit, restructure, or add
  sections to the paper on your own initiative — propose the change and wait for an explicit
  go-ahead, even when the edit is obviously implied by new results and even when the results
  are already validated and committed elsewhere. "The numbers are ready" is not authorization
  to write the prose. Two reasons this is a hard rule and not a preference: the repo syncs
  with Overleaf, so an unasked-for edit can collide with work in progress there; and the
  paper's framing is the project lead's call, not a lane's. **Reporting** new results into
  `agent-handoff/ref/fpga-silicon-measurements-for-paper.md` needs no permission and is the
  right default — that doc exists precisely so results can land without touching the paper.
  When permission *is* given: never `git push` the paper submodule (Overleaf owns the remote),
  and leave the parent's submodule pointer unbumped.
  **Go-ahead is per session and per AREA, not per edit.** Once the project lead approves a
  class of change ("fix the numbers", "apply 2, 4, 5, 6"), further mechanical instances of
  that class need no fresh approval — asking again is friction, not caution. Structural
  reordering and authorial content (abstract, new prose, section order) still need their own
  go-ahead.

## A checkpoint is not a decision point

**When the next step is known and inside the current goal, take it and report afterwards.**
Do not stop to confirm. Finishing a diagnostic and knowing what follows is a status update,
not a permission request; turning it into one wastes a turn and, on this project, board
time.

Ask only when:
- proceeding under any assumption would be **unsafe or irreversible** — bitstream reflash,
  force-push, destructive delete, anything outward-facing;
- the answer is **genuinely the project lead's** (paper framing, authorial content, project
  direction);
- **or you have RETRACTED a conclusion.** That one is worth surfacing every time. A session
  that produced seven retractions produced them because reasoning ran one step past the
  evidence — the lead needs to see that, whereas next steps they do not.

**Never bundle.** If eight items need no permission and one does, do the eight and ask the
one question at the end. Stalling finished work behind an open question is the single most
common way this goes wrong.

Corollary: pause when you have **concluded** something, not when you have **finished**
something.

## Context & compaction

Board-debug threads here run long. **Do NOT routinely recommend `/compact`.** Keep
working and let context management happen on its own; a compaction suggestion at a
natural-looking checkpoint is usually just an interruption.

**Do not announce being "low on" or "out of" context, and do not use it as a reason to
stop or to defer a hard step.** That has been a false alarm every time it was raised so
far: work continued productively immediately afterwards, and the actual failures were
process slips (grepping for a hex constant emitted in decimal, `grep -c "A\|B"` that
cannot say which matched, running a rebuilt binary the harness does not actually load,
reading a Makefile's echoed recipe as an error) rather than anything caused by lost
state. Hedging before a hard problem reads as an excuse and wastes a turn. If a
mistake happens, name the specific slip and the check that would have caught it.

Raise compaction **only when not compacting would actually damage progress** — i.e. context
is close enough to exhausted that the next steps would be taken with important state
already lost, and that state is not yet in committed docs/memory. In that case say so
in one line, with the specific thing at risk. Otherwise say nothing about compaction.

The durable protection is not compaction, it is committing: land findings in
`agent-handoff/` and commit messages as you go, so a summary losing detail costs
nothing.
Never compact unilaterally — you can only recommend; it is the user's call.

When you recommend `/compact`, also give a short **compaction brief** — the generic
summarizer can't tell what's load-bearing, so tell it: what to **keep verbatim**
(current task + exact next step, un-committed decisions/rationale, open blockers,
live file paths/values still in flight) and what's safe to **compress** (resolved
sub-threads, tool-output noise, superseded approaches, anything already committed to
docs/memory). A couple of lines is enough; skip it only when there's nothing special
to preserve beyond the obvious.

## Where things live

| What | Where |
|------|-------|
| Current state + next step | `capstone/agent-handoff/state/` |
| Test matrix + cookbook | `capstone/agent-handoff/ref/` |
| Architecture + design docs | `capstone/agent-handoff/design/` |
| Active WIP plans | `capstone/agent-handoff/plans/` |
| Bug-fix investigations, root-cause trails, audits | `capstone/agent-handoff/history/` (dated `DD-MM-YYYY_HH-MM-SS_name.md`) |
| Archived session notes | `capstone/agent-handoff/history/` |

**`design/` is for design decisions and architecture only.** A bug fix,
root-cause investigation, or audit — even a substantial one — is *not* a design
decision; it belongs in `history/` as a dated note, not in `design/`.

## Delegating to subagents

Delegation is **opt-in and at the main session's discretion** — not automatic.
The main (Opus) session owns planning, synthesis, reviewing subagent output, and
all final decisions. Subagents start cold, so any that could write or run tests
**inherit this file's hard constraints** (no real-person names; serialize QEMU
suites — the shared `rootfs.ext2` write-lock means never two in parallel;
`ninja -j90` never `-j112`; never commit unless asked; submodule source stays
uncommitted). Subagents do not recurse.

- **Delegate** (notify the user when it's substantial):
  - broad read-only code/file search → built-in **Explore**
  - regression corpus / lit / QEMU suites → **corpus-runner** (read-only, serialized,
    never touches the board)
  - "what does the SILICON actually do?", and any FPGA-only failure that QEMU does not
    reproduce → **rtl-oracle** (reads `capstone-ariane` `.anvil` sources and diffs them
    against our QEMU helpers)
  - verifying a root-cause claim / "X is fixed" / "Y is ruled out" **before** it enters
    ISSUES.md, a commit message, or the paper → **claim-auditor** (adversarial; tries to
    refute)
  - classifying a large board/QEMU run log → **board-log-forensics**
  - checking paper numbers against the measurements doc → **paper-numbers-checker**
    (read-only on `paper/`; reports, never edits)
  - bounded multi-step research with a clear question → **general-purpose**
- **Keep in the main Opus session** (never delegate to subagents): compiler/codegen
  and capability-ABI changes; subtle-correctness or concurrency debugging; **choosing the
  next experiment in a live investigation**; the paper; commits; and anything involving
  real-person names.

**Treat every subagent report as a claim, not a fact.** Findings that will be acted on,
committed, or published must be verified against the primary source — a quoted `file:line`
you can re-read, or a command you re-run. A confident subagent report is exactly as
trustworthy as a confident guess by this session, which the history in `ISSUES.md` shows
is not very. Agents are instructed to quote evidence and to say UNRESOLVED; hold them to
it, and treat a conclusion with no quoted evidence as unverified.

Full roster, rationale, and prompt patterns: **`agent-handoff/ref/SUBAGENTS.md`**.

**FPGA/board sessions may be run by EITHER Opus lane (A or B)** — B is explicitly
**not** prohibited from the board (permanent rule). The board is a single shared
physical resource (secret token, human-in-the-loop, can't be parallelized), so board
sessions are **serialized across lanes**: never two at once — coordinate timing and
hand off sequentially. (Built-in **subagents** still never touch the board; the
corpus-runner is board-free.)

Peer **lane B** (a separate Opus session on `capstone-bootstrap-b`) is a different
thing from subagents — see `capstone/agent-handoff/ref/SUBAGENTS.md` (the peer-lane
guide is archived at `capstone/agent-handoff/history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`). A third category
is an **external collaborator running their own (non-Claude) coding agent**: give
them a **self-contained, stock-toolchain** task doc in `plans/` that is **decoupled
from our in-flux compiler/ABI/board** (so the churn here can't block them), and keep
the collaborator's real name out of the repo. Non-Claude agents don't auto-read this
file — the `agent-handoff/ONBOARDING.md` callout covers pasting it as context. The
reproduction/repro-artifact half of a new benchmark can go this way; the
capability/compiler/board half stays in an owning lane.
