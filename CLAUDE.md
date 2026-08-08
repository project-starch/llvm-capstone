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

A standing instruction stays standing. "Proceed", "don't ask", "keep going until it runs"
covers the whole goal, not the next step of it — so it does not need renewing at each
checkpoint, and re-asking reads as not having heard it.

This section is about *checkpoints inside work already asked for*. It does NOT relax
anything under "Hard constraints" — those are the project lead's rules, and an agent must
not widen its own permissions by editing this file. In particular the paper rule stands as
written: if the scope of a given go-ahead is unclear, ask, and asking there is not the
friction this section is about.

## Capture the lesson when it is cheap — do not wait to be asked

Process improvements have so far been written only when the project lead asked for them, which
means they land long after the cost. The rule below this one was worth writing after the *third*
instrument error in a session; it got written after the tenth.

**Triggers. When any of these fires, stop and ask whether the lesson generalises:**

- a **RETRACTION** — you already have to surface it, so ask in the same breath what would have
  caught it;
- the **same class of mistake twice** in one session — recurrence is what separates a pattern from
  an accident;
- a **gate, check or tool found to have been silently not working** — that class repeats, and the
  next instance is already being written somewhere;
- a **wasted boot, wasted session, or a result that had to be thrown away**.

**Then, before writing anything, check whether it is already covered** — CLAUDE.md, the skills, the
handoff docs. Most of the time it is, and the right action is to do nothing, or to sharpen one
existing sentence. Only write a new rule if it is genuinely absent *and* it would have prevented
the loss.

**Guardrails, because the failure mode here is noise:**

- **One rule at a time**, in the smallest scope that works.
- **Pick the right home.** A procedure for a recurring task → a skill. A habit that applies
  everywhere → this file. A fact about one investigation → `agent-handoff/`, not here.
- **Prefer sharpening an existing rule** to adding a new one. This file earns its keep by being
  read; every addition taxes that.
- **Never touch "Hard constraints", and never widen your own permissions.** Those are the project
  lead's. Propose, do not enact.
- **Say what you considered and rejected**, so the lead can see the alternatives were weighed
  rather than a rule being reflexively appended.

## A CLEAN result is not evidence until the check is known to fire

**Before believing a zero, a pass, or a "not found", show the check can produce the opposite.**
Run it against a case that must trip it. If you cannot make it fire, you have learned nothing about
the subject and something about the instrument.

This is the single most expensive mistake made on this project. On 2026-08-08 one session hit it
**ten times**; four became published claims that had to be retracted:

* `grep -c` returned 0 because grep here is **ugrep** and goes quiet on binary-ish output → "this
  binary contains no `movc`" (it contained nine);
* a check keyed to one function found nothing because the code under test lived in **another
  function**, and one keyed to one packing shape missed a **different packing shape** → two builds
  wrongly declared "never measured";
* a preflight gate tested a marker over the **whole transcript** when the mandated control always
  emits it → the gate could never fire, and had not, in its entire life;
* a preflight run with `RUNGS=` instead of `BAKED_RUNGS=` printed **GO** having checked nothing;
* an analysis tool with an unset input dir printed `dataset: 0 builds` and a table of `0/0` scores,
  which reads exactly like "no rule fits" rather than "no data was loaded";
* six directed RTL tests came back clean and were recorded as "the hardware is innocent" — none of
  them ever created the triggering condition.

Cheap habits that catch all of the above:

* **Give every detector a positive control.** A gate that has never blocked anything is not a
  passing gate, it is an unproven one. Negative-test it the day you write it.
* **Prefer `python3` to `grep`** for anything that must be counted or must return zero meaningfully.
* **Make "no data" an ERROR, not a zero.** A tool that finds nothing should exit non-zero and say
  where it looked, never print an empty result that renders like a finding.
* **`pgrep -f <pattern>` matches your own shell.** Match on a verified PID or a distinctive
  substring that cannot appear in your own command line.
* When a result is *surprisingly* clean, suspect the instrument before the subject.

## Debugging a blocker: BATCH VARIANTS, and make every run RETURN

*(Execution mechanics — bake, boot, invoke, classify, release — live in the `board-run` skill,
which auto-loads. This section is the experiment-design half: what to build and in what order.)*

Default method for any "it hangs/fails somewhere on the board and we don't know where"
problem. Learned the expensive way on 2026-07-31: six board sessions were spent probing a
wedge one hypothesis at a time and produced nothing usable — the eventual answer came from
one session that ran four variants.

**1. Make every run produce a result.** A wedged domain emits nothing, so a failed run only
ever says "somewhere after the last marker" — one bit per session, and possibly a bit about
the wrong function. Build variants that stop early and **return a marker** (stage number +
error code) instead of running to the failure. A build that returns always yields data, so
the bisection converges instead of guessing.

**2. Batch them into ONE boot.** Booting the board is ~2–3 min and dominates a short run,
so N hypotheses as N sessions is mostly boot time. Stage all the variants into the same
initramfs (they are just extra `.dom` files, one firmware rebuild covers all of them) and
run them in sequence from a single boot. `run_sqlite_stages_fpga.py` is the worked example.

**3. Order them so the cheap/safe ones run first.** A wedge takes the core with it, so
everything after the first wedge is lost. That is not a limitation to engineer around —
the first variant that fails to return *is* the bisection point. Put ascending stages in
order, and put a probe you expect to hang last.

**4. Keep the real path byte-identical.** Put staged logic in a separate function behind
`#ifdef`, never as `#ifdef`s threaded through the production path — otherwise the
bisection is about a build that doesn't matter.

**5. Batch HYPOTHESES, not just stages of one hypothesis — and batch them WIDE.** This is
the rule that kept being missed on 2026-07-31 *after* the rest of this section was written:
the ladder got batched, but each new idea then went to the board on its own. Three loads
(`w10/w2/w3`, then `wc0/wc9`, then a full run) tested what one load of six domains would
have. A firmware rebuild plus JTAG load plus boot costs ~5 minutes and covers *every* `.dom`
in the image, so the marginal cost of the seventh variant is a few seconds of run time.

Before spending a load, write down every question currently open and build a domain for each
one — including the controls. Concretely, the load that should have been run instead of
those three: does cap-init still work at the higher store count (stage 0), does the last
known-good stage still pass (stage 9), does the changed function pass now (stage 10), does
the next stage up pass (stage 2, stage 3), plus the two ordering controls. Six domains, one
boot, and no need to guess which single question was the most valuable.

**Ordering under batching is what makes it safe:** a wedge kills the rest of the session, so
put every domain you expect to RETURN first, in ascending order, and at most ONE
expected-to-wedge domain last. If two might wedge, they need two loads — or accept losing
whatever follows the first.

Never send a single-domain load unless it is the final confirmation run of something already
bisected.

Corollary for instrumentation: prefer a diagnostic that **converts a hang into a wrong
answer** (a clamp, an early return, a bounded loop) over one that only observes the hang.
Observation of a wedged core is unreliable here — the debug register path has twice
returned AXI error-slave junk (`0xca11ab1ebadcab1e`), and a pc sampled under `stepi` says
nothing about free-running execution.

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
| Skills (auto-loading procedures) | `.claude/skills/<name>/SKILL.md` |

**`design/` is for design decisions and architecture only.** A bug fix,
root-cause investigation, or audit — even a substantial one — is *not* a design
decision; it belongs in `history/` as a dated note, not in `design/`.

## Skills

A **skill** is a procedure that auto-loads when a task matches its `description` — no one has
to remember to open a reference doc. That makes it the right home for a workflow applied under
time pressure on a shared resource, where the cost of skipping a step is a wrong verdict.

| Skill | Use it for |
|---|---|
| `board-run` | ANY board execution: bake into the image, boot, invoke, classify, release. Also read it before interpreting a run that produced no result. |

A skill is just `.claude/skills/<name>/SKILL.md` with YAML frontmatter (`name`, `description`);
no tooling is needed to add one. Two things to know:

- **`.gitignore` ignores `/.claude/*`.** Tracked subtrees need an explicit negation — there are
  now `!/.claude/agents/` and `!/.claude/skills/`. Without one a new skill is silently NOT
  committed, and the commit message describes a file that isn't there. Check `git ls-files`.
- **Skills are not subagents.** A skill loads instructions into the *current* session; a
  subagent runs work in a *separate* context. Use a skill for "how do I do this correctly",
  a subagent for "go do this and report back".

## Delegating to subagents

Delegation is **opt-in and at the main session's discretion** — not automatic.
The main (Opus) session owns planning, synthesis, reviewing subagent output, and
all final decisions. Subagents start cold, so any that could write or run tests
**inherit this file's hard constraints** (no real-person names; serialize QEMU
suites — the shared `rootfs.ext2` write-lock means never two in parallel;
`ninja -j90` never `-j112`; never commit unless asked). Subagents do not recurse.

**Submodule source SHOULD be committed.** The old rule that it "stays uncommitted" is
withdrawn (2026-08-05). It cost real work: the live OpenSBI monitor — the file carrying the
SHA/ECSA/RGID trace markers that every board verdict is classified by — sat 680 lines ahead of
its last commit in a working tree only, one `git checkout` away from destroying the basis for
the entry-stall-vs-wedge rule the drivers and the `board-run` skill both encode. The same was
true of the device-tree memory map, a kernel-module `copy_from_user` fix, and the buildroot
target ordering. Commit submodule work on a branch (`capstone-bootstrap`, or a task branch),
message it like any other change, and run `precommit-scan.sh` over it — the name rules apply to
submodule commits exactly as they do here. Pushing stays a separate decision: those remotes are
shared, so ask before pushing.

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

**Run an auditor on your own judgement, without being asked, when a trigger fires.** Auditors
refuted FOUR substantive claims on 2026-08-06 -- a `ctvec` root cause, a bitstream regression, a
`PcacheInitialize` localization, and "this monitor fix is safe" (which would have unmasked a
silent region-table overrun). Each cost 8-20 minutes and saved considerably more. Triggers:

* before recording a **root cause or localization** in `SILICON-BLOCKER.md` / `ISSUES.md`, or in
  a commit message that claims one;
* before a **monitor, bitstream, or otherwise irreversible** change;
* when a result **contradicts** a documented prior finding -- one of you is wrong and it is
  cheaper to find out which before acting;
* when a conclusion rests on **N=1** on a system with known nondeterminism.

Not for routine passes, mechanical rebuilds, or anything already verified. And name your own
weakest link in the prompt ("the gap I want you to attack hardest is X") -- generic "check this"
produced thin reports; naming the soft spot produced every one of the refutations.

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
