# Subagents: roster, when to use them, and the rules they inherit

Built-in subagents are a different thing from **peer lane B** (a separate Opus session —
see the archived `../history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md`) and from an **external collaborator's own agent** (see the ONBOARDING
callout). This file covers only the in-session subagents defined in `.claude/agents/`.

Delegation is **opt-in and at the main session's discretion**. The main Opus session owns
planning, synthesis, review of subagent output, and every final decision. Subagents do not
recurse.

## Roster

| agent | model | use it for | never |
|---|---|---|---|
| **rtl-oracle** | sonnet | What the silicon actually does; RTL↔QEMU divergence. Any FPGA-only failure QEMU won't reproduce. | edits, builds, board |
| **claim-auditor** | opus | Adversarially refute a finding *before* it enters ISSUES.md, a commit, or the paper | edits, builds, board |
| **board-log-forensics** | sonnet | Classify a large run log: pass / miscompute / fault / hang / transfer failure / infra flake | drives the board |
| **paper-numbers-checker** | sonnet | Cross-check paper numbers and status claims against the measurements doc | **edits `paper/`** |
| **corpus-runner** | sonnet | Run the regression corpus / lit / QEMU suites and report pass-fail | fixes code, board |
| **Explore** (built-in) | — | Broad read-only code/file search | — |
| **general-purpose** | — | Bounded multi-step research with a clear question | — |

## Keep in the main session

Compiler/codegen and capability-ABI changes; subtle-correctness debugging; **choosing the
next experiment in a live investigation**; the paper; commits; anything touching
real-person names.

The reason for the third item is concrete: deciding that a bisection was invalid, or that
one instruction couldn't matter because a descriptor field said the branch was never
taken, depends on holding the whole failure history at once. A subagent starting cold will
produce a confident answer to the wrong question.

## The rules every agent inherits

Each agent definition restates these, because subagents start cold and do not read
`CLAUDE.md`:

1. **Never name a real person.** Submodule git histories (notably `capstone-ariane`)
   contain real contributor names and email addresses; agents *will* encounter them via
   `git log`/`git blame`. Never reproduce them — use "the RTL author", "upstream", "the
   board owner", "the collaborator".
2. **Never print or commit the FPGA console URL or token.** Placeholder:
   `<FPGA-CONSOLE-URL>`. Redact anything token-shaped, including from quoted logs.
3. **Never touch the FPGA board.** It is a single shared physical resource with a secret
   token and a human in the loop; board sessions are serialized across lanes by the main
   session.
4. **Never commit.**
5. **Serialize QEMU.** All suites share one `rootfs.ext2` write lock — never two at once.
   Only `corpus-runner` runs suites; everything else reads logs.
6. **`ninja -j90`, never `-j112`** (a parallel debug-link storm takes the whole box down).
7. **Never `git checkout --` inside a submodule** — it destroys uncommitted local work
   with no undo.

## How to prompt them

- **State the question, the acceptance criterion, and where the sources are.** Agents
  start cold; a goal plus guardrails beats a list of steps.
- **Demand quoted evidence.** `file:line` plus the actual lines. A conclusion without a
  quote has to be re-derived, which costs more than doing it yourself.
- **Require an explicit "unresolved" channel.** An agent that cannot say "I'd need to read
  the decode path to answer that" will guess instead.
- **Say what silence means.** "Not mentioning an instruction reads as *checked and fine*,
  so list anything you did not resolve."

## How to read what comes back

**Treat every report as a claim, not a fact.** Before acting on, committing, or publishing
a subagent finding, verify it against the primary source — a quote you re-read, or a
command you re-run. This is not distrust of the model; it is the same standard this
session's own conclusions have repeatedly failed to meet (see the retraction trail in
`ISSUES.md` under C-13 and R-9). A finding with no quoted evidence is unverified by
definition.

For anything that will be recorded or published, the cheap move is to route it through
**claim-auditor** first and let something adversarial try to break it.

## Not a subagent: the name/secret release gate

`capstone/tests/precommit-scan.sh` is a **script on purpose**. A gate on an absolute rule
must be deterministic and fail identically every time; "usually right" is the wrong
property. Never delegate it, and never weaken a pattern to make a commit pass.

Its exact-name denylist lives **outside the repo** at
`~/.claude-c/secrets/name-denylist.txt` (mode 600) — hardcoding the names in a committed
script would itself be the violation it is meant to prevent.
