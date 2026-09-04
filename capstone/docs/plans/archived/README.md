# Archived plans

A plan lands here when it is **finished, superseded, or overtaken by events** — never because it
was wrong. Several of these were correct and simply completed; a few were correct reasoning that
later evidence moved past. Nothing is deleted, because a plan's *rejected* alternatives are often
the most useful thing in it.

**Read the status line inside an archived plan with the archive date in mind.** A plan that says
"PROPOSAL, awaiting a decision" was accurate when written; the decision has since been taken. The
reason for archiving is recorded below, not edited into the document.

## Archived 2026-09-04 — the S-12 cluster

S-12 was root-caused, fixed in RTL, synthesised and flashed. Everything that existed to work
around it, to instrument it, or to decide whether to instrument it, is therefore closed.

| plan | why archived |
|---|---|
| `sqlite-wherecode-notcap-plan.md` | this *is* S-12, under its pre-diagnosis name. Its own status line says "NOT root-caused by this investigation" — that is now historical. |
| `s12-codegen-mitigation-proposal.md` | a compiler-side workaround, deprioritised by the project lead in favour of fixing the hardware. Kept as the record of the option not taken. |
| `s12-fix-synthesis-request.md` | synthesis ran and the bitstream is flashed. |
| `s12-recorder-go-no-go.md` | the go/no-go it asked for was decided — NO GO. The predicted reading would have been uninformative, which is exactly the check the request existed to force. |

## Archived 2026-09-04 — S-07 debugging-process plans

| plan | why archived |
|---|---|
| `18-08-2026_s07-v3-make-debugging-fast.md` | proposals for making board debugging cheaper. Overtaken: S-12 was ultimately debugged in ~14 s of RTL simulation rather than on the board, which is the same goal reached by a different route. |
| `18-08-2026_s07-v4-per-boot-or-per-run.md` | as above. |

Note that **S-07 itself is not closed** — `ref/ISSUES.md` records it as fixed in simulation with
its silicon validation *downgraded*. `S07-instrumentation-complete-spec.md` therefore stays live.

## Archived 2026-09-04 — superseded

| plan | why archived |
|---|---|
| `xlang-phase2-seam-TODO.md` | says so itself: superseded 2026-08-01 by `plans/capstone-column-xlang.md`. |
