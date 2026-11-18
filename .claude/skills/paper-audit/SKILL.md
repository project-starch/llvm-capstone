---
name: paper-audit
description: >-
  Audit paper text (a section, an appendix, a paper PR) with subagents: which auditors to launch,
  what to put in their prompts, how to triage what they return, and when to audit again. Use when
  asked to audit, check or review paper text, before calling a paper PR ready to merge, and after
  any rewrite of paper prose. The audits are read-only; editing the paper still needs the lead's word.
---

# Auditing paper text

This procedure comes from a study of the 33 audits run on the evaluation text on 2026-11-16..18
(three rounds: the first draft, the fairness of the comparison, the PRs before merge). The study
followed every finding to its end: applied and kept, applied and later removed, deferred, or wrong.
What follows is what that record supports.

Auditing needs no permission. **Editing the paper does** (CLAUDE.md, "ASK BEFORE EDITING THE PAPER"),
and every edit goes through an `eval/*` PR that the lead merges.

## 0. Before launching anything

1. **Settle the framing first.** Every audit pushes toward saying more. Not one of the 33 asked
   whether stating a true thing harms the paper. The lead had to ask that, after the additions were
   already in a PR, and much of what the audits had added was cut again. So before
   round 1, know: which sections may be edited, the page budget, and what the lead does not want
   said. Use what the lead has already told you. If nothing is known, ask once and carry on with the
   default in section 3.
2. **Audit the current source.** Fetch `main`, rebase the PR, rebuild the PDF. One audit ran on a
   PDF older than the source and three of its findings were void on arrival.
3. **Write one brief file** in the scratchpad and give every auditor its path. Contents:
   - the tree and the PDF to read, the sections in scope, the sections that are context only;
   - the primary records by path, with search terms (results log, pre-registration, README, the
     provenance comments in `macros/results.tex`);
   - the definitions an outsider needs (what each build is, which prototype, how many runs);
   - **a "known, do not re-report" list.** A brief that announced a stale sentence as a fact got that
     sentence back from all seven auditors;
   - read-only, no names of people, no board, no commit.
4. **A formula is audited before anyone sees it.** Put one measured number through it and check
   the sign and the size of the result. A correction formula was once pre-registered and sent to a
   co-author inverted, in our favour. An auditor caught it only because it happened to be asked.

## 1. Which auditors

Launch a round in ONE message so the agents run in parallel. Wait for all of them before changing
anything: two auditors once disagreed on a presentation question and the first to arrive won.

**Core set, every round** (about 10 minutes each):

| Lens | Agent | The prompt must say |
|---|---|---|
| Numbers | `paper-numbers-checker` | The record paths. "Recompute every value. Do not trust the PR body or the macros' comments." Counts of things ("N of M applications") are numbers too. |
| Claims | `claim-auditor` | Three to five named weakest links, and a falsification target ("a free list in any of these programs would falsify the sentence"). One open slot: "true but misleading to a reader who later sees the raw records". |
| Wording and house style | `general-purpose` | "First extract the paper's conventions from the sections we did not write, then judge ours against them." Lab jargon, undefined terms, number formats, captions. Each item: file:line, quoted text, exact replacement. |
| Over-disclosure | `general-purpose` | The lead's framing, quoted. The rule: keep a detail only if a sentence becomes false, or a result is hidden, without it. "A reviewer might ask" is not a reason to keep. |

**On a trigger:**

| Trigger | Lens | Notes |
|---|---|---|
| The text compares us against a baseline | Fairness (`general-purpose`) | Enumerate every difference between the arms and classify each as removed, corrected, disclosed or hidden, with the sentence or action for each. Needs the configs, kits and run scripts, not only the paper. |
| One number rests on N=1, a probe, or a derivation | Single-claim audit (`claim-auditor`) | Give it the raw readings and the probe sources, not your summary. The best yield per minute of the whole study. |
| A new measurement is being designed | Design panel (`general-purpose`) | Ask for the cheapest run that would change the decision and a written prediction. |
| The submission is near and the records are complete | Blind re-derivation (`general-purpose`) | Recompute every number from transcripts without seeing the paper's values. Carry its limiting notes through to the lead; they were dropped once. |
| There is a page budget to meet | Concision (`general-purpose`) | Give the budget in lines. Allow it to drop content and list every drop. With "lose nothing" it stops near 20 % and cannot arbitrate. |
| The text reads machine-written | AI-writing signs (`general-purpose`) | Ask for judgement on this paper: rhythm, bookkeeping paragraphs, a buried result. A fixed keyword list found nothing. |
| The text is frozen | Layout (`general-purpose`, reads PDF pages as images) | Always last. Layout fixes did not survive the next text edit. |

**Did not earn their cost.** Run them only when the lead asks:
- a mock programme-committee review from the PDF alone: it echoed the prompt's own list and guessed
  at costs without knowing which data existed;
- a completeness lens ("what would a reviewer want added"): its additions were later removed, and
  its cuts were ignored;
- whole-paper terminology or cross-section checks when only two sections may be edited: real
  findings, but they become a deferred list nobody owns;
- prior-work research for a paragraph that is not certain to exist.

## 2. Prompt rules

- **Name the weakest links.** Every refutation in the record came from a named soft spot or a
  falsification target. "Check this" returned confirmations.
- **Do not pre-list the issues as things to check.** They come back confirmed. Put known issues in
  the brief's "do not re-report" list instead.
- **Give raw evidence, not your reading of it.** Auditors corrected the brief twice and caught two
  errors in a prompt's own evidence lines.
- **Ask for file:line, the quoted text and an exact replacement**, and a forced verdict per item
  (holds / false / misleading / unresolved). Allow "not accessed" and "unresolved".
- **A rewrite may not add a claim.** One suggested rewrite introduced a cause that the records did
  not support. It reached a PR and had to be retracted.
- **No background commands in an auditor.** A leftover one re-woke an agent and its stub turn
  replaced the report. Take the report from the hand-back message.
- Do not audit "the plan". An audit of changes that are "assumed to be in" checks nothing that
  exists.

## 3. Triage before any edit

Never apply a whole round in one pass. That one habit caused most of the later cuts. For each
finding:

1. **Verify it against the primary source.** Auditors were wrong more than once: a style
   unification that merged two different measured quantities, a replacement sentence without
   support, a rewrite that added a cause the records do not show.
2. **Put it in one class:**
   - *A sentence is false.* Fix it by **weakening or deleting** the sentence. A caveat is the last
     choice. The caveats were what the lead later removed.
   - *True, and nobody asked.* Do not add it. Add only if a sentence is false without it.
   - *Outside the editable sections.* One list for the lead, each item with the decision needed.
   - *Needs a measurement.* Pre-register it. Ask what the paper would say differently for each
     outcome. If nothing, do not run it.
3. **Resolve conflicts between lenses by the record and the lead's framing**, never by arrival
   order or by count. Three auditors agreeing is one finding.
4. Run the over-disclosure lens **inside the round**, on the planned edits, before they are applied.

## 4. Apply, then audit the rewrite

After any rewrite, run `claim-auditor` again with: "only the problems that remain; file:line, quoted
text, exact replacement". Half of the second audit's findings were errors the rewrite had introduced.
Then rebuild, and run layout last.

## 5. Report to the lead

- What changed, in one line per change. What is theirs to decide, as a separate short list.
- Keep it short. The lead asked for less four times in one round.
- Do not bury the audit result among board status. A summary lost that way had to be asked for again.
- A retraction is surfaced at once, as CLAUDE.md requires.

## What audits do not catch

- **A wrong measurement.** Every lens took the readings as given. One reading that all of them
  accepted was an artefact of the patch under test.
- **Whether a true statement helps the paper.** That is the lead's call.
- **Their own duplicates.** Without the "do not re-report" list, parallel auditors spend most of
  their reports on the same three items.
