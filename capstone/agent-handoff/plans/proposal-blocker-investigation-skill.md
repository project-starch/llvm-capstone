# Proposal: a `blocker-investigation` skill, centred on repro minimisation

Status: **proposal, not implemented** — for review before writing the skill.
Date: 2026-08-03.

## 1. Is it worth it?

**Yes, but only if scoped narrowly.** The case rests on measured cost, not on a feeling that
process docs are nice.

On 2026-08-02/03, investigating R-14, roughly eight candidate causes were proposed and refuted.
**Three of them were stated as "the root cause" and then retracted**: the merged-string-blob
derivation, repeated `ldc` from one cap-table slot, and `stc` with a non-zero immediate off a
lui-derived base. None of the three failed because the evidence was thin at the time. Each
failed because a *specific, repeatable* step was skipped — and the same three steps keep
recurring (see §3).

Against that: two documents already cover part of this ground, and a third copy would be worse
than nothing.

| Already covered | Where |
|---|---|
| batch variants, one boot, ordering, make every run RETURN | `CLAUDE.md` §"Debugging a blocker" |
| bake → boot → invoke → classify → release, entry-stall vs wedge, control voids the boot | `board-run` skill |

**So the new skill must not restate batching, ordering, or execution.** Its unique content is
exactly two things: **repro minimisation** and **hypothesis hygiene**. If it cannot be written
without duplicating the other two, it should instead become a section inside `board-run`.

## 2. Minimisation is the core — and the parent's instinct is right

The single largest lever in the whole investigation was shrinking the reproducer:

    SQLite-derived probe image   1 624 128 B   181 carves   1 draw per boot, mostly R-16-blocked
    standalone ladder rung          10 896 B    10 carves   10 probes per boot, R-16 never blocked

150x smaller, 18x fewer carves. Two consequences, and the second matters more than the first:

1. **Throughput**: ~5 min for ten probes, versus a firmware rebuild + boot per single draw.
2. **It changed the answer.** Every candidate that survived at SQLite scale died once probes
   were small enough to vary one thing at a time. Minimisation was not an optimisation of the
   search; it *was* the search.

### The technique that actually worked: a two-directional squeeze

Shrinking alone is not it. What converged was:

* **shrink DOWN** from the failing case — `r14lp` → `e1sml`/`e2one` → `k1200`;
* **grow UP** from a trivially-passing case — `clp1` → `clp16` → `cgs8` → `cgpad` → `h2adj`;
* stop when the two meet at a **pair differing in ONE variable**.

**The terminal artifact of minimisation is a PAIR, not a small failing case.** A small failing
case still leaves the cause unlocated; the pair localises it. Here the pair is
`k800` (passes) / `k1200` (fails) — *identical source*, differing only in the size of a dead
`volatile` pad. That pair is what makes the RTL verdict defensible and is what the board owner
receives.

### Three failure modes of minimisation, all observed

1. **It can stop reproducing for an unrelated reason.** R-16 is image-sensitive, so a smaller
   or merely different image may fail to *enter*, which says nothing about the bug. Re-verify
   the control on every step.
2. **It can mask.** `w3out` "simplified" by outlining the stores and turned a hang into a
   **silent wrong answer** (0 where 4 is correct). A step that changes the failure mode is not
   a minimisation — it is a different bug.
3. **The compiler can delete the thing under test.** `cld2/4/8` were CSE'd into ONE `ldc`
   regardless of N, and memory barriers did not stop it; only a loop produced the repetition.
   Verify the artifact, never the intent.

## 3. Hypothesis hygiene — the other half

The rule that would have prevented all three retractions:

> **Before believing a documented mechanism, check its PRECONDITION against THIS build.**

* merged-string-blob — refuted by `:143` (same literal 8x still wedges) and by `r14b` failing
  with merging OFF (`cl::init(false)`, never set by `build-ladder-domain.sh`);
* LDC linear-clearing — the cited RTL note applies to LINEAR values, but the cap-table entries
  are provably **NONLIN** (`delin t2` precedes every `stc t2, N(gp)` in the binary);
* non-zero `stc` immediate — refuted by `zoff`, which forces every store to `imm=0`, verified
  in the disassembly, and still fails.

Supporting rules, each earned:

* **Build the arm that would REFUTE, not the one that would confirm.**
* **Differential testing is cheap and board-free** — `k1200` returns the correct 4 under QEMU
  while failing on silicon. A passing reference implementation is evidence about *where* the
  fault is.
* **Read the primary source, not the parsed output** — an `ex_code` enum read off a garbled
  parse made 28 look like `ILLEGAL_OPERAND_VALUE`; the enum's own comments say `OUT_OF_BOUNDS`.
* **Report reachability, not just pass/fail** — R-16 biases *which constructs can be measured
  at all*, so "arm X fails and arm Y does not" is unsupportable unless Y actually entered.

## 4. Proposed structure (~110–130 lines)

    name: blocker-investigation
    description: minimise a hardware/compiler blocker to a one-variable pair and keep the
                 hypotheses honest; use when a failure is located but not explained, when a
                 candidate cause is about to be recorded, or when a repro is too big to iterate.

    0. When this applies / when it does NOT (a located, explained bug does not need it)
    1. Get a returning number            -> one line, cross-ref CLAUDE.md, do not restate
    2. MINIMISE (the bulk)
         two-directional squeeze; terminal artifact is a one-variable PAIR
         the three failure modes, each with its concrete incident
         verify the artifact by disassembly; sha256 the set; distinct sentinels
    3. HYPOTHESIS HYGIENE
         precondition check against THIS build; refute-don't-confirm; differential;
         primary source over parsed output
    4. VALIDITY                          -> cross-ref board-run, do not restate ordering
    5. LANDING IT
         retract loudly and in the doc; commit as you go; package the PAIR as the hand-off

## 5. Implementation steps

1. Write `.claude/skills/blocker-investigation/SKILL.md` to the structure above.
2. Grep it against `CLAUDE.md` §"Debugging a blocker" and `board-run` for duplicated sentences;
   replace any duplication with a one-line cross-reference.
3. Verify: frontmatter parses, every cited path/rung exists, zero occurrences of the console
   host, the token, or any `http(s)://` literal.
4. Confirm `git ls-files .claude/skills/` lists it — `.gitignore` has `/.claude/*` and needs the
   `!/.claude/skills/` negation (already present; a new skill was silently uncommitted once).
5. Add pointers: `CLAUDE.md` skills table, `ONBOARDING.md`, `SILICON-BLOCKER.md`.

## 6. Acceptance criteria

* An agent handed "this hangs on the board, find out why" produces a **one-variable pair**, not
  a smaller failing case alone.
* No sentence duplicated from `CLAUDE.md` or `board-run`.
* Every rule cites the incident that produced it — rules without a scar get skipped.

## 7. Risks

* **Duplication drift** — three documents describing one workflow will diverge. Mitigation: the
  skill owns minimisation + hygiene *only*, and cross-references the other two.
* **Over-generalisation** — these lessons come from one hardware blocker. Keep the wording
  specific to "a failure that reproduces but is not explained"; do not present it as a universal
  debugging method.
* **It may belong inside `board-run`.** If review finds §2 and §3 too thin to stand alone, fold
  them in as sections rather than creating a second skill that is mostly cross-references.
