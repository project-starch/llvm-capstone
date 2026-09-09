# Proposal: `precommit-scan` should WARN on removed and context lines, not BLOCK

**Status: proposal, not applied. This is a change to a release gate, so it is the project lead's.**
Written 2026-09-10 by the board lane after hitting the problem three times in one session, twice
while trying to record the fix for the first instance.

## The problem, stated as what it costs

`precommit-scan.sh` appends `git diff` output wholesale (`:58`, `:60`, `:67`) and greps the result.
(This document deliberately does not reproduce the credential alternation itself: writing it out is
enough to block the commit that adds this file, which is the problem in miniature.)
It therefore treats a **removed** line, and a **context** line, exactly like an **added** one.

The consequence is that a committed false positive is permanently uncleanable:

- the commit that **deletes** the offending text is blocked by the text it is deleting;
- the commit that **moves** the entry elsewhere is blocked twice, once on the deletion and once on the
  addition;
- editing anywhere **within three lines** of it is blocked, because the phrase appears as diff context;
- so is rewording the phrase, which is itself a deletion.

That is not hypothetical. `ISSUES.md`'s C-4 entry introduces a proposed status line using the
registry's own word for such a word, followed by a colon and then the value — a shape the credential
pattern at `:163` matches by construction. It is a false positive by inspection: the value that
follows is `FIXED 2026-07-28`, a date, not a credential. Because of it, **C-4 cannot be moved to `ISSUES-ARCHIVE.md` even though its status
is now final**, which is a registry rule the gate is silently preventing us from following. The entry
now carries a note saying so, which is the wrong place for that information to live.

## Why this is a scope correction and not a weakened pattern

The distinction matters, because CLAUDE.md is explicit that patterns must not be weakened to make a
scan pass, and that rule is right.

**Every pattern stays exactly as it is.** What changes is only which lines they BLOCK on:

| line kind | today | proposed |
|---|---|---|
| commit message | BLOCK | BLOCK |
| untracked file contents | BLOCK | BLOCK |
| added (`+`) diff lines | BLOCK | BLOCK |
| context (` `) diff lines | BLOCK | WARN |
| removed (`-`) diff lines | BLOCK | WARN |

The argument that this loses nothing:

1. **A secret being introduced always appears as an added line.** Introduction is still blocked.
2. **Moving a secret between files still blocks**, because the destination is an added line.
3. **A removed line is content leaving the tree.** It cannot introduce anything.
4. **If a removed line carries a real credential, it is already in history**, and a blocked commit does
   not undo that. The remedy there is rotation — which this project already knows, having deferred a
   full history rewrite for exactly that reason and recorded that the real fix is to rotate.
5. **A context line is unchanged pre-existing content.** Blocking on it means the gate's verdict
   depends on where in a file you happened to edit, which is not a property anyone wants in a gate.

Nothing above is an argument that removed lines are uninteresting — only that they are not a reason
to refuse a commit. They should still be scanned and still be reported.

## The change

Two edits to `capstone/tests/precommit-scan.sh`. After the block that assembles `$TMP` (i.e. after the
`if [[ -n "$RANGE" ]] … else … fi` at `:56-72`) and before the checks:

```bash
# Removed and context lines are scanned but do not BLOCK -- see
# docs/plans/precommit-scan-removed-lines-proposal.md. Every pattern is unchanged; only the set of
# lines that can FAIL the scan is narrowed, and a secret being introduced is always a `+` line.
DEL="$(mktemp)"; trap 'rm -f "$TMP" "$DEL"' EXIT
KEPT="$(mktemp)"
grep -E '^[-@ ]' "$TMP" | grep -vE '^\+\+\+|^---' > "$DEL" 2>/dev/null || true
grep -vE '^[-@ ]' "$TMP" > "$KEPT" 2>/dev/null || true
mv -f "$KEPT" "$TMP"
```

and immediately before the final verdict block at `:190`:

```bash
if [[ -s "$DEL" ]]; then
  # Reuse the SAME alternation as the blocking secret checks at :163 and :167 -- do not retype it
  # here, or the two will drift and the warning will stop covering what the gate covers. Factor it
  # into a variable (e.g. SECRET_RE) at the top of the script and reference it in all three places.
  DM=$(grep -inE "$SECRET_RE" "$DEL" | grep -viF '<FPGA-CONSOLE-URL>' || true)
  if [[ -n "$DM" ]]; then
    echo "=========================================================="
    echo "NOTE (not blocking): REMOVED or CONTEXT lines match a pattern that BLOCKS when added."
    echo "Content leaving the tree cannot introduce a secret. If any of these is a REAL"
    echo "credential it is already in history and needs ROTATION, which refusing this"
    echo "deletion would not achieve. Confirm by eye:"
    echo "$DM" | head -20
  fi
fi
```

**Note the `$TMP` split keeps the message and untracked content**, because those lines do not start
with `-`, `@` or a space in the way diff body lines do. Verify that rather than assuming it — see the
controls below.

## Controls this change MUST pass before it is trusted

A gate change that has not been shown to still fire is worth nothing. Each of these should be run and
recorded:

1. **An ADDED line carrying a fake token still BLOCKS** (exit 1). This is the one that matters.
2. **A REMOVED line carrying the same fake token WARNS and exits 0.**
3. **The same fake token in the COMMIT MESSAGE still BLOCKS.**
4. **The same fake token in a new UNTRACKED file still BLOCKS** — this is the 2026-08-18 case the
   untracked scan exists for, and the split must not drop it.
5. **A real-person name in an added line still BLOCKS**, and in a removed line warns — the name rules
   are the reason this gate exists at all, and they must be checked separately from the secret ones.
6. **A MOVE — the token deleted from file A and added to file B in one commit — still BLOCKS**, on the
   addition.

Use a fake credential that matches the pattern but is obviously not real, and never a value from
`~/.claude-c/secrets/`.

## What I did instead, and why the record is worse for it

Rather than apply this unilaterally, the affected entries carry notes explaining that they are
final-but-resident. That is the honest option but it puts tooling apologetics inside issue entries,
where a reader looking for the state of a hardware defect has to skip a paragraph about a grep. If the
change is accepted, those notes should be deleted in the same commit — which, once removals stop
blocking, will finally be possible.
