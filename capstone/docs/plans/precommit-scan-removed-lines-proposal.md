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

**Filter each `git diff` AT THE POINT IT IS APPENDED — do not post-process `$TMP`.**

My first draft of this patch split `$TMP` afterwards with `grep -vE '^[-@ ]'`, and that draft was
WRONG in a way that would have quietly reduced coverage: `$TMP` also holds the commit message and the
full text of every untracked file, and **an indented line in either of those starts with a space**. A
post-hoc split would have dropped every indented line of a brand-new file out of the blocking set —
which is precisely the 2026-08-18 case the untracked scan was added for, where a new `plans/` document
passed a scan that had never read a byte of it. A gate patch that silently narrows what is scanned is
worse than the problem it fixes, so the split has to know which lines came from a diff.

So: introduce `DEL` once, and route each of the three `git diff` invocations at `:58`, `:60` and `:67`
through a helper that sends removals and context to `DEL` and everything else to `$TMP`. The message
and the untracked `cat` continue to append to `$TMP` untouched.

```bash
# Removed/context diff lines are scanned but do not BLOCK -- see
# docs/plans/precommit-scan-removed-lines-proposal.md. Patterns are unchanged; only the set of lines
# that can FAIL is narrowed, and an INTRODUCED secret is always a `+` line. Applied per-diff, never to
# $TMP as a whole: $TMP also carries the commit message and untracked file contents, whose indented
# lines start with a space and must keep blocking.
DEL="$(mktemp)"; trap 'rm -f "$TMP" "$DEL"' EXIT
split_diff() {          # stdin = diff; blocking lines -> $TMP, informational -> $DEL
  awk -v del="$DEL" '
    /^\+\+\+|^---/ { print; next }                 # file headers: keep blocking
    /^[-@ ]/          { print > del; next }          # removals, hunk headers, context: warn only
                      { print }                      # additions and everything else: blocking
  ' >> "$TMP"
}
```

then replace each `git diff … >> "$TMP"` with `git diff … | split_diff`, and immediately before the
final verdict block at `:190`:

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

Control 4 below exists specifically to catch the mistake my first draft made. Run it.

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
