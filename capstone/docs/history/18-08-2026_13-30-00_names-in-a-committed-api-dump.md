# Personal names reached a committed file, and the gate could not have caught it

Found 2026-08-18 while scanning for names before writing up an unrelated paper --
not by the gate, and not by review.

## What happened

`benchmarks/micropython/spatial-corpus/github-bodies.json` stores the raw GitHub
API response for 27 issue bodies. It exists for a good reason: `is_spatial` in the
spatial corpus was set by READING each report rather than its title, after the
temporal audit had to undo fourteen rows whose class had been read off titles. A
reader has to be able to check that, so the bodies are stored verbatim.

Several of those reports open by introducing their authors, and one carries an
explicit credit line. So three commits, already pushed, named four real people and
their affiliation. That breaks the repository's absolute rule against personal
names in committed content.

## What was done

Twelve identifying lines removed -- self-introductions and credit lines -- and a
redaction note put in their place. Every PoC, `file:line`, quoted source and ASan
trace is untouched, because that is the whole reason the file is committed.

Then, on an explicit decision, the three commits were rewritten with
`git filter-branch --tree-filter --prune-empty` and the branch force-pushed.
**Other lanes must re-sync `nested-allocators`.** The rewrite was verified with a
positive control: the same check was run against the pre-rewrite commit objects,
where it fires on all four name patterns, and against the rewritten ones, where it
does not. The first attempt at that control probed the tip of the backup ref, which
was already clean because the redaction commit preceded it -- a check that reports
"clean" having looked at the wrong object.

One deliberate exception remains: the tool URL `github.com/<org>/CMASan` in
`ref/cmasan-sp25-and-our-corpora.md`. That is an artifact link to an open-source
tool, the same category as a dependency URL, and it names an organisation rather
than a person.

## Why the gate did not catch it

Two independent reasons, and the second is the one that generalises.

1. `precommit-scan.sh` has printed the same warning on every run this session: the
   out-of-repo denylist file it expects is absent, so the exact-name check was
   SKIPPED and only the name-independent heuristics ran. I read past that warning
   perhaps thirty times.

2. Even with the file present it would not have helped. The denylist holds the
   project lead and collaborators. These are arbitrary third parties introducing
   themselves in text pasted in from outside, and no list of known names can
   anticipate that.

So the missing capability is not a longer denylist. It is a rule for **text that
enters the repository from outside**: an API dump, a pasted issue body, a quoted
mailing-list post. Those carry identities nobody chose to write.

## What would have caught it

Cheapest first:

- Make the missing denylist an ERROR, not a warning. A gate whose main check is
  disabled should refuse to pass, exactly as this project already insists that "no
  data" must not render as a zero result. That single change turns thirty ignorable
  warnings into one blocking one.
- Add a shape-based rule for imported text, independent of any name list: a stored
  API response or pasted body should be run through a redaction pass before it is
  staged. `Hi all, I'm ...`, `Credits:`, `contribution:` and `Reported by` cover
  what was actually present here.

Neither is implemented; this note records the finding, not a fix.

## The one good thing

The incident produced the positive control `precommit-scan.sh` never had. Its first
run on the redaction commit BLOCKED, because the message quoted the denylist's
own path and that path contains a banned substring. A gate that has never refused
anything is unproven; this one is now known to fire.
