# PostgreSQL 17.5 engine defects, reached through SQL

19 defects in PostgreSQL 17.5 and its contrib extensions, each a `trigger.sql`
run against a real server. Sibling of [`../mmgr-repros`](../mmgr-repros), which
holds 8 upstream memory-manager defects pinned at 17.0 and reduced to C
programs; these are not reducible that way, so they use the `script-trigger`
schema described in [`../../SCHEMA.md`](../../SCHEMA.md).

## Classes

| class | cases |
|---|---|
| invariant | 9, 10, 11, 12, 14 |
| integer | 4, 5, 6, 7 |
| spatial | 1, 2, 3 |
| spatial/oob-read | 15, 16 |
| type-confusion | 17, 18 |
| null-deref | 8 |
| wrong-answer | 13 |
| authz | 19 |

**Six of the nineteen are what a sanitizer can see.** The rest are integer
wraparound, broken invariants, type confusion, an authorization failure and a
wrong answer -- defects with no memory error to detect. A corpus filtered to
sanitizer-visible defects would keep six of these and call the other thirteen
clean, which is the reason this corpus records the class per case.

## Arms

| arm | state |
|---|---|
| `spatial` | not run. Needs the Capstone application VM -- real postgres in a domain -- which is built and has never been run |
| `sublet` | not run, same reason |
| `cheribsd-revocation` | measured for all 19 |

CheriBSD runs each case **twice**: the plain purecap build gives the verdict,
and a separate `--enable-cassert` purecap build supplies the reachability
witness through PostgreSQL's own `Assert()` and its `MEMORY_CONTEXT_CHECKING`
chunk sentinel. The two must be separate runs, because `Assert()` aborts and so
changes control flow -- a build that aborts can never be the one whose silence
is being reported. For the wrong-answer and integer cases the witness is instead
a differential directive in the trigger itself:

    -- EXPECT-ERRORS: N     a correct build rejects N statements
    -- EXPECT-ABSENT: <re>  a correct build can never print this

## What these results do not say

Three cases (12, 14, 18) need a postmaster and cannot run under the single-user
backend, so their rows are `not-run`, not `silent`. Case 13's `EXPECT-ABSENT`
directive was written from its own run output: it records what was seen rather
than testing an independent expectation, so it will always fire on this arm.
Its `case.json` says so, and `audit-evidence.py` flags the overlap.

Our own constructed memory-context demonstrators are deliberately not here.
They are this project's fixtures, and `../../README.md` places those under
`ports/*/security-tests` rather than in bug material.
