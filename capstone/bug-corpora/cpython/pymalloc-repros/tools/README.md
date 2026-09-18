# The two scripts behind the inventory's numbers

Every count in `docs/ref/cpython-pymalloc-defects.md` comes from one of these.
They are committed so the numbers can be re-derived rather than trusted.

    triage.py       44 live -> the three axes -> 32 in the pymalloc bucket
    apply-test.py   of the 11 group-B cases in that bucket, which are really live

## Inputs

Both take a CPython clone; `apply-test.py` also takes a **pristine worktree of
the pin**, which it never writes to (`git apply --check` only):

    git clone --bare https://github.com/python/cpython $REPO
    git -C $REPO worktree add --detach $TREE v3.13.7

    CPYTHON_REPO=$REPO CPYTHON_TREE=$TREE python3 apply-test.py

The defaults are where the 2026-09-18 run had them. A clone missing the `main`
or `3.13` refs silently yields an empty group rather than an error — if a count
comes back 0, check the refs before believing it.

## Why apply-test.py runs group A as well

Group A's answer is known independently: those fixes were cherry-picked onto the
3.13 branch at or just after our tag, so they should apply. Running them through
the same test is the positive control. The 2026-09-18 run: group A 21/21, group B
1/11 — the check fires, and the two groups separate.

**It is a one-directional test.** "Applies" proves the pre-fix code is there.
"Does not apply" proves nothing, because the surrounding code may merely have
moved: `gh-145244` fails the test and the defect is in `Modules/_json.c:1621` of
the pinned tree all the same. Read the failures; do not count them.
