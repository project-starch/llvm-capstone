# blobwrite -- 8504d37b99

Upstream fix `8504d37b99`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row20 / sqlite-8504d37b99 -- sqlite3_blob_write()/read reads a released page
 * through an invalidated cursor. blobReadWrite() (vdbeblob.c:372) calls
 * sqlite3BtreeIntegerKey() (420) with no cursor-valid check, then the preupdate
 * hook (421) runs with that key. If the blob cursor was saved (another writer
 * modified/deleted the row), the page was released but pCur->pPage left set, so
 * the cell is parsed from a released page. Needs -DSQLITE_ENABLE_PREUPDATE_HOOK.
 * Fixed 3.51.0 (reseek first, skip hook if it cannot restore).
 *
 * CONTROL arm: open a blob, invalidate its cursor via another statement's write,
 * then blob_write -> IntegerKey on the released page. On unprotected Capstone the
 * released-page read is not caught, so it returns and the domain RETURNS.
```

## Not a memory-safety defect (reviewed 2026-10-15)

The trigger reaches the bug condition and no trigger can make it an invalid access. Both halves
are now measured rather than argued, by probe sites 40 and 41 in `shared/probe/`.

**The condition is reached.** Site 40 sits at the `sqlite3BtreeIntegerKey(p->pCsr)` whose result
the preupdate hook is given, and prints `eState=3 (CURSOR_REQUIRESEEK) iPage=-1 info.nSize=69`:
exactly the state upstream `8504d37b99` describes.

**No page is read through the invalid cursor.** Site 41 sits in `getCellInfo` and prints, at
every call, the branch taken and the cursor's state. Six calls parse a page, all at `eState=1`
(`CURSOR_VALID`) -- ordinary work on pinned pages. The one call at `eState=3` takes the cached
branch: `info.nSize=69` is still set, so it returns the cursor's cached cell info and the
unpinned page is never touched.

**That is structural, not a property of this trigger.** A cursor reaches `CURSOR_REQUIRESEEK`
only through `saveCursorPosition`, which calls `saveCursorKey` *first*; for an integer-key table
that is `pCur->nKey = sqlite3BtreeIntegerKey(pCur)`, which populates `info.nSize` through
`getCellInfo`. Only then does it release the pages and set the state. It clears
`BTCF_ValidNKey|BTCF_ValidOvfl|BTCF_AtLast` in `curFlags` and never `info.nSize`. So a
`REQUIRESEEK` integer-key cursor always carries valid cached cell info. The operations that do
clear `info.nSize` -- `moveToChild`, `moveToParent`, `moveToRoot`, `sqlite3BtreeMovetoUnpacked`,
`sqlite3BtreeNext`, `sqlite3BtreePrevious`, `sqlite3BtreeInsert` -- all leave the cursor valid.
The two conditions are mutually exclusive, so rewriting the trigger cannot produce the read.

**What the defect is.** Upstream's own title: "Avoid invoking the preupdate hook from within
sqlite3_blob_write() if the cursor is already invalid." The hook receives a key from a cursor
that no longer points at the row. That is a correctness defect. The fix reseeks first and skips
the hook when the row has been deleted; it adds no bounds or lifetime check.

Host ASan is silent on this case, and all three arms report MISSED. Neither is a miss: there is
no invalid access to find. This case belongs with the five moved to `not-in-scope/` on 2026-10-06
-- defects that are neither spatial nor temporal -- rather than in any arm's denominator.
Proposed for withdrawal when the review of this corpus ends, with case 01 and any other case
found not to carry a memory-safety defect, so the corpus is renumbered once.
