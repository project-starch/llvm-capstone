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
