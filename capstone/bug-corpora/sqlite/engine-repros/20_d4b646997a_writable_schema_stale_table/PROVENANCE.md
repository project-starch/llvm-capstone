# wschema -- d4b646997a

Upstream fix `d4b646997a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row8 / sqlite-d4b646997a -- sqlite3StartTable() records the table in
 * pSchema->pSeqTab BEFORE the table is finalized:
 *
 *     if( !pParse->nested && strcmp(zName, "sqlite_sequence")==0 ){
 *       pTable->pSchema->pSeqTab = pTable;
 *     }
 *
 * If the CREATE then fails, sqlite3Prepare's cleanup does
 * sqlite3DeleteTable(db, sParse.pNewTable) and the Table is freed, but
 * pSchema->pSeqTab still points at it. Later AUTOINCREMENT codegen dereferences it:
 *
 *     autoIncBegin: sqlite3OpenTable(pParse, 0, p->iDb, pDb->pSchema->pSeqTab, OP_OpenRead)
 *
 * Fixed 3.36.0. Core.
 *
 * Reaching it needs an explicit "CREATE TABLE sqlite_sequence", which
 * sqlite3CheckObjectName normally rejects -- but only when
 * (db->flags & SQLITE_WriteSchema)==0, so PRAGMA writable_schema=1 is the opener.
 * The statement must then fail AFTER StartTable; a duplicate column name does it
 * (sqlite3AddColumn -> "duplicate column name"), which runs after the pSeqTab store.
 *
 * NOTE this does NOT need a corrupt database file, contrary to the first assessment
 * of this row: writable_schema plus a post-StartTable error is sufficient, and the
 * domain is :memory:-only.
 * NOTE -DSQLITE_DQS=0 -> SQL string literals must be single-quoted.
```

## Review 2026-10-16 -- the published verdict was right by accident, and the bundle's was wrong

### The four readings

| reading | result |
|---|---|
| host ASan | the 2026-10-11 bundle's reading; not re-run, this host has no x86 ASan runtime |
| virtual-malloc | `cause=24 pc=0x3fbb1568e0` offset **648** in `sqlite3OpenTable`, fault address **`0xb0`**; scored CAUGHT by the judge |
| virtual-nested-pools | `cause=25 pc=0x3f90956b04` offset **268** in `sqlite3OpenTable`, fault address `0x3f90b6a120`, inside the arena |
| cheribsd-revocation | **MISSED**, `RETURNED rc=0`, positive control trapping; its `INSERT` then fails `rc=11`, *database disk image is malformed* |

### The defect, measured as a boolean

Probe site 59 hooks the store, the free and the read:

```
SEQTAB  pTable=0x3f81b74d70 recorded in pSchema->pSeqTab before the CREATE is finalized
FREE    pTable=0x3f81b74d70 still_pSeqTab=1
wschema before insert (autoIncBegin reads the freed pSeqTab)
OPEN    pTab=0x3f81b74d70 zName=0xffffffff00000079 tnum=-2118687248 tabFlags=0xce0c6828
        nCol=0 HasRowid=0 pIndex=0x0
```

`still_pSeqTab=1` is the defect: `sqlite3Prepare`'s cleanup deletes the Table while
`pSchema->pSeqTab` still points at it. The trigger reaches it on every arm -- `create-seq`
returns `rc=1 duplicate column name: a`, which is the post-`StartTable` failure the case needs.

### Base Capstone's CAUGHT is a downstream NULL dereference

The chain closes arithmetically:

1. `tabFlags` reads back `0xce0c6828` out of the dead block. `HasRowid(X)` is
   `((X)->tabFlags & TF_WithoutRowid)==0` with `TF_WithoutRowid` = `0x0020`, and
   `0x28 & 0x20` is nonzero -- so the garbage flags send the code down the **WITHOUT ROWID**
   branch. The probe prints `HasRowid=0`.
2. That branch calls `sqlite3PrimaryKeyIndex(pTab)`, which is
   `for(p=pTab->pIndex; p && !IsPrimaryKeyIndex(p); p=p->pNext){} return p;`. The stale
   `pIndex` is `0`, so it returns **NULL**.
3. `pPk->tnum` dereferences that NULL. `int tnum` sits after eleven pointers in `struct Index`,
   so under the 16-byte-pointer layout its offset is `11 * 16 = 176` = **`0xb0`** -- the
   measured fault address, exactly.

Every read **of** the dead object succeeded. The fault is on a NULL a helper returned because
the stale flags chose the wrong branch: the same class as cases 01, 03 and 09.

Two details worth keeping. The clobbered `zName` at offset 0 -- `0xffffffff00000079`, which is
memsys5's `Mem5Link{next=0x79,prev=-1}` -- is **never dereferenced** on either arm, because
`sqlite3TableLock` only stores the pointer; that is why the allocator metadata in it is harmless
here, unlike in cases 15 and 16 where the clobbered first member was followed. And the address
judgement is correct and sufficient on this case: `kind=UAF off=0 n=8 req=224` on the Table's
own block, the second such case after 23.

### Why CheriBSD completes, and what that says

On that arm site 59 prints the same read with **different leftover bytes**:

```
OPEN  pTab=0x211600 zName=0x100000001 tnum=0 tabFlags=0x0 nCol=0 HasRowid=1 pIndex=0x0
```

`tabFlags=0x0`, so `HasRowid` is 1, the rowid branch runs, and the only reads are `tnum` and
`nCol` -- both ints. Nothing is dereferenced, so nothing traps, and the damage surfaces as
`INSERT rc=11` instead.

**Same defect, same field, different leftover bytes, opposite outcomes on the two non-Sublet
arms.** That is the sharpest instance so far of what decides their readings on a nested
use-after-free. Sublet's does not move with it: `cause 25` at offset 268 is the first read of
the dead Table, and the lifetime of the sub-allocation is what it checks.

### The record this corrects is a disagreement between two record sets

The physical campaign recorded this arm as `completes`, which is what `results/matrix.tsv` and
the three-arm table carry, so **the published cell is correct**. The 2026-10-11 virtual bundle
scores it **CAUGHT**, because the cause-24 fault lands in `sqlite3OpenTable` and that is the
declared `fault_sites` entry. So the published verdict is right while the bundle this review
otherwise relies on is wrong, and the two had never been compared.

That is the reverse of case 16, where `matrix.tsv` said `detected` and the bundle had already
said `NO-READING:unattributed`. No count moves either way, but the base cell here is now right
on purpose rather than by luck, and both rows are marked.

`fault_sites` stays `sqlite3OpenTable`; `fault_sites_why` now records that a function-level
attribution cannot work for this case, because the defective read and the consequence of
letting it succeed are both inside that one function, and that the discriminators are the pc
offset and the fault address.

### Why the two arms saw different bytes, and why that settles what the fault means

`tabFlags` sits at offset **120**, and `Mem5Link{next,prev}` is eight bytes at offset 0, so the
allocator's own metadata cannot reach it. Its value at the read is therefore whatever the block
happens to hold: the Table's own last state if nothing has reused the block, or a new owner's
bytes if something has.

* **CheriBSD: not reused.** `tabFlags=0x0` is the Table's **own** value. `sqlite3StartTable`
  allocates with `sqlite3DbMallocZero`, and the CREATE fails in `sqlite3AddColumn` with
  *duplicate column name* before any flag is ever set, so zero is what was there. The
  neighbouring fields agree -- `tnum=0`, `nCol=0`, `pIndex=0x0`, all still zero -- and only the
  first eight bytes have changed, to `zName=0x100000001`, which is `Mem5Link`.
* **base: reused and partly overwritten.** `tabFlags=0xce0c6828` and `tnum=-2118687248` are
  somebody else's data.

The two arms are different builds with different feature flags, and the virtual arms are the
only ones with the lookaside on, so memsys5's free list is in a different state when the Table
is released. That is a build and timing difference, not a difference in what either mechanism
can check.

**The decisive form of that argument is inside the base arm itself.** The physical campaign
recorded this arm as `completes` -- that is the `·` in `matrix.tsv` and in the three-arm table --
while the virtual build of the same arm faults with `cause 24` at `0xb0`, in the 2026-10-11
bundle and again when re-measured on 2026-10-16. **One mechanism, one case, two builds, opposite
outcomes.** A detection capability does not flip with the build; a dependence on what the
allocator left in a struct field does.

So "base caught what CheriBSD missed" is not what happened. Base was led down a branch it should
not have taken by a stale flag word and then crashed on a NULL a helper returned. Had that field
been zero on base too, base would have completed exactly as CheriBSD did; had CheriBSD's block
been reused, CheriBSD would have faulted, which is what it does in cases 15, 16 and 18. For
either arm to count as a detection it would have to refuse the stale read itself -- which is what
Sublet does, at `sqlite3OpenTable+268`, on the first read of the dead Table, at a fault address
inside the arena, and that reading does not move with the allocator state.
