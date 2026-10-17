# rtreestatic -- eab0e10304

Upstream fix `eab0e10304`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-3 (rtree arm) / sqlite-eab0e10304 -- rtree leaves a freed heap buffer bound
 * SQLITE_STATIC to a persistent cached statement.
 *
 * nodeNew() allocates the node and its payload as ONE block:
 *     pNode = sqlite3_malloc(sizeof(RtreeNode) + pRtree->iNodeSize);
 *     pNode->zData = (u8 *)&pNode[1];
 * and nodeWrite() binds that payload into the cached pWriteNode statement without
 * ever binding it back to NULL:
 *     sqlite3_bind_blob(p, 2, pNode->zData, pRtree->iNodeSize, SQLITE_STATIC);
 *     sqlite3_step(p);
 *     rc = sqlite3_reset(p);
 * sqlite3_reset() does not clear bindings, so when nodeRelease() does
 * sqlite3_free(pNode) the statement's parameter 2 dangles into the freed block.
 * Fixed 3.23.0 (16 days after 3.22.0 shipped) by adding sqlite3_bind_null(p, 2).
 *
 * This is the SECOND arm of the same defect; case_fts3_static_bind.c is the fts3 arm.
 * Both use the route the fix's own commit message names -- "obtain a pointer to the
 * persistent statement using sqlite3_next_stmt() and attempt to access the freed
 * buffer using sqlite3_expanded_sql() or similar" -- with the "or similar" variant,
 * re-STEPping the statement, because sqlite3_expanded_sql() is compiled out in this
 * port: -DSQLITE_OMIT_FLOATING_POINT=1 makes 3.22.0's sqliteInt.h also
 * `#define SQLITE_OMIT_TRACE 1`, so the function body is `return 0;`. See
 * case_fts3_static_bind.c for the full write-up of that.
 *
 * Host ASan oracle on 3.22.0 (page_size=1024, 120 rows), identical under
 * -DSQLITE_RTREE_INT_ONLY which this group uses:
 *   re-step  : READ of size 820, 40 bytes into a freed 864-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1   (upstream's route, for comparison)
 *     free  nodeRelease <- rtreeUpdate
 *     alloc nodeAcquire
 *   Offset 40 is exactly sizeof(RtreeNode) on the host, i.e. &pNode[1] == zData,
 *   which confirms the single-block layout is what is being read.
 *
 * WHY THIS ARM IS WORTH HAVING ALONGSIDE THE fts3 ONE: its consequence is far
 * sharper. The re-step writes the freed block's current contents over the node's
 * row in %_node, and a later full scan then simply LOSES the rows that lived in
 * that node -- quantitative, observable data loss with no memory-safety tool:
 *     full_scan_before=120 -> full_scan_after=71      (49 rows gone)
 *     node data: same length 820, sum changed, first4 0x00000031 -> 0x00000000
 * Note PRAGMA integrity_check returns ok here: it does not validate rtree shadow
 * tables, so the row count is the signal, not integrity_check.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```

## Review 2026-10-16 -- every verdict holds; the trigger was two programs and could not show its own consequence

### The four readings

| reading | result |
|---|---|
| host ASan | `heap-use-after-free`, READ of size 820, 40 bytes into a freed 864-byte region, `sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step`; freed by `nodeRelease <- rtreeUpdate`, allocated by `nodeAcquire`. **2026-10-11 bundle, taken on the pre-unification trigger**; not re-run, this host has no x86 ASan runtime |
| virtual-malloc | **MISSED**, `RETURNED`, controls as declared |
| virtual-nested-pools | **CAUGHT**, `cause=25 pc=0x3f848020f8` offset 204 in `memcpy`, fault address `0x3f9af79980` (inside the arena), controls as declared |
| cheribsd-revocation | **MISSED**, `RETURNED rc=0`, positive control trapping, revocation knobs 1/0 |

No verdict changes and no count moves. What changes is the trigger, two defects inside it, and
the attribution.

### The trigger was two different programs

`ports/sqlite/cheribsd/cases/case_rtree_static_bind.c` carried a documented deviation: before
the re-step it asked memsys5 for 16 blocks of node size and filled them with `0x5A`, because on
purecap the freed block was not being reused and the copy was a no-op. The corpus `case.c` had
no such step -- and measured on this platform, it could not demonstrate its own consequence
either:

```
rtreestatic VERDICT len_changed=0 sum_changed=0 first4_changed=0 rows_lost=0
rtreestatic note node data unchanged; the read was silent and the freed block had not been
             reused yet
```

That is the corpus trigger on virtual-malloc, 2026-10-16. The stale read still happened -- the
Sublet arm faults on it, and probe site 56 witnesses it -- but the bytes copied over the row
were the node's own, so the row was rewritten unchanged. The header's claim of
`120 -> 71, 49 rows gone` was a run in which the block happened to hold a partially valid node
image.

The reclaim step is now part of the corpus `case.c`, and the CheriBSD copy is that file **byte
for byte**, as case 02's was made to be. Neither reason `ports/sqlite/cheribsd/port.json` gives
for the two case sets differing applies here: this case opens `:memory:` and needs no database
file, and the step is part of the trigger rather than probe instrumentation. On the unified
trigger all three arms show the same consequence:

```
probe_allocs=16 size=820 filled with 0x5A
restep rc=101
after present=1 len=820 first4=1515870810          (0x5A5A5A5A)
node_rows_after=6 scan_after=ERROR(query failed)
VERDICT len_changed=0 sum_changed=1 first4_changed=1 marker_in_row=1
        rows_lost=ALL(scan failed), was 120
integrity rc=0
```

The failure mode is not quantitative: what the scan does depends on the bytes that happen to be
in the block, and here the whole table stops being readable while `PRAGMA integrity_check`
still returns ok, because it does not validate rtree shadow tables. The signal is that the row
changed and the scan no longer returns the table.

### Two defects in that step, fixed

1. **The request size was hardcoded as `page_size - 64` = 960.** `iNodeSize` is 820 here:
   `getNodeSize` clamps it so that at most `RTREE_MAXCELLS` = 51 cells fit, `4 + 51*16 = 820`.
   The old value worked only because memsys5 rounds 960 and `sizeof(RtreeNode) + 820` into the
   same power-of-two size class -- the probe's `req=1024` on the Capstone arm is that class. The
   size is now taken from the node payload length the case has just read out of `%_node`.
2. **`recovered_freed_node` was a check that could not fire.** It compared a fresh allocation's
   first four bytes against the node's first four, but memsys5 has already written
   `Mem5Link{next,prev}` over the freed block's first eight bytes, so those bytes are a freelist
   index and never the node image. It printed `no` on runs where the marker did reach the row.
   Recovery is now reported by `marker_in_row` in the VERDICT line, which is the check that can
   fire: `0x5A5A5A5A` in the `%_node` row can only have come from those blocks.

### The attribution is now pointer identity

`fault_sites` is `["sqlite3VdbeSerialPut", "memcpy"]` and `fault_sites_why` admitted that any
`memcpy` under the statement would count, and that it was declared after the runs. Probe site 56
puts a hook at the `SQLITE_STATIC` bind in `nodeWrite`, at the free in `nodeRelease`, and at
`if( len>0 ) memcpy(buf, pMem->z, len)` in `sqlite3VdbeSerialPut`, the last filtered to
`len>400` so that the node payload is separable from the hot function's ordinary traffic. On
CheriBSD purecap:

```
BIND    stmt=0x216a00 zData=0x21e440 n=820 node=0x21e400
FREE    node=0x21e400 zData=0x21e440 (binding still points here)
rtreestatic before re-step
SERIAL  z=0x21e440 len=820 first4=0x5a5a5a5a
```

One pointer value at all three hooks. The content is an independent second argument:
`0x5A5A5A5A` cannot have been in that block before the free. So the `memcpy` the Sublet arm
faults in is the one copying out of the freed node payload, not some other `memcpy` under the
statement.

### What the address judgement does here, and why it is not the evidence

It gets this case wrong in **both** allocator states, which is worth recording because it is the
fourth instance of that theme in this corpus:

* **virtual-malloc**: the block is still free at the read, so the probe reports
  `kind=UAF off=64 n=820 req=1024 blk=... ca=blk+64`. Correct, and `off=64` is exactly
  `sizeof(RtreeNode)` under the 16-byte-pointer layout, which independently confirms
  `zData == &pNode[1]`.
* **CheriBSD**: the block has been reclaimed by the trigger's own `0x5A` allocation, and a read
  of 820 bytes from offset 64 runs past that allocation's 820 requested bytes, so the probe
  reports `kind=OOB`. That is an artifact of the request size, not an over-read -- the same
  misclassification case 09 had to correct. Before the request size was fixed it reported
  nothing at all: `hits=197 witnesses=0`, which is what the old `status` line meant by "reaches
  the defect site but cannot show this access was the stale one".

The witness does corroborate identity from the other side: its `blk=0x21e400` is the node
pointer the bind and free hooks recorded.

### Records corrected

* `case.c` carries the reclaim step; the CheriBSD copy is it byte for byte.
* `fidelity` declares the part of the trigger that is ours, and why.
* `fault_sites_why` records the identity evidence in place of "any memcpy would count".
* The three physical rows in `results/matrix.tsv` are marked: the two `not-detected` ones because
  they were measured on the earlier trigger and are unchanged as verdicts, and the `sublet` one
  because it is a kernel-pc halt, so the Sublet cell rests on the virtual arm's `cause 25`.
* `run-probe-cheri.sh`'s SITE map entry for `rtreestatic` moves from the arm's old site 4 to 56,
  the shared one.
