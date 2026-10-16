# mem5design -- mem5-design

Upstream fix `mem5-design`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row25 / sqlite-mem5-design -- MEMSYS5 stores freelist metadata INSIDE freed
 * blocks (src/mem5.c: memsys5Link/Unlink write Mem5Link{next,prev} into the freed
 * block; CTRL_FREE is out-of-band in aCtrl[]). A dangling pointer into a block
 * freed back to the pool then reads/overwrites allocator metadata. Design-inherent,
 * never patched (paper: PoisonCap). Present in 3.22.0 (mem5.c:143 MEM5LINK).
 *
 * CONTROL arm: allocate two adjacent blocks, keep a dangling pointer to the first,
 * free it, force a neighbour free so memsys5 links/coalesces (writing in-band
 * metadata), then READ through the dangling pointer. The bytes read are now
 * allocator freelist indices, not the caller's data -- the aliasing the paper
 * names. On unprotected Capstone (whole pool is one allocation, invisible to ASan)
 * this READ SUCCEEDS and the domain RETURNS. Sublet would revoke the freed block.
```

## Review 2026-10-16 -- measured on all four readings, then proposed for withdrawal

All four readings were taken on the rebuilt platform and they agree with each other:

| reading | result |
|---|---|
| host ASan | use-after-free reported -- **but in a configuration without memsys5** |
| virtual-malloc | MISSED, `RETURNED rc=0`; the log carries the demonstration: `mem5design before=2779054080 after=4294967295 (in-band metadata overwrote caller data)` |
| virtual-nested-pools | CAUGHT, `cause=25 pc=0x3f8981235c in run_case` |
| cheribsd-revocation | MISSED, `RETURNED rc=0`, the same before/after values, positive control trapping (`In-address space security exception`) |

That is the corpus's premise stated as a number: the stale read succeeds and returns
freelist indices, 4294967295, out of a block the caller had left holding 2779054080, and the
only arm that refuses the read is the one that bounds and revokes sub-allocations.

**Why the case is withdrawn anyway.** The trigger is

```c
volatile unsigned int *p = (unsigned int *)sqlite3_malloc(128);
void *q = sqlite3_malloc(128);
...
sqlite3_free((void *)p);
sqlite3_free(q);          /* coalesce writes Mem5Link through p's body */
after = *p;               /* dangling read */
```

which is the construction the arms already ship as their positive control,
`controls/control_uaf_mem5.c`, whose own header labels it `CONTROL, not a case`:
`sqlite3_malloc(64)`, `sqlite3_free(p)`, then a read through `p`. This case adds one
neighbour free so the in-band overwrite is observable in the value, and nothing else.
Both run on every Sublet image, so both report the same event, and counting this directory
as a case adds 1 to the denominator and 1 to Sublet's numerator for a reading the control
already provides on its own. The sharpest form of that: the control's own header states what
`tools/arms.json` requires of it -- "memsys5 stock hands out offsets inside one arena, so it
COMPLETES; memsys5 on its Sublet port issues each block as its own capability and retires it
on free, so it FAULTS" -- and that is, line for line, what this case measured on 2026-10-16.
The case's result is the control's declared expected behaviour.

Three facts make a rewritten trigger impossible rather than merely unnecessary:

1. `upstream_fix` is `mem5-design`. There is no SQLite defect and no fix to cite -- the
   in-band freelist is how `memsys5` is built, in 3.22.0 and in every other release. Every
   other case in this corpus names a commit or a release that removed its defect.
2. `consumer` is `run_case`: the dangling read is in our own trigger, not in SQLite. That is
   why this is the one case whose probe site lives in the case source rather than in
   `sqlite3.c`, and it means no upstream input exists that could drive the access instead.
3. The host-ASan reading on record was taken in a build **without** `memsys5`, where
   `sqlite3_malloc` is the system allocator. In that configuration it is an ordinary `malloc`
   use-after-free, so the oracle never witnesses the in-band-metadata aliasing the case is
   named for.

One record is corrected rather than altered: the physical `sublet: detected` row in
`results/matrix.tsv` is a genuine delivered fault in the domain, `cause 24` at
`pc=0x101da6878`, and not one of the 28 kernel-pc halts -- but it is the control's event, and
the row is marked there to say so.

**Keep the directory and the source.** The demonstration is worth having in the corpus and
is cited in the paper's premise; it is excluded from the counted corpus, from the three-arm
table and from the per-axis denominators.

The withdrawal joins case 01 (not reached), case 04 (not a memory-safety defect) and case 09
(duplicate of case 10) in the single renumber at the end of this review. Four withdrawals,
four different reasons, and this is the fourth kind: not a trigger that misses, not a defect
out of scope, not a second row for one defect, but the corpus's own control counted as a
finding. With it, 33 becomes 29.
