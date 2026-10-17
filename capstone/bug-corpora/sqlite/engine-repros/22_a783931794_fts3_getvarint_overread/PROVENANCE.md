# a783931794_0 -- a783931794

Upstream fix `a783931794`. Collected in round R3 (backport filter).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
a783931794_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Collected by the backport filter: upstream fix a783931794 removes code that is still
 * present in 3.22.0 (or guards a function 3.22.0 has unguarded), and its own regression
 * test ships the triggering database as a hex dump. Host ASan on stock 3.22.0:
 *
 *     heap-buffer-overflow in sqlite3Fts3GetVarint
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory FILESYSTEM.
 * ext/misc/memvfs.c cannot be used here: it serves only the main database, so the pager
 * cannot create a rollback journal, and it takes its buffer as an INTEGER in the URI --
 * a pointer forged from an integer, which on a capability machine is untagged and faults
 * on first use.
 *
 * STATEMENT BY STATEMENT, deliberately. Handing the whole script to one sqlite3_exec()
 * stops at the first error, and these scripts routinely take SQLITE_CORRUPT on an early
 * statement while the defect is reached by a later one. Screening the whole script in one
 * call scored 3 of 17; per statement it scores 10 of 17, matching a real file exactly.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the statement that
 * should reach the defect means SQLite rejected the image first -- a real negative, not a
 * silent pass.
```

## Review 2026-10-16 -- holds in full, and host ASan could be re-run this time

### The four readings

| reading | result |
|---|---|
| host ASan | **re-run**: `heap-buffer-overflow`, 0 bytes after an 8-byte region, `sqlite3Fts3GetVarint <- fts3GetDeltaVarint <- fts3PoslistPhraseMerge <- fts3EvalIncrPhraseNext`, allocated by `sqlite3_malloc <- fts3EvalIncrPhraseNext` -- identical to the 2026-10-11 bundle |
| virtual-malloc | **MISSED**, `RETURNED`, controls as declared |
| virtual-nested-pools | **CAUGHT**, `cause=28` offset 1048 in `sqlite3Fts3GetVarint`, fault address `0x3f96bd0ee8` |
| cheribsd-revocation | **MISSED**, `RETURNED rc=0`, every statement `rc=0`, over-read witnessed |

`cause 28` is CAP_OOB, the **bounds** check -- the right code for a spatial over-read, where the
temporal cases turn on `cause 25`. No verdict changes and no count moves.

### The defect has no bound to check at all

```c
static void fts3GetDeltaVarint(char **pp, sqlite3_int64 *pVal){
  sqlite3_int64 iVal;
  *pp += sqlite3Fts3GetVarint(*pp, &iVal);
  *pVal += iVal;
}
```

There is no end pointer in scope: the caller is expected to have stopped first.
`fts3EvalIncrPhraseNext` allocates the buffer that is walked --
`char *aDoclist = sqlite3_malloc(nByte+1)` -- and passes it to `fts3PoslistPhraseMerge` as `pR`.

### The probe gives allocation-and-bound identity

Site 62 prints the buffer at the allocation and hands the decoder's pointer to the address
judgement at the read, so the witness names the same block:

```
LB_OBJECT site=62 ALLOC   aDoclist=0x2433c0 bytes=2
LB_WITNESS site=62 kind=OOB off=2 n=1 req=2 blk=0x2433c0 tag=1 cb=0x204a00+262144 ca=0x2433c2
LB_SUMMARY site=62 hits=14 witnesses=1 first_witness_at_hit=2
```

`ca = blk + 2` on a 2-byte request: exactly one byte past the end, on the block the ALLOC hook
had just recorded. **`kind=OOB` is the correct classification here**, unlike case 25 (withdrawn, campaign numbering) where the
same judgement reported `kind=UAF` because the over-read crossed into a freed neighbour. The
witnessed allocation is 2 bytes where host ASan reports an 8-byte region -- the case allocates
several and the two builds witness different ones; same line, same defect.

### The quietest case in the corpus

On CheriBSD every statement returns `rc=0` and nothing downstream complains: the over-read feeds
a varint whose value is simply wrong, and no wrong answer ever surfaces. Cases 21, 23 and 24
each produce a visible wrong result -- a rewritten row, a failed scan, a malformed-image error --
and this one produces none. Worth keeping as the other end of that range: a defect that is
invisible without a memory-safety mechanism in the strongest sense.

Why both non-Sublet arms are silent is case 19's reason: what is read is **bytes**, inside the
arena, through a capability with `tag=1` and whole-arena bounds, and no pointer is forged, so
there is nothing for the baseline tag check to fire on. Sublet rejects it because the
sub-allocation's **bounds** are what it checks.

## Re-measured 2026-10-17: one arena on every arm, every arm from the corpus source

Two things were wrong with the earlier spatial readings and both are fixed here. The arms did not
share an arena -- the virtual images were built at the 1 MiB header default, CheriBSD at 262144 --
and on an arena-granular arm the arena IS the bound, so the two were comparing arenas rather than
mechanisms. And this arm compiled its own copy of the case from `ports/sqlite/cheribsd/cases/`
rather than the corpus `case.c`. Both are closed: the case declares `arena_bytes` (1048576 here),
all three builders read it, and `build-corpus-cheri.sh` now takes the case program from the corpus.

| arm | verdict | evidence |
| --- | --- | --- |
| virtual-malloc | **MISSED** | both controls as declared; reached and silent, by the Sublet catch below on the same build pair |
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3fa821d5b4` in `sqlite3Fts3GetVarint`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 25 void: probe site 62 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=14 witnesses=1`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
