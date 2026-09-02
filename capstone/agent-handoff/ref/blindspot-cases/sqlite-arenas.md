# SQLite's internal arenas, read out of the source

*Evidence for the column that decides every row of `sqlite-bugs.csv`. Source
read: `sqlite-src-3530400` (SQLite 3.53.4, full source, not the amalgamation).
Every claim below carries a file:line you can re-read. Companion: `mruby.md`.*

## Why this file exists

The blind-spot thesis only holds if SQLite really does sub-allocate behind the
malloc boundary. That was assumed when `sqlite-bugs.csv` was seeded, and the
arena column was marked `-CANDIDATE` throughout. This file replaces the
assumption with source. Two arenas are now **verified**, and one component that
a call-site count put on the safe side turned out to be an arena as well.

## Arena 1: lookaside — verified

**One allocation for the whole pool.** `setupLookaside()` in `src/main.c`:

    pStart = sqlite3Malloc( szAlloc );          /* szAlloc = sz*cnt */

Default size, `src/global.c:220`:

    #define SQLITE_DEFAULT_LOOKASIDE 1200,100   /* 120KB of memory */

**Slots are carved by pointer arithmetic from that one pointer**, same function:

    p = (LookasideSlot*)pStart;
    for(i=0; i<nBig; i++){ ...; p = (LookasideSlot*)&((u8*)p)[sz]; }
    ...
    for(i=0; i<nSm;  i++){ ...; p = (LookasideSlot*)&((u8*)p)[LOOKASIDE_SMALL]; }

`LOOKASIDE_SMALL` is 128 (`src/sqliteInt.h:1625`); the pool is split into big
slots and small slots at `lookaside.pMiddle`.

**Allocation never reaches malloc.** `sqlite3DbMallocRawNN()`, `src/malloc.c:643`:
if `n <= db->lookaside.sz`, the result is popped off `pSmallFree` / `pSmallInit`
/ `pFree` / `pInit`.

**Free never reaches free().** `sqlite3DbNNFreeNN()`, `src/malloc.c:458ff`: if the
pointer lies inside the pool it is pushed back onto a freelist and the function
returns:

    pBuf->pNext = db->lookaside.pSmallFree;
    db->lookaside.pSmallFree = pBuf;
    return;

**SQLite has no CHERI awareness at all.** `grep -rli cheri src/` returns nothing,
so there is no `cheri_bounds_set`/`__builtin_cheri_bounds_set` narrowing anywhere
in the carve path.

**What lives there.** The core parse and query machinery, because it allocates
through the connection allocator: `Expr` (`src/expr.c:941,982,1050`), `Window`
(`src/window.c:1222,2385`), and the same pattern throughout `select.c`,
`build.c`, `vdbe.c`, `json.c`.

## Arena 2: the page cache — verified, and it was nearly missed

A count of allocator call sites puts `pcache1.c` on the "direct heap" side: it
calls `sqlite3Malloc`, never `sqlite3DbMalloc`. That reading is **wrong**, and
the file says so itself (`src/pcache1.c:60-78`), listing three page sources of
which the third is:

    (3)  PCache-local bulk allocation.
    ... a chunk of heap memory (defaulting to 100 pages worth) that is allocated
    when the page cache is created ... N pages worth of memory are allocated
    using a single sqlite3Malloc() call and that memory is used for the first N
    pages allocated.

`pcache1InitBulk()` does exactly that — `zBulk = pCache->pBulk = sqlite3Malloc(szBulk)`
followed by a carve loop that derives every page and its `pExtra` by arithmetic
(`pX->page.pExtra = (u8*)pX + ROUND8(sizeof(*pX))`). And `pcache1FreePage()`,
`src/pcache1.c:472-479`:

    if( p->isBulkLocal ){
      p->pNext = pCache->pFree;
      pCache->pFree = p;
    }

Same structure as lookaside: one malloc, arithmetic-derived children, free by
freelist push. Mode (2), `SQLITE_CONFIG_PAGECACHE`, is the same shape over a
caller-supplied buffer.

**Method note worth keeping:** the call-site count is a heuristic and it
mislabelled this component. Any arena verdict has to be read out of the
allocation path, not counted.

## The consequence, stated carefully

For an object inside either arena:

- **Temporal.** The memory is recycled without `free()` ever being called, so a
  revocation scheme that sweeps regions passed to the allocator has nothing to
  quarantine. A stale pointer to a freed `Expr`/`Window`/page stays usable, and
  the slot is handed straight to the next request.
- **Spatial.** The derived pointer's bounds are whatever the carve path gave it.
  Since SQLite narrows nothing, the expectation is that a slot capability spans
  the whole pool, making slot-to-slot access legal. **This half is under audit**
  — whether a purecap CHERI-LLVM build narrows bounds implicitly anywhere in
  this path is exactly the step most likely to be wrong, and the answer decides
  whether the spatial half of the claim survives. Treat the temporal half as the
  load-bearing one until that lands.

## The size-dependent case, which makes a clean experiment

`sqlite3DbRealloc()` (`src/malloc.c:699`) returns the same pointer while the
object still fits its slot. Once it does not, `dbReallocFinish()`
(`src/malloc.c:715`) splits:

    if( isLookaside(db, p) ){
      pNew = sqlite3DbMallocRawNN(db, n);
      memcpy(pNew, p, lookasideMallocSize(db, p));
      sqlite3DbFree(db, p);          /* -> freelist, NOT free() */
    }else{
      pNew = sqlite3Realloc(p, n);   /* -> real realloc, old region freed */
    }

So a stale-pointer-after-growth bug (the shape of CVE-2024-0232 in the JSON
parser, which allocates through the Db allocator — `src/json.c:1163`) has **two
different verdicts depending on payload size**: small enough to live in
lookaside and the old block goes to a freelist (blind); large enough to be on
the heap and the old block is genuinely freed (revocation can see it).

That is a directed experiment rather than an argument: one PoC, two payload
sizes, opposite outcomes predicted in advance.

## Component map

| Component | Allocator | Arena verdict |
|---|---|---|
| expr, window, select, build, vdbe, json | connection (`sqlite3Db*`) | **lookaside — blind** |
| pager / btree page content | pcache1 bulk | **page cache — blind** |
| fts3, fts5, rtree, session, rbu | `sqlite3_malloc` direct | ordinary CHERI coverage |
| pager/btree metadata | mixed | per-object, must be traced |

The shape of that table is the study's central claim in one line: **the bugs
CHERI catches in SQLite are the extension bugs, and the ones it misses are the
core-engine bugs** — not because the core is worse code, but because the core
allocates from a pool and the extensions call malloc.

## Arena 3: the b-tree balance scratch — spatial only, and the CHERI port already tripped on it

`balance_nonroot()` (`src/btree.c:8390-8407`) sizes ONE block and cuts three
differently-typed regions out of it:

    szScratch = nMaxCells*sizeof(u8*)      /* b.apCell  -- POINTERS */
              + nMaxCells*sizeof(u16)      /* b.szCell */
              + pBt->pageSize;             /* aSpace1   */
    b.apCell  = sqlite3StackAllocRaw(0, szScratch);
    b.szCell  = (u16*)&b.apCell[nMaxCells];
    aSpace1   = (u8*)&b.szCell[nMaxCells];

`sqlite3StackAllocRaw` is `alloca(N)` under `SQLITE_USE_ALLOCA` and
`sqlite3DbMallocRaw(D,N)` otherwise (`src/sqliteInt.h:4750-4757`). Either way the
block is a single allocation, so an overrun from `apCell` into `szCell`, or from
`szCell` into `aSpace1`, stays inside one capability and is not a spatial
violation. Unlike arenas 1 and 2 the block itself is properly released, so this
one is a spatial blind spot only.

**Independent evidence that capabilities really live in this block:** the
CHERI-adapted amalgamations on this machine (`~/cheri-work` and
`~/cheribsd-26.07`, both SQLite 3.53.4) carry exactly one functional deviation
from upstream here:

    upstream  src/btree.c:8406   aSpace1 = (u8*)&b.szCell[nMaxCells];
                                 assert( EIGHT_BYTE_ALIGNMENT(aSpace1) );
    patched   sqlite3.c:81634    aSpace1 = (u8*)__builtin_align_up(&b.szCell[nMaxCells], 16);

Someone had to raise the alignment from 8 to 16 to make this work on CHERI,
which is what a 128-bit capability needs. So the "no CHERI awareness" statement
above is a claim about UPSTREAM; the tree actually measured is patched, and this
is the patch.

## A second blind-spot axis: in-bounds uninitialised reads

The arena argument is about *where* an object lives. This one is orthogonal and
covers 21 of the table's rows: CHERI checks that an access is **in bounds** and
that the capability is **valid**. It does not check that the bytes were ever
**written**. An uninitialised read inside your own allocation is, to the
hardware, a perfectly legal load.

Two source facts make this worse in SQLite specifically, and both are verified:

1. **A freed lookaside slot is not scrubbed in a release build.** The trashing
   memset in `sqlite3DbNNFreeNN` is `#ifdef SQLITE_DEBUG`
   (`src/malloc.c:429` and `:468`). So a recycled slot still holds the previous
   occupant's bytes — and on purecap those bytes can include a **still-tagged,
   still-valid capability**. The comfortable assumption that "uninitialised
   memory holds junk, junk has no tag, the load traps" does not hold for an
   arena slot: the load may quietly succeed and hand out a live pointer to
   somebody else's object.
2. The same shape shows up outside lookaside. CVE-2019-13751 (Magellan 2.0)
   reads allocated-but-unwritten bytes in FTS3; the region is a real malloc, so
   bounds are correct and the read is in-bounds. Nothing traps.

This is why the five "uninitialised pointer read" CVEs from 2019-2020 —
`sqlite3WindowRewrite`, `multiSelect`, `selectExpander`,
`isAuxiliaryVtabOperator`, `AggInfo` init — were **downgraded** in the table from
`CAUGHT-tag-candidate` to `ARENA-DECIDES`. The first reading assumed the tag
check saves us; it only does when the stale bytes are not themselves a valid
capability, and in a lookaside slot they very often are.

## Turning a table row into a measurement

The pattern is already in the tree and should be copied rather than reinvented:
`tests/cheri-baseline/blindspot-mruby-ary-delete/`. What makes it a measurement:

1. **Three revocation configurations** (spatial / temporal / eager), because a
   blind spot that survives eager revocation is a much stronger result.
2. **A vehicle control in the SAME boot** — `catch_control.c` mallocs 64 bytes
   and walks 4096, and must die on SIGPROT (rc=162). Without it, "CHERI missed
   it" cannot be told apart from "this harness never reports anything". Every
   verdict before that control was added had been a miss.
3. **The oracle is the ANSWER, not a crash.** In an arena the program does not
   fault; it returns the wrong value. The mruby case ends with a `String` where
   a `C` instance belongs. A SQLite equivalent is a query returning wrong rows,
   a wrong `sqlite3_errmsg`, or a value read back from a recycled slot.
4. **The trigger has to be shown to fire on the host first**, on the affected
   version, against a known-good version as the differential.

Concretely for the top rows: a lookaside UAF needs a statement whose parse tree
is freed and whose slot is then handed to the next allocation within the same
connection, with the observable being the value that comes back. That is why
`alloc_arena` is the column to fill first: it says whether the observable is a
crash (heap rows: CHERI should trap) or a wrong answer (arena rows).

## What is still open

1. The bounds half of the claim (under audit, see above).
2. `sizeof(Expr)` and `sizeof(Window)` on a **purecap** target: pointers are 16
   bytes there, so both structs are larger than on x86-64. They must still fall
   under the 1200-byte slot size for the lookaside path to be taken at all.
3. Per-CVE tracing: this file establishes the arenas; each row of the CSV still
   needs its own object traced to one of them before its verdict is more than a
   component-level inference.
