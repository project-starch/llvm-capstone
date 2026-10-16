# 415540ddaa_0 -- 415540ddaa

Upstream fix `415540ddaa`. Collected in round R3 (backport filter).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
415540ddaa_0 -- upstream corrupt-database regression image, backported to 3.22.0.
 *
 * Collected by the backport filter: upstream fix 415540ddaa removes code that is still
 * present in 3.22.0 (or guards a function 3.22.0 has unguarded), and its own regression
 * test ships the triggering database as a hex dump. Host ASan on stock 3.22.0:
 *
 *     heap-buffer-overflow in sqlite3Fts5GetVarint
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

## Review 2026-10-16 -- proposed for withdrawal as a duplicate of case 00

### What looked like the answer, and why it was not enough

Both cases declare `sqlite3Fts5GetVarint` as their consumer, and the 2026-10-11 host-ASan
stacks are identical to six frames:

```
00: sqlite3Fts5GetVarint <- fts5SegIterReverse <- fts5SegIterSeekInit <- fts5MultiIterNew
                         <- sqlite3Fts5IndexQuery <- fts5ExprNearInitAll
25: sqlite3Fts5GetVarint <- fts5SegIterReverse <- fts5SegIterSeekInit <- fts5MultiIterNew
                         <- sqlite3Fts5IndexQuery <- fts5ExprNearInitAll
```

with identical allocation stacks ending in `fts5DataRead`, both on a 64-byte region, differing
only in how far past the end the read goes (4 bytes against 110).

That is **not** sufficient, because `fts5SegIterReverse` has two unchecked offsets feeding a
varint decode and one of them is a macro:

| line | code | buffer read |
|---|---|---|
| `sqlite3.c:196322` | `fts5IndexSkipVarint(pLeaf->p, iPoslist)` -- a **macro**, defined at `:196291` | `pLeaf->p`, the current leaf |
| `sqlite3.c:196367` | `iOff += fts5GetVarint(&pLast->p[iOff], (u64*)&pIter->iRowid)` | `pLast->p`, the page the forward search found |

A read through the macro produces no intermediate frame, so in an ASan stack it is
indistinguishable from the decode at `:196367`. The frames alone therefore cannot tell these
two cases apart, and they cannot tell either case's own line apart either.

### The measurement that settles it

Probe site 60 puts a hook at **each** candidate, printing the offset, the buffer length and
whether the offset is already past it. Both cases were built into one image and run:

```
case 00:  SKIPVAR pLeaf=... iPoslist=15 nn=22 szLeaf=20    oob=0
          LASTVAR pLast=... iOff=148    nn=24 szLeaf=20502 oob=1
case 25:  SKIPVAR pLeaf=... iPoslist=15 nn=22 szLeaf=20    oob=0
          LASTVAR pLast=... iOff=158    nn=24 szLeaf=22    oob=1
```

Both take the same line, `:196367`, reading the same field of the same structure in a 24-byte
buffer; the other candidate is in bounds in both. The only difference is the offset the corrupt
image supplies, 148 against 158.

### The limit, stated rather than glossed

The two cases cite **different upstream commits**, `adfb203a7d` and `415540ddaa`, and this
environment is offline, so what each changed cannot be read. Upstream may well have fixed two
distinct unchecked offsets, or hardened this function more than once. What is measured here is
that **in the 3.22.0 pin both regression images land on one unguarded read**. This corpus counts
observables in the pin, so in the pin it is one case. That is the same standard case 09 was
withdrawn under, on stronger evidence: there the argument was a rewrite of the trigger, here it
is a direct hook on each candidate line.

### Case 00 is the one kept

It names the buffer (`pLast`) and the caller, its `live_proof` quotes the upstream commit
message, it was already reviewed and reclassified from temporal to spatial on 2026-10-15
(`9061e8fb`), and it carries probe site 33 at this same line. Case 25's `object` field said only
"the fts5 page buffer the varint decoder reads from" -- which does not say *which* of the two
buffers that function hands to a decoder, and that is what hid the overlap. It is corrected
there now.

### No arm is re-run

For case 09's reason: there is nothing here that case 00 does not already measure, and case 00's
own readings were taken on this platform on 2026-10-15.

One earlier reading is corrected rather than altered. The old `status` line said "the probe
witnesses the defective access (kind=UAF, hits=5, witnesses=1)". `kind=UAF` is **not** evidence
of a use-after-free here. This is a spatial over-read, and the address judgement classifies by
where the pointer landed -- the injector's own documentation says an over-read that crosses into
a freed neighbour reads as UAF, one that lands in a live block reads as nothing. So `kind=UAF`
means the read left the buffer and landed in a free block, which is a second confirmation of the
over-read, not a statement about its class. Also recorded: the physical sublet row reports
`cause 5`, a plain load-out-of-bounds, where the virtual arm reports `cause 28` (CAP_OOB) -- two
different codes for the same over-read across builds.

### The siblings are not duplicates

Checked the same way, from the recorded stacks:

* **26 `8f5b14a5c2`** goes `sqlite3Fts5GetVarint <- sqlite3Fts5GetVarint32 <- fts5DecodePoslist
  <- fts5DecodeDoclist <- fts5DecodeFunction <- sqlite3VdbeExec` -- a different path, through the
  `fts5_decode` debugging function, with a different allocator (`sqlite3Fts5MallocZero`).
* **27 `a783931794`** is fts3 entirely: `sqlite3Fts3GetVarint <- fts3GetDeltaVarint <-
  fts3PoslistPhraseMerge <- fts3EvalIncrPhraseNext <- fts3EvalPhraseNext`.

### Withdrawals so far

With case 01 (not reached), 04 (not a memory-safety defect), 09 (duplicate of case 10) and 18
(the corpus's own positive control), this is the fifth, and the second of the duplicate kind.
33 becomes 28 at the single renumber that ends this review.
