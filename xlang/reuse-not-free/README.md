# Reuse-not-free: the class no allocator-based defense can see

*Class 3 of `agent-handoff/design/sharing-bug-taxonomy-and-novelty.md`. Started 2026-08-14.*

**The owner recycles a buffer IN PLACE. No `free()` ever happens.** The borrower's pointer
stays valid, tagged and in bounds; only the *data's identity* changes. Every defense that
keys on an allocator event — ASan, MTE, quarantine-and-sweep, CHERI's revocation in any
configuration — is blind by construction, because there is no event to key on.

## Result 1 — the paper's motivating example IS this class

`sqlite3_column_text()` borrowed across `sqlite3_step()` is **reuse-not-free, not
use-after-free.** Verified three independent ways.

**Documentation** ([sqlite.org/c3ref/column_blob.html](https://www.sqlite.org/c3ref/column_blob.html)):
*"The pointers returned are valid until a type conversion occurs …, or until `sqlite3_step()`
or `sqlite3_reset()` or `sqlite3_finalize()` is called."* The following sentence — *"The memory
space used to hold strings and BLOBs is freed automatically"* — is about **ownership** (do not
call `sqlite3_free()` on it), not about when the loan ends. It reads like free-then-use and is
not; treat it as a trap.

**Source.** `OP_Column` → `sqlite3VdbeMemFromBtree()` → `sqlite3VdbeMemClearAndResize()`
(`src/vdbemem.c:301-324`), documented as *"If `pMem->zMalloc` already meets or exceeds the
requested size, this routine is a no-op."* When the next row fits the existing allocation — the
common case — the next `step()` overwrites the previous row's bytes at the identical address.
No `free()`, no `realloc()`. A second, purer path exists: `sqlite3VdbeSerialGet()`
(`src/vdbeaux.c:4174-4177`) points `pMem->z` straight into the pager's page-cache buffer, which
the pager recycles page-for-page with no allocator event either.

**Experiment, with a positive control** (`sqlite-column-text/`, system libsqlite3 3.45.1):

| Arm | Differs by | Result |
|---|---|---|
| **A — control** | `strdup` → `free` → read | **ASan fires**: `heap-use-after-free`, aborts |
| **B — subject** | borrow across `sqlite3_step()` | **ASan silent.** Same pointer, `AAAA` → `BBBB` |

Four consecutive rows return the **identical address** `0x…d988`. Repeated with
`SQLITE_CONFIG_LOOKASIDE` fully disabled — still silent, so this is not lookaside masking a
free. Full transcript in `sqlite-column-text/EVIDENCE.txt`.

**The control is what makes this evidence.** ASan is proven to fire on a real UAF in the same
binary; its silence on the subject is therefore a fact about the subject, not about the
instrument.

## Result 2 — a SECOND, distinct blindness mechanism

`sqlite3_column_name()` (the shape behind RUSTSEC-2021-0037, diesel) is **not** the same bug.
Its loan ends at an automatic *statement reprepare*, and at the SQLite level the string **is**
freed — but it lives in a **lookaside slot**, which the connection recycles without returning
memory to the process allocator ([sqlite.org/malloc.html](https://www.sqlite.org/malloc.html)).

Matched pair (`sqlite-column-name/`), differing only in whether lookaside is enabled:

| Arm | Result |
|---|---|
| lookaside **ON** (default) | **ASan silent**; stale pointer reads SQLite's `0xaa` trash-fill |
| lookaside **OFF** | **ASan fires**: `heap-use-after-free … freed by … sqlite3_free` |

**Do not conflate the two.** `column_text` = no free at all. `column_name` = freed, but
recycled inside an internal pool. Both defeat ASan; they are different stories, and having
both is stronger than having either.

## Candidate corpus — triaged, not yet built

| ID | Specimen | Why it qualifies | Repro cost |
|---|---|---|---|
| **A1** | cassandra-rs — RUSTSEC-2024-0017 / CVE-2024-27284 | `ResultIterator` holds a **single** `Row row_;` re-pointed by `decode_next_row()` at a new offset of the same live buffer. Source-verified. **The advisory misfiles it as CWE-416 "freed memory" when nothing is freed** — a citable instance of the class being systematically mislabelled | hard (needs live Cassandra) |
| **A2** | Go `database/sql` `RawBytes` — [golang/go#65201](https://github.com/golang/go/issues/65201) | *"only valid until the next call to Rows.Next, Rows.Scan, or Rows.Close"*; the bug made `Scan` **write into** the driver's buffer | medium |
| **A3** | git `git_path()` / `mkpath()` | 4-slot **static ring buffer**; `strbuf_reset` keeps the allocation, so **zero allocator events ever**. Purest form | **easy, ~20 lines, no deps** |
| **A4** | LMDB `MDB_val` | Points into the mmap'd B-tree page — *"Do not modify or free them, they commonly point into the database itself"* (`lmdb.h:278`). Contract verified; **no filed defect found** | easy |
| **A5** | curl connection-pool reuse — CVE-2021-22924, CVE-2022-27782 | Object recycling with an authority change, no allocator event. The *harm* model at scale; the borrower does not hold a stale pointer, so use as analogue only | hard |

Hierarchical (class 4) candidates found in the same sweep, recorded for later: Nokogiri
`XPathContext` outliving its `Document` (GHSA-p67v-3w7g-wjg7, easy); libexpat CVE-2022-43680
(**inverted** hierarchy — a *child*'s free destroys the root's shared DTD); curl
CVE-2026-10536 (a literal parent/child handle graph via `CURLOPT_STREAM_DEPENDS`); GLib
`GMainContext` freed under live `GSource` children. And [curl#17578](https://github.com/curl/curl/issues/17578)
is the bridge — a class-4 trigger (teardown order) producing class-3 damage (a pooled object
carrying the previous transfer's data).

**Ruled out, so nobody re-hunts them:** CVE-2018-16840 (plain intra-function UAF);
CVE-2025-12863 (**REJECTED** by the CNA — do not cite); RUSTSEC-2021-0022 and libexpat#85
(realloc moved the block → free-then-use); RUSTSEC-2021-0128 (Rust lifetime unsoundness).

`sqlite3_close()` with unfinalized statements is worth keeping as a **negative control**: it
returns `SQLITE_BUSY` and leaves the connection a zombie until all statements are finalized —
SQLite defends the invariant everyone else violates.

## Next

1. Build A3 (git ring buffer) as a ~50-line standalone — no dependencies, purest specimen.
2. Port `sqlite-column-text` into a Capstone domain: does revoke-on-free catch it? **It should
   not** — nothing is freed, so revocation is never triggered. The fix has to be the *lender*
   revoking at `step()`, i.e. borrowing, which is exactly the paper's argument.
3. CHERI column, three configs. Expect 0/N in all three, including eager.

Item 2 is the interesting experiment: this class is the one where **our own revoke-on-free is
also blind**, and only explicit lender-driven revocation at the contract point works. That
distinction should be stated plainly rather than glossed.
