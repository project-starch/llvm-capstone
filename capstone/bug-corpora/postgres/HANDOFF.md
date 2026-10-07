# Where the PostgreSQL work stands

Written 2026-10-06, when this branch was parked to go back to SQLite.

## The four buckets, as built

|            | nested | non-nested |
|------------|-------:|-----------:|
| temporal   | 5      | **0**      |
| spatial    | 9      | 5          |

18 cases of which 15 have been run. `temporal, non-nested` is the one empty
cell and the candidate that fills it is identified below.

- `mmgr-repros` 5 — temporal, nested. All five run on five arms.
- `sql-repros` 9 — spatial, nested. 01-05 run on three arms; 06, 07, 08, 09 not run.
- `c-repros` 5 — spatial, non-nested. All five reproduced under host ASan and
  run on the purecap arm with the fault attributed to the defect; the two
  Capstone arms cannot host them (see that corpus's note).

## Candidates verified and classified, not yet built

| commit | bucket | cost |
|---|---|---|
| `2aa6be6e64` + `011eedcdc3` | **temporal, non-nested** | highest value, highest cost — see below |
| `ac3afd1d00` | temporal, nested | induce OOM inside the aligned-allocation path |
| `c05c3baf16` | spatial, **both** | `src/common/pg_lzcompress.c`, so one defect fills two cells |
| `319e8a6441` | spatial, nested | upstream hardened seven contrib extensions at once; `10ebc4bd67` adds coverage tests whose inputs can serve as triggers |
| `d1bd9a7dc3` | spatial, nested | ltree casefolding changes byte length |
| `2543b9ea92` | spatial, nested | the DSA allocator corrupting its own pagemap; needs test_dsa, and whether a single-user backend can run DSA is unverified |

### The one that fills the empty cell

`2aa6be6e64` with `011eedcdc3`. `contrib/pgcrypto/openssl.c:765` allocates the
OSSLCipher with `MemoryContextAllocZero` (nested), but the object that gets
freed twice is `od->evp_ctx`, created at `:769` by `EVP_CIPHER_CTX_new()` —
OpenSSL's allocator, which is libc. So it is a **temporal, non-nested defect in
the backend**, which is also the counterexample to the idea that the backend is
all nested.

Blocked on the trigger. Upstream's reproducer is bug #19527's
`encrypt_iv(repeat('A',1073741308)::bytea, ...)`, about 1 GB of bytea, which
the domain cannot run. The defect is not about size: it is "raise an error
while an OSSLCipher is live". **Before writing any reduction, look for a cheap
error path** — a bad key length or a bad padding may reach the same place for
nothing. Only if none exists does this need `resowner.c` added to the port.

## Open items from Diego's review of PR #178

1. **Scope narrowing — done.** The 14 out-of-scope cases are removed.
2. **Old case 16 (now `sql-repros/05`) is not reached on any arm.** Confirmed:
   all three arms report `errors 4/4`, the server rejects every statement like
   a fixed build. Its trigger uses `convert_from(..., 'SQL_ASCII')`, which
   validates against the database encoding and fails before the defect.
   **`sql-repros/06` inherits the same broken route** and carries a
   `harness_limit` saying so; it must not take a VM slot until the route works.
   A candidate not yet tested: `SET client_encoding = 'SQL_ASCII'` on a UTF8
   database.
3. **`sql-repros/03` is silent on both Capstone arms and unexplained.** Still
   open. The lead: `CAPSTONE_LEVEL0_SHRINK` may be off, so a pointer carries
   the bounds of the whole arena rather than of the object. This is measurable
   the same way the purecap size-class bounds were measured for `c-repros` —
   read the bounds the pointer actually carries, rather than arguing about it.
4. **`differential` is not a detection, and the verdicts need defining in the
   README.** Not done.
5. **`sql-repros/README.md` is stale** and not merely out of date: it describes
   cases that no longer exist (12, 14, 18, 13), says `spatial` and `sublet`
   were never run when both ran on 2026-10-05, and says "measured for all 19".
   It needs rewriting, not patching, and that is where item 4 belongs.
6. **Chinese titles — done.** No CJK anywhere in the corpora.

## Two things that are inferred and should be measured

- **`sql-repros` cases are all marked `allocator_layer: aset` by inference**,
  from "the default query context is an AllocSet". Only case 01 has indirect
  evidence (PostgreSQL's own chunk-header check fired). This is measurable:
  the chunk header encodes `MemoryContextMethodID`, which is what
  `GetMemoryChunkContext` dispatches on.
- **Generation and Bump have no cases at all.** Sixteen of the nested cases
  are aset and one is slab. Bump is the interesting gap: its chunks have no
  header and it has no pfree, so the aset effect where a stale read lands on
  the allocator's own free-list links cannot happen there, and a temporal
  defect would look different.

## Repository state

Seven commits sit on top of `6d707b7bf34a`, which is the head Diego
force-pushed after rebasing onto dev and adding his two commits. Nothing has
been pushed. A pre-commit and a commit-msg hook are installed in
`llvm-capstone/.git/hooks`, shared by every worktree, refusing any commit whose
message or content is not English-only. They are self-testing: writing this
paragraph tripped them once.
