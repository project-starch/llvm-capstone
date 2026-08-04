# LMDB — value handle ⟷ freed transaction page use-after-free

**One line.** A Lua value handle borrows the zero-copy pointer that `mdb_get`
returns into a transaction's page; reading it after `mdb_txn_commit`/`_abort`
dereferences a page the transaction already freed.

## Classification

**DOCUMENTED-CONTRACT reproduction — NOT a filed bug.** lmdb.h states the value
lifetime verbatim, and this case reproduces a binding that violates it. It is
**not** pinned to an upstream issue/fix commit (see "Filed-bug search" below).

## Identity

| | |
|---|---|
| Native library | LMDB (liblmdb) **0.9.31-1build2** (system `liblmdb.so.0`) |
| Language pair | **C ⟷ Lua** (reference PUC Lua 5.4.7, shared toolchain) |
| Binding | `minilmdb.c` — a minimal, deliberately-unsafe binding **written here** |
| Contract violated | `lmdb.h:249-251` and `lmdb.h:1275-1276` (quoted below) |
| Detection | ASan **heap-use-after-free** (clean trap, with control) |

### The contract (verbatim, `/usr/include/lmdb.h`)

> Values returned from the database are valid only until a subsequent update
> operation, or the end of the transaction. Do not modify or free them, they
> commonly point into the database itself.  — lmdb.h:249-251 (also :1275-1276 on `mdb_get`)

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the `minilmdb.val` userdata from `txn:get(k)`. It stores
   only `{const char* data; size_t size}` — a **borrowed** pointer into the
   transaction's page, deliberately *not* copied out.
2. **Separate native resource:** the LMDB **overflow page buffer** that holds
   the value bytes, malloc'd by `mdb_put`→`mdb_cursor_put` inside the write txn
   and owned by the LMDB txn/page domain.

Two distinct allocations, two owners (LMDB txn vs Lua GC), one crossing apart.

**Direction:** native-frees. Ending the transaction frees the page while the Lua
handle still references it; the next `val:read()` derefs freed memory.

## Why a safe binding is not vulnerable (and this one is)

The real bindings copy the value out the instant `mdb_get` returns, while the
pointer is valid: shmul/lightningmdb does `lua_pushlstring(L, v.mv_data, v.mv_size)`
directly (`lightningmdb.c`, `txn_get`/cursor `get`). That copy is a Lua string in
the GC domain — no borrow survives the txn. `minilmdb.c` instead returns a
**deferred** handle and reads it lazily, which is exactly the shape the contract
forbids.

## Detection note (why a multi-page value)

For an ordinary small value LMDB **pools** the freed dirty page onto
`env->me_dpages` (`mdb_page_free`) rather than calling `free()` — the same
allocator-pooling that masks the free in the libdbus case, so ASan would see
nothing. LMDB instead **`free()`s a multi-page overflow buffer outright** at txn
end (`mdb_dpage_free`: `!IS_OVERFLOW || mp_pages==1` → pool, **else free**). The
trigger therefore stores a ~280 KB value (multi-page overflow), so the borrowed
pointer lands in a buffer LMDB really frees, and the stale read is a clean
heap-use-after-free rather than a silent stale read.

## Filed-bug search (due diligence)

- **shmul/lightningmdb** (the canonical C binding): copies values out immediately
  with `lua_pushlstring` — not vulnerable. Its 3 open issues (#14 comparator,
  #16 lpack→compat-5.3, #17 `get_path`/mapsize-grow crash) are unrelated to value
  lifetime. No "value used after txn end" issue/PR exists.
- No filed UAF of this shape was found in the Lua-LMDB bindings. This case is
  therefore the **documented-contract** reproduction, faithful to lmdb.h.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), system liblmdb 0.9.31, gcc/clang ASan,
  4 KB pages. `MDB_NOSUBDIR` single-file DB on an `os.tmpname()` path, no
  `MDB_WRITEMAP` (dirty pages malloc'd, ASan-visible).
- **Vuln** (`txn:commit()` then `h:read()`): ASan **heap-use-after-free** at
  `minilmdb.c:117` (`l_val_read`); freed by `mdb_txn_commit`→`free`
  (`l_commit`, minilmdb.c:101); allocated by `mdb_put`→`mdb_cursor_put`→`malloc`
  (`l_put`, minilmdb.c:79). `txn:abort()` frees identically (`mdb_txn_abort`).
- **Control** (`h:read()` *before* the txn ends): `read ok=true len=280000` —
  correct bytes, no ASan error.
- `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln output contains `heap-use-after-free`; control output contains
`read ok=true` and no `AddressSanitizer` line. All three required.
