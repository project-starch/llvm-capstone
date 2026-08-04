# Lua-CDP Capstone column — the 13 pure Lua↔C cases on Capstone

The Capstone half of the fair CHERI-vs-Capstone comparison for the
`xlang/lua-cdp/` corpus. Each of the 13 **pure Lua↔C** cross-domain-pointer
use-after-frees is distilled into a standalone C shim and run as a Capstone
domain under QEMU, with the revoke-on-free allocator. The security question:
**does the revoke prevent the stale cross-domain access?** — measured per case,
with a no-revoke control that must MISS for the catch to be attributable.

## Why only 13 (not the full 15)

The committed corpus has 15 rows; two shapes cannot run on Capstone and are
excluded here, not worked around:

- **C++ cases** (sol2 #1373, sol2 #1080, LuaBridge #319, wxLua #115) — Capstone
  has no libc++/STL for `capstone64`, and a C++ `new`-expression crashes the
  backend. Excluded.
- **LuaJIT cases** (xmlua #35, tarantool #7657) — LuaJIT is an asm JIT with no
  `capstone64` backend. Excluded.

The remaining 13 are all pure reference-Lua ↔ C, which compiles cleanly for
`capstone64`. See `rows.tsv` for the list.

## Methodology — reuse, don't reinvent

This column **reuses the mruby xlang column's harness verbatim** — the mruby
column already runs Lua-shaped standalone shims on Capstone (its rows 1–2 are
`rlua_userdata_uaf` / `rlua_escaped_handle_uar`). Reused from `../../capstone/`
without modification:

| Reused | Role |
|---|---|
| `mock_mruby_capstone.c` | the revoke-on-free allocator TU — provides `malloc`/`free` on top of `rof_malloc`/`rof_free`, so a shim's own `free()` **revokes** |
| `xlang_shim_domain.c` | the parameterised freestanding domain (`#include ROW_SRC`), `mock_report`, the 3 shared regions |
| `xlang_shim_host.c` | the Buildroot-Linux host controller that creates the domain and reads results |
| `revoke_on_free_alloc.h`, `start.S`, `link.ld` | allocator + freestanding runtime |

The **only new code** is one shim per case under `shims/`, plus thin
build/run drivers (`build-lua-capstone.sh`, `run-lua-capstone.sh`) that repoint
the shim directory at this corpus.

### Each shim is a faithful distillation

A shim reproduces exactly the memory-lifecycle events its case's CHERI/Capstone
verdict depends on — the two distinct allocations, the free that ends the native
resource's life, and the stale dereference one crossing later — at the **real
access offset** named by the case's ASan/valgrind trace. Every shim cites its
`../<slug>/boundary.md` free-site and stale-use-site in its header comment. No
library, VM, or ownership semantics are re-implemented; only the memory events.

The access offset matters for how the catch manifests under QEMU: an offset-0
load/store faults cleanly (delivered cause-24/25); an interior access forms its
address with `cincoffset` on the revoked capability and hits QEMU's
assert-on-untagged gap (an emulator limitation, still a prevented access). See
`rows.tsv` `fault_route`.

## Configs

Capstone has two configs that map onto CHERI's three (identical to the mruby
column):

- `bounds` = spatial only, no revocation ↔ CHERI **spatial**
- `revoke` = revoke-on-free ↔ CHERI **eager**

There is **no** Capstone analogue of CHERI's async quarantine default — a fact
about the mechanisms, reported rather than papered over. Only `revoke` (and its
no-revoke control) is run here.

## Build / run

```bash
source ../../../capstone/tests/capstone-test-env.sh
# one row, both variants, on QEMU:
./run-lua-capstone.sh luac_openssl_ctx_uaf
# the whole column (13 rows × 2 variants = 26 boots):
./run-lua-capstone.sh
```

`run-lua-capstone.sh` builds the host controller once, then for each row builds
the revoke + control domains and runs each in its own QEMU boot (a faulting
domain halts its boot, so rows are never batched). Outcomes are classified
MISS / FAULT / INVALID / NORESULT into `results.tsv`.

## Predictions and results

`rows.tsv` holds the predictions, committed **before** the runs — every row is a
temporal cross-domain UAF, so every prediction is `FAULT` under revoke and
`MISS` under the control, mechanism `revocation`. The measured outcomes go in
`expected-results.tsv` (the reproduce baseline), and the security comparison
table (per case × spatial/async/eager/Capstone) lives with the CHERI half.

**Status:** row 1 (`luac_openssl_ctx_uaf`) measured — revoke `FAULT` (cause 24) /
control `MISS`. Full 13-row pass: see `results.tsv`.
