# tarantool #7657 — LuaJIT cdata ⟷ C `struct merge_source` use-after-free

**One line.** A LuaJIT cdata wraps a refcounted C `struct merge_source`;
`source:pairs()` iteration mints a fresh un-ref'd cdata each step whose GC
finalizer (`lbox_merge_source_gc` → `merge_source_unref`) can drop the refcount
to 0 mid-iteration, freeing the struct while the next `lbox_merge_source_gen`
still derefs it → SIGSEGV.

## Identity

| | |
|---|---|
| Library | [Tarantool](https://github.com/tarantool/tarantool), builtin `merger` module (LuaJIT) |
| Language pair | **C ⟷ LuaJIT**. The coupled resource is a first-class C DBMS object (`struct merge_source`), not FFI plumbing — a strong CDP case. |
| Upstream | https://github.com/tarantool/tarantool/issues/7657 (+ tuple-merger #29) |
| Vulnerable build | `tarantool/tarantool:2.8.3` (`2.8.3-0-g01023dbc2`), pre-fix. Also reported on 2.7.3 / 2.8.4 / 2.10.0 / 2.10.2 / master-446. |
| Fix | PR **#7664**, commit **`e52fabf9058453efc0661092822feff609615ef1`** — "lua/merger: fix use-after-free during iteration"; backported to 2.10 (`f9aecfb`). |
| Fixed build (control) | `tarantool/tarantool:2.11` (`2.11.5-0-g12a9ceb870`) |
| Native dep | Tarantool's builtin `merger` C module + bundled LuaJIT (both in the stock image). |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** a LuaJIT cdata of ctype `CTID_STRUCT_MERGE_SOURCE_REF`,
   wrapping a `struct merge_source *`. Created by `merger.new_table_source(...)`,
   and re-minted by every `lbox_merge_source_gen` step as the iterator's next
   `state`.
2. **Separate native resource:** the C `struct merge_source` (refcounted, with a
   vtab), allocated by the merger source constructor and destroyed by
   `merge_source_unref` when its refcount hits 0.

Two distinct allocations. **Not** the string-borrow shape.

**Direction:** GC-frees. A GC pass mid-iteration collects an intermediate cdata
(crossing 1); its finalizer `merge_source_unref` destroys the `struct
merge_source` (crossing 2); the next `lbox_merge_source_gen` derefs the freed
struct's vtab.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: official prebuilt release images via docker (no from-source build needed —
  this is a plain SIGSEGV, not an ASan-only report).
- Vulnerable `2.8.3`: **SIGSEGV** (`SEGV_MAPERR addr 0`, `rip=0x0`, `cr2=0x0`) —
  a call through the freed/nulled `merge_source` vtab, via `lj_BC_FUNCC` into the
  merger gen path; tarantool's crash handler fires and docker exits 139.
  Deterministic (3/3) because `trigger.lua` forces a `collectgarbage('collect')`
  inside `fetch_chunk`, pinning the mid-iteration free the issue describes
  ("gc is called" at iter 5312).
- Control, fixed `2.11`: **no crash** — prints `7000`, exit 0.
- Full trace + control in `evidence.txt`.

## Vehicle note (LuaJIT-only)

This coupling is LuaJIT `cdata` and LuaJIT does not target `capstone64`. For a
reference-Lua Capstone vehicle the `struct merge_source` would be carried as a
**userdata** with a `__gc` metamethod — the same two-object coupling and the same
free-during-iteration bug, just PUC-Lua userdata in place of the cdata.

## PASS signature

`run.sh` passes iff the **vulnerable** image (2.8.3) SIGSEGVs in the merger gen
path — tarantool crash report (`Segmentation fault` / `crash_signal_cb`, or docker
exit 139) with `lj_BC_FUNCC` and a null-address fault (`SEGV_MAPERR` / `addr: 0`)
— AND the **fixed** image (2.11) completes cleanly printing `7000` with no crash
report. Either half missing = FAIL.
