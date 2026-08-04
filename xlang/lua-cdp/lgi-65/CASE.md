# lgi #65 — GC guard userdata ⟷ `GArray` in a C struct field use-after-free

**One line.** `iface.methods = { method }` marshals a Lua table into the `GArray`
backing `DBusInterfaceInfo.methods` and wraps it in a Lua-GC guard; the guard's
`__gc` (`guard_gc` → `g_array_unref`) frees the `GArray` while the C struct still
points at it, so reading `iface.methods` re-derefs the freed `GArray`.

## Identity

| | |
|---|---|
| Library | [`lgi`](https://github.com/lgi-devs/lgi) — reference-Lua GObject-introspection binding |
| Language pair | **C ⟷ Lua** (reference Lua 5.4; guard userdata) |
| Upstream | https://github.com/lgi-devs/lgi/issues/65 |
| Fix commit | **`358371fd`** ("Fix marshalling array 2c with transfer != none") — adds `array_detach` so an owned field-set `GArray` keeps its data segment |
| Vulnerable tree | pinned HEAD **`7a2276f`** with the C-array field-set guard forced back to `g_array_unref` (see build.sh) |
| Native dep | gobject-introspection 1.86, glib 2.88 (GArray), Gio typelib |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the guard userdata created by `marshal_2c_array`
   (`lgi_guard_create`, `__gc` = `guard_gc`, `core.c:256`) that owns the marshalled
   `GArray`.
2. **Separate native resource:** the `GArray` (from `g_array_sized_new`) whose
   data segment is installed into the C struct field `DBusInterfaceInfo.methods`.

**Direction:** GC-frees. The field-set transfers array ownership to the C struct,
but on the vulnerable tree the guard still `g_array_unref`s the whole `GArray`;
the struct field is left dangling and read afterwards.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- The bug is **fixed on current HEAD** (`358371fd`, 2013): for an owned
  (`GI_TRANSFER_EVERYTHING`) field set, `marshal_2c_array` detaches only the
  `GArray` container (`array_detach`) and leaves the data for C, so nothing
  dangles. We reconstruct the pre-fix behaviour by forcing that guard back to
  `g_array_unref` on a buildable HEAD (a plain `git revert 358371fd` conflicts,
  and the historical tree predates lgi's Lua 5.4 support).
- Vuln (guard reverted): valgrind **Invalid read of size 8** in
  `marshal_2lua_array` (`marshal.c:562`) ← `lgi_marshal_field` ←
  `lgi_marshal_access`, freed by `guard_gc` (`core.c:256`); block alloc'd by
  `g_array_sized_new` ← `marshal_2c_array`. exit 99.
- Control (read before GC, then detach the field): clean, `DONE`, exit 0.
- Fixed (pinned HEAD, `array_detach` present): clean, `DONE`, exit 0.
- `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln: valgrind exit 99 with an `Invalid read` in `marshal_2lua_array` whose freed
block is attributed to `guard_gc`. Control and fixed: exit 0, `DONE`, no invalid
read. All three required.
