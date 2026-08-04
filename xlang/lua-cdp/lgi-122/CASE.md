# lgi #122 — cairo.Region userdata ⟷ boxed `cairo_region_t` GC use-after-free

**One line.** A `cairo.Region` lgi record is finalised (its boxed
`cairo_region_t` freed by the record `__gc` → `g_boxed_free` →
`cairo_region_destroy`) while a second `__gc` finaliser still calls
`r:get_extents()` on it — a GC-order UAF across two allocations.

## Identity

| | |
|---|---|
| Library | [`lgi`](https://github.com/lgi-devs/lgi) — reference-Lua GObject-introspection binding |
| Language pair | **C ⟷ Lua** (reference Lua 5.4; userdata records) |
| Upstream | https://github.com/lgi-devs/lgi/issues/122 |
| Fix commit | **`94f970d8`** ("Make objects unusable in the __gc metamethod") — nils the finalised record's metatable in `record_gc` |
| Vulnerable tree | pinned HEAD **`7a2276f`** with `94f970d8` **reverted** (see build.sh for why not the historical tree) |
| Native dep | cairo 1.18.4 (+ cairo-gobject), gobject-introspection 1.86, glib 2.88 |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the `cairo.Region` lgi record userdata (its `__gc` =
   `record_gc` frees the owned boxed value).
2. **Separate native resource:** the boxed `cairo_region_t`, allocated by
   `cairo_region_create()` and freed by `g_boxed_free` → `cairo_region_destroy`.

**Direction:** GC-frees / GC-order. Lua 5.4 finalises the region record *before*
the `{}`-proxy (it was marked for finalisation later), so the boxed region is
freed first; the proxy's `__gc` then reads it.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- The verbatim issue reproducer is **guarded on current HEAD**: `record_gc` nils
  the finalised record's metatable (commit `94f970d8`, "Fixes: issue #122"), so
  the second finaliser's `r:get_extents()` fails at the Lua level
  ("attempt to index a userdata value") before reaching cairo. Same situation as
  xmlua #35's high-level guard — so we drive the actual buggy state by reverting
  that guard onto a buildable HEAD (the pre-fix 2017 tree predates lgi's Lua 5.4
  support and will not compile).
- Vuln (guard reverted): valgrind **Invalid read of size 4** in
  `cairo_region_get_extents`, freed by `record_gc` (`record.c:438`) →
  `record_free` → `g_boxed_free`; block alloc'd by `cairo_region_create`. exit 99.
- Control (safe access, `r` kept alive across GC): clean, `DONE`, exit 0.
- Fixed (pinned HEAD, guard present): clean, `DONE`, exit 0.
- `./build.sh && ./run.sh` → PASS.

## PASS signature

Vuln: valgrind exit 99 with an `Invalid read` in `cairo_region_get_extents` whose
freed block is attributed to `record_gc`. Control and fixed: exit 0, `DONE`, no
invalid read. All three required.
