# xmlua #35 — xpath object cdata ⟷ freed `xmlDoc` nodes use-after-free

**One line.** `libxml2.xmlXPathEvalExpression` returns `ffi.gc(object,
xmlXPathFreeObject)` with **no tie to the document**; if the `xmlDoc` is freed
first, the xpath object's finalizer (`xmlXPathFreeNodeSet`) reads freed node
`type` fields.

## Identity

| | |
|---|---|
| Library | [`xmlua`](https://github.com/clear-code/xmlua) (clear-code), LuaJIT-FFI libxml2 binding |
| Language pair | **C ⟷ LuaJIT** (FFI cdata) |
| Upstream | https://github.com/clear-code/xmlua/issues/35 (**OPEN** — no fix commit) |
| Buggy site | `xmlua/libxml2.lua:650-656` — `ffi.gc(object, xmlXPathFreeObject)` |
| Native dep | libxml2 (verified 2.15 / .so 16.1.2) + luacs (xmlua dep) |

## The two coupled objects (why unambiguous CDP)

1. **LuaJIT-GC handle:** the xpath object cdata (`ffi.gc` → `xmlXPathFreeObject`),
   whose nodeSet holds raw pointers to document nodes.
2. **Separate native resource:** the `xmlDoc` node tree (its own `ffi.gc` →
   `xmlFreeDoc`).

**Direction:** GC-order. The document is collected first (frees the nodes); the
xpath finalizer then reads freed node `type` fields.

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: LuaJIT 2.1, libxml2 .so 16.1.2, valgrind.
- **Faithful low-level path:** the high-level `doc:search` API *guards* this by
  wrapping nodes as `Element.new(document, node)` (`searchable.lua:106`), keeping
  the doc alive — so the UAF is only reachable via the internal wrapper. The
  repro drives xmlua's actual `libxml2.xmlXPathEvalExpression` (the buggy site).
- Vuln (doc freed first): valgrind **Invalid read of size 4** in
  `xmlXPathFreeNodeSet` ← `xmlXPathFreeObject`.
- Control (xpath object freed first, nodes still valid): clean, `DONE`.
- `./build.sh && ./run.sh` → PASS.

**Vehicle note:** LuaJIT `cdata` — reproduce in reference-Lua **userdata** form
for a Capstone build (a userdata wrapping the xpath object, coupled to the doc).

## PASS signature

Vuln: valgrind exit 99 with `xmlXPathFreeNodeSet` invalid read. Control: exit 0,
`DONE`, no invalid read. Both required.
