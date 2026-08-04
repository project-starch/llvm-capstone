# LuaBridge #319 — Lua wrapper ⟷ destroyed C++ temporary use-after-free

**One line.** `getA():fn()` chains: `getA()` makes a Lua userdata wrapping an `A`,
`:fn()` returns `A&` which LuaBridge wraps as a **non-owning reference**; the
intermediate userdata is collected, so `a1:getI()` reads the destroyed `A`.

## Identity

| | |
|---|---|
| Library | [`LuaBridge`](https://github.com/vinniefalco/LuaBridge) (vinniefalco), header-only |
| Language pair | **C++ ⟷ Lua** |
| Upstream | https://github.com/vinniefalco/LuaBridge/issues/319 |
| Version | current `Source/` header; reference Lua 5.4 |
| Native dep | none beyond a C++17 compiler |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the userdata from `getA()` (owns an `A` by value).
2. **Separate reference handle:** `a1`, a non-owning reference LuaBridge pushes
   for `fn()`'s `A&` return, pointing into that userdata's `A`.

Two distinct allocations. The reference outlives the owning userdata.

**Direction:** GC-frees. The owning userdata is collected; `a1` derefs the freed
`A` (whose `~A` set the sentinel `i = -1`).

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: PUC Lua 5.4.7 (shared toolchain), LuaBridge current, gcc 15 ASan.
- Vuln (`getA():fn()`): `a1:getI()` returns **-1** — the destructor sentinel,
  proving the wrapped `A` was destroyed before the read.
- Control (non-chained `local a=getA(); a:fn()`): returns **5** (owned copy).
- Note: ASan does not reliably fire — Lua reuses the freed userdata block — so
  the **-1 sentinel** is the deterministic proof, not ASan.

## PASS signature

Vuln prints `got -1`; control prints `got 5`. Both required.
