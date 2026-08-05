# Real-Lua cross-domain reproduction on Capstone (luaossl #124)

First corpus case reproduced through the **real Lua interpreter** on Capstone,
instead of a pure-C shim. The cross-domain double-free is now driven by Lua's own
`userdata` + `__gc` + garbage collector; revoke-on-free must catch the second
owner's stale access. This is the fidelity upgrade the interpreter bring-up unlocks
(see 05-08-2026_06-00-00_gp-captable-lua-bringup.md for how real Lua runs at all).

## Why this is newly possible

The corpus shims are pure C (`shims/luac_*.c`) because, until now, Capstone could
not run real Lua -- so the common denominator both Capstone and CHERI could execute
had to drop below the language runtime (see `xlang/lua-cdp/WHY-SHIM.md`). CHERI could
always run real Lua (CheriBSD is a full OS); Capstone was the blocker. With real Lua
now running on Capstone, the reproduction can move back up to the actual runtime on
both platforms -- same fairness, higher fidelity: the free fires through the GC.

## The reproduction (LUA_CDP_X509 in lua_domain.c)

luaossl #124: one `X509_STORE` co-owned by a Lua-GC store `userdata` (whose `__gc`
= `X509_STORE_free`) and an `SSL_CTX` (whose free also frees `cert_store`), because
`setStore` is set0 -- takes ownership, no refcount up-ref. Modelled as:

- `store_new()` -> `malloc(152)` X509_STORE (refcount@136 = 1), wrapped in a real
  `userdata` with metatable `__gc = xs__gc`.
- `ctx_new()` -> a MockSSL_CTX `userdata`, metatable `__gc = sx__gc`.
- `ctx:setStore(store)` -> `ctx->cert_store = store` (set0 alias; NO up-ref; NO Lua
  reference kept, so the GC can free the store userdata independently).
- Two `lua_gc(LUA_GCCOLLECT)` passes (from C, so no base library needed):
  1. store userdata collected -> `xs__gc` -> `X509_STORE_free`: rc 1->0 -> `free`
     -> **REVOKE**.
  2. ctx userdata collected -> `sx__gc` -> `X509_STORE_free(cert_store)`: the
     `cert_store + 136` refcount access through the **revoked** alias.

The C `X509_STORE`/`SSL_CTX` are minimal stubs (only the memory lifecycle the bug
depends on -- what the shim already distilled); everything cross-domain (the
`userdata` handles, the `__gc` metamethods, the GC-driven free) is the genuine
interpreter.

## Result (QEMU) -- CAUGHT vs MISSED on the same code

Marker trace (csdebugprint; 500/501 = xs__gc, 510/511 = sx__gc):

- **REVOKE** (`LUA_CDP_X509`): `480 481 482 483  500 501  510` -> then
  `helper_cscincoffsetimm: rs1->tag` FAULT. The first free (xs__gc, 500/501)
  completes; the second owner's refcount access (sx__gc, after 510) faults on the
  revoked/untagged `cert_store` -- the CDP contract point. No `511`, no `CDP-MISS`.
  This is the corpus's "assert-on-untagged" catch route, now reached through Lua GC.
- **CONTROL** (`+LUA_CDP_NO_REVOKE`, `xlang_set_no_revoke()`): confirmed
  `480 481 482 483  500 501  510 511 484` -> `CDP-MISS: double-free survived`. The
  stale access at 510 now COMPLETES (511, 484 reached). Same code; only the revoke
  differs -> the reproduction is real, not an artifact.

## What this proves / does not

- PROVES: real Lua `userdata` + `__gc` + GC-driven collection run on Capstone, and
  revoke-on-free catches a genuine GC-driven cross-domain double-free at the stale
  access. The whole corpus can move to real-Lua-on-both (Capstone + CHERI).
- DOES NOT: run the real OpenSSL (the C object stays a stub -- freestanding domain,
  no OS); still QEMU, not silicon; one case, not all 13.

## Reproduce

```
LUA_CDP_X509=1 bash xlang/lua-cdp/capstone-lua/build-lua-gp-captable.sh          # revoke -> CAUGHT
LUA_CDP_X509=1 LUA_CDP_NO_REVOKE=1 OUT_DIR=... build-lua-gp-captable.sh          # control -> MISS
# run each via run-domain-smoke.py with the capstone_new.ko swap (see bringup note).
```
