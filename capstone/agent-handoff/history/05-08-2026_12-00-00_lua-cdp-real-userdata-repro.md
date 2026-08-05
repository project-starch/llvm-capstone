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

## Scaled to the full 13-case corpus (real Lua)

All 13 corpus cases now have a real-Lua reproduction (lua_domain.c): 11 single-object
borrowed-view UAFs (`LUA_CDP_UAF`, one generic parameterized harness -- a C object
owned by a Lua-GC userdata whose `__gc` frees it, a borrowed view caching the raw
pointer, deref after GC), plus 2 double-frees (`LUA_CDP_X509` luaossl-124 two-userdata,
`LUA_CDP_OPENSSL` lua-openssl-141 close-then-`__gc` single-object). (size,off,write)
per case are the real bug's ASan values; the on-Capstone catch is offset-independent.

Results on QEMU (REVOKE = fault at the stale access = CAUGHT; CONTROL = completes =
MISS). Every case ran individually with revoke ON, and the 11 UAF controls ran in one
no-revoke boot:

| case | shape | control | revoke |
|------|-------|---------|--------|
| luaossl-124 (X509 dblfree)    | 2 userdata  | MISS | CAUGHT |
| lua-openssl-141 (ctx dblfree) | close+__gc  | MISS | CAUGHT |
| curl_multi_backptr            | UAF W@64    | MISS | CAUGHT |
| ffi_closure                   | UAF R@32    | MISS | CAUGHT |
| ldbus_message                 | UAF R@0     | MISS | CAUGHT |
| lgi_cairo_region              | UAF R@4     | MISS | CAUGHT |
| lgi_garray                    | UAF R@0     | MISS | CAUGHT |
| lmdb_value                    | UAF R@0     | MISS | CAUGHT |
| luv_costate                   | UAF R@24    | MISS | CAUGHT |
| pgconn                        | UAF R@376   | MISS | CAUGHT |
| sdl_window                    | UAF R@0     | MISS | CAUGHT |
| tvbuff                        | UAF R@16    | MISS | CAUGHT |
| uv_fs                         | UAF R@0     | MISS | CAUGHT |

**13/13 CAUGHT under revoke; 13/13 MISS under the no-revoke control** -- matching the
pure-C shim corpus's Capstone result (13/13), now driven through real Lua's GC. The
11 UAF controls pass in ONE boot (`LUA_CDP_NO_REVOKE` runs every case fresh-state ->
all "MISS survived" -> "UAF-LADDER done"); each revoke case is its own boot (the
untagged-access catch aborts QEMU -- the corpus's assert-on-untagged route). Both
write@offset and read@0/@offset shapes catch (cincoffset and direct-load).

Two dispatch/knob bugs were found and fixed during this run: LUA_CDP_OPENSSL was
initially missing from both the domain_main dispatch and the build-script knob, so
the openssl build silently ran the staged demo (result=400) -- caught because its
markers were absent (free=0 fault=0), re-run after the fix.

## The CHERI half: real Lua on both platforms

To make the fair comparison use real Lua on BOTH sides (the point of the whole
exercise), the same reproductions must run on CHERI too. On CheriBSD real Lua is an
ordinary purecap program (full OS, not freestanding), so the only gate was "does Lua
build purecap?" -- and it does.

Recipe (`xlang/lua-cdp/cheri/real/build-real-lua-cdp.sh`): CHERI-clang,
`--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d
--sysroot=$ROOTFS -mno-relax -ftls-model=initial-exec`, plus two fixes found here:
- `-nostdinc -isystem <clang-builtins> -isystem $ROOTFS/usr/include`: the default
  header search leaked to the HOST glibc (`/usr/include/bits/floatn.h` ->
  unsupported `__float128`); forcing sysroot-only headers fixes it.
- `-include cheri-lua-prelude.h`: no-ops the shared Lua source's leftover diagnostic
  probes (`DBGP`/`DBGC`, which Capstone defines via capstone_lua_libc.h), so the
  interpreter source stays byte-identical across platforms.

STATUS: real Lua 5.4.7 builds purecap (a CheriBSD RISC-V PIE), and the luaossl-124
reproduction (`cheri/real/cdp_x509.c` -- same userdata/__gc/setStore logic as the
Capstone LUA_CDP_X509 domain, as a normal `main()` with CheriBSD malloc/free) builds
and links against it purecap. NEXT: stage into the CheriBSD image and run under the
revocation configs via the existing `cheri-baseline/` drivers (eager -> CAUGHT at the
stale access; async/default -> MISS), and port the other 12 reproductions (same
template). This is the CHERI column at the real-Lua fidelity, to sit beside the
Capstone 13/13 above.

## Reproduce

```
LUA_CDP_X509=1 bash xlang/lua-cdp/capstone-lua/build-lua-gp-captable.sh          # revoke -> CAUGHT
LUA_CDP_X509=1 LUA_CDP_NO_REVOKE=1 OUT_DIR=... build-lua-gp-captable.sh          # control -> MISS
# run each via run-domain-smoke.py with the capstone_new.ko swap (see bringup note).
```
