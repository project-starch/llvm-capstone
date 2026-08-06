# Real-Lua reproductions (Capstone and CHERI)

The corpus's fair comparison (see `README.md`, `WHY-SHIM.md`) is measured through a
distilled **C shim** per case, because until recently Capstone could not run a real
language runtime freestanding. That constraint is gone: **reference Lua 5.4.7 now
runs on both platforms**, so each cross-domain-pointer (CDP) bug is reproduced through
the *actual* interpreter — real `userdata`, a real `__gc` metamethod, a GC-driven
free — with only the wrapped native C object left as a minimal stub. Same fairness
(byte-identical workload both sides), higher fidelity (the free fires through Lua's
own collector, which is what makes these bugs "cross-domain").

This is a **security-axis** measurement (is the stale cross-domain access caught?),
the same question the shim column answers, now at real-runtime fidelity. It is **not**
a claim that either platform runs the full upstream libraries (OpenSSL/SDL/libuv/… are
still stubbed) — and both columns are under QEMU, not silicon.

## The 13 cases

All are a Lua-GC handle co-owning a native C object with a native free:
- **11 single-object UAFs** — an owner `userdata` (`__gc` frees the object) plus a
  borrowed handle that caches the raw pointer; after the GC frees+invalidates the
  object, the borrowed deref is the stale access.
- **2 double-frees** — luaossl #124 (two userdata co-own one `X509_STORE`, set0) and
  lua-openssl #141 (a cipher userdata's `close()` frees the ctx, its `__gc` re-reads).

`(size, offset, read/write)` per case are each bug's real ASan values; the capability
catch is offset-independent.

## Capstone — freestanding gp-captable domain

Driver: `capstone-lua/lua_domain.c` (built by `capstone-lua/build-lua-gp-captable.sh`).
Real Lua runs as a Capstone domain; the revoking allocator (`revoke_on_free`) fires a
hardware REVOKE on every `free`, so the second owner's stale access through the revoked
capability faults.

```
# one UAF case (revoke) -> CAUGHT (fault at the stale access)
LUA_CDP_UAF=1 CDP_UAF_CASE=<0..10> bash capstone-lua/build-lua-gp-captable.sh
# all UAF cases, no-revoke control -> MISS (the UAF completes)
LUA_CDP_UAF=1 LUA_CDP_NO_REVOKE=1 bash capstone-lua/build-lua-gp-captable.sh
# the two double-frees (+ LUA_CDP_NO_REVOKE for the control):
LUA_CDP_X509=1  bash capstone-lua/build-lua-gp-captable.sh   # luaossl #124
LUA_CDP_OPENSSL=1 bash capstone-lua/build-lua-gp-captable.sh # lua-openssl #141
```
Run each `.dom` via `capstone/tests/runtime-qemu/run-domain-smoke.py` with the
descriptor-delivery module swap (`capstone_new.ko`) — see the bring-up note below.

**Result: 13/13 CAUGHT under revoke, 13/13 MISS under the no-revoke control.**

## CHERI — CheriBSD purecap process

On CheriBSD real Lua is an ordinary purecap program. One combined binary
(`cheri/real/cdp_real.c`, dispatched on `argv[0]`, copied to the 13 row names) is run
under the three revocation configs by the same `cheri-baseline/` drivers the shim
column uses:

```
bash cheri/run-real-lua-cheri.sh    # build 13 purecap binaries, boot CHERI-QEMU once,
                                    # run spatial/temporal/eager, classify
```

| config | knobs | result |
|--------|-------|--------|
| spatial  | revocation OFF                              | 0/13 (MISS) |
| temporal | revocation ON, ASYNC (the DEPLOYED default) | 0/13 (MISS) |
| eager    | revocation ON, every free                   | 13/13 (SIGPROT = CAUGHT) |

## The comparison (real Lua on both)

```
Capstone revoke-on-free : 13/13 CAUGHT
CHERI eager             : 13/13 CAUGHT   (= Capstone's synchronous revoke)
CHERI async (default)   :  0/13          (the deployed config does not catch the
                                          CDP at the contract point)
```

## Notes

- Getting real Lua onto Capstone needed four i128/capability codegen fixes in the
  LLVM backend — merged into `capstone-bootstrap` (this branch inherits them; the
  fixes are on `CapstoneISelLowering.cpp`, with lit tests `cap-i128-*.ll`).
- Full trail: `capstone/agent-handoff/history/05-08-2026_06-00-00_gp-captable-lua-bringup.md`
  (interpreter bring-up) and `…12-00-00_lua-cdp-real-userdata-repro.md` (reproductions
  and both-platform results). **Correction, see below**: that bring-up doc's claim that
  the cjalr ABI's `.capstone_cap_init` under-tags Lua's static tables is wrong (verified
  leaf-by-leaf with `-capstone-cap-init-print`) — cjalr was blocked by two absolute jump
  tables, not by under-tagging. `capstone/agent-handoff/history/06-08-2026_19-45-00_lua-runs-on-capstone-cjalr-jumptables.md`
  has the retraction and the fix.
- CHERI toolchain quirks captured in `cheri/real/build-real-lua-cdp.sh`: `-cheri-tgot-tls`
  (the purecap rtld rejects traditional TLS) and forced sysroot-only includes.
- The 13-case CDP corpus above runs on the **gp-captable** domain. A separate track got
  real Lua running on the simpler **cjalr** ABI (fixing the jump-table issue above),
  which is what runs the official `binary-trees` benchmark — see `PERF-MEMORY.md` for the
  CHERI-vs-Capstone performance and memory results on that workload.
