# Lua-cURLv3 (lcurl) — easy userdata ⟷ collected multi handle use-after-free

**One line.** `multi:add_handle(easy)` stores a raw C back-pointer
`easy->multi = multi`, but nothing on the Lua side keeps the multi reachable
*from* the easy; when the last Lua reference to the multi is dropped, the GC
frees the multi userdata while the easy still holds `easy->multi`, and the next
`easy:close()` writes through that dangling pointer → cross-domain UAF.

## Identity

| | |
|---|---|
| Library | [`Lua-cURLv3`](https://github.com/Lua-cURL/Lua-cURLv3) (`lcurl`, moteus) |
| Language pair | **C ⟷ Lua** (reference Lua 5.4) |
| Upstream | fix merged via PR [#80](https://github.com/Lua-cURL/Lua-cURLv3/pull/80); umbrella easy/multi-GC-lifetime report is issue [#5](https://github.com/Lua-cURL/Lua-cURLv3/issues/5) ("memory leak, easy gc not work") |
| CVE / GHSA | none assigned |
| Native library | system **libcurl 8.18.0** (used as-is; NOT rebuilt) |
| Vulnerable commit | **`b2e9474c6bf975a0f1813a387e2c0ec21d3064f3`** (parent of the fix) |
| Fix commit | **`56b4d05c17b406d790721b1921314a5da7da2c58`** — "Fix. Cleanup easy references when calls multi::close": `lcurl_multi_cleanup` now walks the multi's handle table and nulls each `e->multi` before the userdata is freed. (Complementary direction — `easy:close` removing itself from the multi — is `78a4a03`.) |

## The two coupled objects (why this is a two-object CDP)

1. **Lua-GC handle:** the **easy** userdata. Its C struct `lcurl_easy_t` carries
   a back-pointer `multi` (`lcurl_multi_t *`), set by `multi:add_handle`.
2. **Separate Lua-GC object:** the **multi** userdata — a distinct 96-byte
   `lcurl_multi_t` block (its own `lua_newuserdata`) that wraps a native
   `CURLM*`.

Two distinct allocations; the cross-domain pointer is `easy->multi`. This is
**not** the excluded "raw pointer into a Lua string" shape — it is a C
back-pointer from one Lua-GC object into a second, separately-allocated Lua-GC
object, dereferenced after that second object is collected.

**Direction:** GC-frees. Dropping the multi's last Lua ref collects the multi
userdata (crossing 1); `easy:close()` then dereferences the stale `easy->multi`
(crossing 2) → UAF write.

## Reproduction status

**REPRODUCED (2026-08-04), with control.** See the TIER CAVEAT below before
counting this as a canonical native-resource case.

- Env: system libcurl 8.18.0 (not rebuilt), PUC Lua 5.4.7 (shared toolchain),
  gcc 15 ASan on `lcurl.so`.
- Vulnerable `b2e9474`: ASan **heap-use-after-free**, `WRITE of size 8` in
  `lcurl_easy_cleanup` (`src/lceasy.c:87`, the `p->multi->L = L;` store),
  freed by Lua's GC allocator `l_alloc` (the collected multi userdata),
  allocated by `luaM_malloc_` (`curl.multi()`'s `lua_newuserdata`).
- Control, fixed `56b4d05`: **no ASan report** — trigger prints `NO-UAF`.
- `./build.sh && ./run.sh` → PASS. Full trace + control in `evidence.txt`.

## PASS signature

`run.sh` passes iff **both** halves of the differential hold:

- **Vulnerable `b2e9474`:** ASan reports `heap-use-after-free` **AND**
  `lcurl_easy_cleanup` **AND** `src/lceasy.c` **AND** `l_alloc` (the last
  proving the freed block was released by Lua's GC — i.e. it is the collected
  *multi userdata*, establishing the cross-object coupling).
- **Fixed `56b4d05` (control):** `NO-UAF` printed and **no** `AddressSanitizer`
  report.

Either half missing = FAIL.

## TIER CAVEAT — read before slotting this into the canonical 15

This is a genuine two-object cross-domain-pointer UAF, but it is **spare-tier**,
directly analogous to `luv-503`, **not** a canonical native-resource case:

1. **The freed block is Lua-managed userdata memory** (the `lcurl_multi_t`
   handle), freed by Lua's GC (`l_alloc`) — not a curl-owned native heap object.
   The ASan report is on the userdata payload (`p->multi->L`), not on the
   `CURLM*`. So the "wraps a *separate native C resource*" bar is met only in
   the loose sense that the multi userdata wraps a `CURLM*`.

2. **The pure-native variant is masked by libcurl and does NOT reproduce.** The
   textbook native shape — *an easy handle freed while a multi still references
   its `CURL*`* — is **not** a UAF on any relevant libcurl: `curl_easy_cleanup`
   → `Curl_close()` auto-calls `curl_multi_remove_handle()` when the handle is
   still attached (verified in libcurl source since **≤ 7.50**, and empirically
   clean under valgrind on 8.18). The same defensiveness masks the mime↔easy
   and share↔easy couplings on 8.18. That native self-protection is exactly why
   the only reproducible bug lives in **lcurl's own bookkeeping** (`e->multi`),
   not in libcurl.

3. **Not cleanly filed to a single UAF issue.** The fix is a bundled 2016
   lifetime-hardening commit (`56b4d05`, merged in PR #80); the nearest filed
   *issue* (#5) foregrounds the leak symptom. The bug is "filed" as a fix
   commit, not as a dedicated UAF report.

The parent decides whether to keep this as a `luv-503`-style verified spare or
reassign the libcurl slot. What is solid: the differential is real, minimal, and
reproduced with our own ASan trace; and the negative result (all *native*
libcurl handle-lifetime couplings are self-protected on 8.18) is itself a
firm finding.

## Dependencies / build

- System **libcurl 8.18.0** via `pkg-config libcurl` (used as-is).
- Shared reference **Lua 5.4.7** from `../_toolchain` (read-only).
- One bug-unrelated build shim: the 2016 source's `CURLE_SSL_CACERT` switch case
  collides with `CURLE_PEER_FAILED_VERIFICATION` (a deprecated alias in
  libcurl ≥ 8) — `build.sh` drops that single error-string line. It is not on
  the easy/multi lifetime path. (Analogous to lua-openssl #141's
  `-DOPENSSL_NO_SM2`.)
