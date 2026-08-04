# luaossl #124 — Lua userdata ⟷ C `X509_STORE` cross-domain double-free

**One line.** `ctx:setStore(store)` hands an `X509_STORE` to an `SSL_CTX` with
ownership-taking (set0) semantics that forget to bump the refcount, so ONE
`X509_STORE` ends up owned by BOTH the Lua-GC store userdata (whose `__gc` calls
`X509_STORE_free`) AND the `SSL_CTX` (whose `SSL_CTX_free` also frees it) — a
cross-domain double-free.

## Identity

| | |
|---|---|
| Library | [`luaossl`](https://github.com/wahern/luaossl) (wahern) — **new library for this corpus** |
| Language pair | **C ⟷ Lua** (reference Lua 5.4) |
| Upstream | https://github.com/wahern/luaossl/issues/124 |
| CVE / GHSA | none assigned |
| Native library | OpenSSL 1.1.1w `libcrypto`/`libssl` (built from source; see below) |
| Vulnerable commit | **`5be1b44a6a60f32c660cc4ee09d60e676cd8c81a`** (parent of the fix) |
| Fix commit | **`1ae707300bf99805bd93744020c60cf60cdc2294`** — "Fix SSL_CTX_set1_cert_store refcounting issues. Closes #124": adds `X509_STORE_up_ref(store)` before the set0 assignment. |

## The two coupled objects (why this is unambiguous CDP)

1. **Lua-GC handle:** the `x509.store` userdata returned by
   `openssl.x509.store.new()`. Its `__gc` (`xs__gc`) calls `X509_STORE_free`.
2. **Native aggregate that co-owns the same C struct:** the `SSL_CTX` created by
   `openssl.ssl.context.new(...)`. `setStore` stores our `X509_STORE` as
   `ctx->cert_store`; `SSL_CTX_free` frees `cert_store`.

One heap `X509_STORE` (a libcrypto C struct), two owners in two lifetime
domains. This is **not** the excluded "raw pointer into a Lua string" shape — the
freed object is a libcrypto struct, and one of the two owners is a Lua-GC
userdata.

**Direction:** ownership-confusion double-free. `setStore` uses set0 semantics
without an up-ref, so the Lua store userdata and the `SSL_CTX` each believe they
solely own the `X509_STORE`; each frees it once → double free.

## Why the bug is present (and only on OpenSSL ≥ 1.1.0)

The defect is in luaossl's own compat shim for `SSL_CTX_set1_cert_store`. At
`5be1b44`, luaossl never detects that OpenSSL supplies `SSL_CTX_set1_cert_store`
natively (`config.h.guess` probes only C-language attributes, so
`HAVE_SSL_CTX_set1_cert_store` is undefined → `HAVE_SSL_CTX_SET1_CERT_STORE == 0`).
So the shim is **always** compiled. On OpenSSL ≥ 1.1.0 the `X509_STORE` struct is
opaque, so `HAVE_X509_STORE_REFERENCES == !OPENSSL_PREREQ(1,1,0) == 0`, and the
shim collapses to the ownership-taking primitive:

```c
#define SSL_CTX_set1_cert_store(ctx, store) SSL_CTX_set_cert_store((ctx),(store))
```

`SSL_CTX_set_cert_store()` is set0 — it takes ownership and does **not** bump the
store's refcount. That is exactly what `sx_setStore` (`src/openssl.c:8062`) calls.
The fix keeps the shim but prepends `X509_STORE_up_ref(store)`.

## Dependencies

- **OpenSSL 1.1.1w, built from source** (`build.sh`), **shared + `-fsanitize=address`**.
  Two reasons (both in `build.sh`): the 2018 vulnerable commit does not compile
  against this box's OpenSSL 3.5 (`SHLIB_VERSION_HISTORY`, `RSA_SSLV23_PADDING`,
  … were removed in 3.0), and 1.1.1w is ≥ 1.1.0 so the buggy compat path is
  reached. libcrypto is ASan-built because the second free reads/writes the freed
  store's refcount inside libcrypto *before* the `free()` ASan intercepts;
  uninstrumented, that is a nondeterministic SEGV, instrumented it is the labelled
  heap-use-after-free.
- The shared reference **Lua 5.4.7** from `../_toolchain` (read-only).
- `luaossl` built with AddressSanitizer.

## Reproduction status

**REPRODUCED (2026-08-04), with control.**

- Env: OpenSSL 1.1.1w (from source, ASan), PUC Lua 5.4.7 (shared toolchain),
  gcc 15.2 ASan.
- Vulnerable `5be1b44`: ASan **heap-use-after-free**, `WRITE of size 4` in
  `CRYPTO_DOWN_REF` (`X509_STORE_free`, `x509_lu.c:212`) from `SSL_CTX_free`
  (`ssl_lib.c:3262`) run by the **SSL_CTX** userdata `__gc` (`sx__gc`,
  `src/openssl.c:8493`). The 152-byte `X509_STORE` was **freed** first by
  `X509_STORE_free` (`x509_lu.c:230`) from the **Lua store userdata** `__gc`
  (`xs__gc`, `src/openssl.c:7610`), and **allocated** by `X509_STORE_new`
  (`x509_lu.c:162`) ← `xs_new` (`src/openssl.c:7459`). Both frees driven by the
  Lua GC (`GCTM` ← `luaB_collectgarbage`).
- Control, fixed `1ae7073`: **no ASan report** — trigger prints `NO-DOUBLE-FREE`.
- Full trace + control in `evidence.txt`.

## Is the Lua-GC object essential? (resolving the required nuance)

**Yes.** One of the two frees is literally `xs__gc`, the `__gc` finalizer of the
Lua `x509.store` userdata — the `X509_STORE` is owned by the Lua GC domain (the
store userdata) *and* the OpenSSL `SSL_CTX` domain, and the double-free is the
collision of the two at GC time. The underlying set0 primitive could also be
misused from pure C, but the filed bug #124 and this reproduction route one
ownership through the Lua store userdata; that cross-domain co-ownership is the
point. **This qualifies as a cross-domain double-free.**

## PASS signature

`run.sh` passes iff **both** halves of the differential hold:

- **Vulnerable `5be1b44`:** ASan reports `heap-use-after-free` on the
  `X509_STORE`, with BOTH owners present in the report:

  ```
  #1 X509_STORE_free                 x509_lu.c:212   (via CRYPTO_DOWN_REF)
  #2 SSL_CTX_free                    ssl_lib.c:3262
  #3 sx__gc                          src/openssl.c:8493   (SSL_CTX __gc — 2nd free)
  ...
  #8 GCTM                            (the Lua __gc runner)
  freed by:  X509_STORE_free <- xs__gc  src/openssl.c:7610  (store userdata __gc — 1st free)
  ```

  Concretely run.sh requires `heap-use-after-free` **AND** `X509_STORE_free`
  **AND** `in xs__gc` **AND** `in sx__gc` **AND** `SSL_CTX_free` **AND** `GCTM`.

- **Fixed `1ae7073` (control):** no ASan report; the trigger runs to completion
  and prints `NO-DOUBLE-FREE`.

Either half missing = FAIL. The first free is the Lua store userdata (`xs__gc`);
the second is `SSL_CTX_free`. The fix up-refs the store so each free just
decrements and only the second releases it.
