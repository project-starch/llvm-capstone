# lua-openssl #141 — Lua userdata ⟷ C `EVP_CIPHER_CTX` use-after-free

**One line.** A Lua userdata wraps an OpenSSL `EVP_CIPHER_CTX`; an explicit
`c:close()` frees the C context, but the userdata keeps the stale pointer, so the
userdata's `__gc` frees it a second time — a cross-domain double-free.

## Identity

| | |
|---|---|
| Library | [`lua-openssl`](https://github.com/zhaozg/lua-openssl) (zhaozg) |
| Language pair | **C ⟷ Lua** (reference Lua 5.1–5.4) |
| Upstream | https://github.com/zhaozg/lua-openssl/issues/141 (filed 2018-06-08) |
| CVE / GHSA | none assigned |
| Native library | OpenSSL 1.1.1w `libcrypto` (built from source; see below) |
| Vulnerable commit | **`0017afa23dcbbab91a307cd6ae07f60f7427e02f`** (parent of the fix) |
| Fix commit | **`a436c363aa6f963c48ce3c103e16b941ebbfea45`** — "fix issue #141": adds `FREE_OBJECT(1)` (nulls the boxed pointer) + a `if(!ctx) return 0;` guard in `openssl_cipher_ctx_free`. |

## The two coupled objects (why this is unambiguous CDP)

1. **Lua-GC handle:** the cipher userdata returned by `openssl.cipher.*_new(...)`.
2. **Separate native resource:** a heap `EVP_CIPHER_CTX` allocated by libcrypto,
   whose pointer the userdata stores.

Two distinct allocations. This is **not** the excluded "raw pointer into a Lua
string" shape — the freed object is a libcrypto C struct, not Lua-VM internals.

**Direction:** native-frees. `c:close()` frees the C ctx (crossing 1); the
userdata's `__gc` later frees the same, now-stale pointer (crossing 2) → double
free.

## Dependencies

- **OpenSSL 1.1.1w, built from source** (`build.sh`). This box ships OpenSSL
  3.5, whose API the 2018 vulnerable commit no longer compiles against; 1.1.1w
  is the newest release its code builds on. It is built **shared +
  `enable-crypto-mdebug` + `-fsanitize=address`**.
- The shared reference **Lua 5.4.7** from `../_toolchain` (read-only).
- `lua-openssl` built with AddressSanitizer. libcrypto is *also* ASan-built —
  necessary here because the second free runs through `EVP_CIPHER_CTX_reset`,
  which reads the freed ctx *before* the `free()` ASan intercepts. Uninstrumented
  libcrypto turns that read into a bare SEGV; instrumenting it yields the labelled
  heap-use-after-free with alloc/free/use stacks (see `build.sh` for the why).

## Reproduction status

**REPRODUCED (2026-08-03), with control.**

- Env: OpenSSL 1.1.1w (from source, ASan + crypto-mdebug), PUC Lua 5.4.7
  (shared toolchain), gcc 15.2 ASan.
- Vulnerable `0017afa`: ASan **heap-use-after-free**, `READ of size 8` in
  `EVP_CIPHER_CTX_reset` (`evp_enc.c:26`) from `openssl_cipher_ctx_free`
  (`cipher.c:551`) run by the userdata `__gc` (`GCTM` ← `luaB_collectgarbage`).
  The 168-byte `EVP_CIPHER_CTX` was **freed** by the same function at
  `cipher.c:552` (`EVP_CIPHER_CTX_free`) during `c:close()`, and **allocated**
  by `EVP_CIPHER_CTX_new` ← `openssl_cipher_decrypt_new` (`cipher.c:400`).
- Control, fixed `a436c36`: **no ASan report** — trigger prints `NO-DOUBLE-FREE`.
- Full trace + control in `evidence.txt`.

## PASS signature

`run.sh` passes iff **both** halves of the differential hold:

- **Vulnerable `0017afa`:** ASan reports `heap-use-after-free` on the
  `EVP_CIPHER_CTX`, with the second free reached from the GC finalizer:

  ```
  #0 EVP_CIPHER_CTX_reset           (via EVP_CIPHER_CTX_cleanup)
  #1 openssl_cipher_ctx_free        src/cipher.c:551
  ...
  #6 GCTM                           (the Lua __gc metamethod)
  #11 luaB_collectgarbage
  freed by:  EVP_CIPHER_CTX_free <- openssl_cipher_ctx_free  src/cipher.c:552  (c:close())
  ```

  Concretely run.sh requires `heap-use-after-free` **AND**
  `openssl_cipher_ctx_free src/cipher.c:551` **AND** `GCTM` **AND**
  `EVP_CIPHER_CTX_free` in the output.

- **Fixed `a436c36` (control):** no ASan report; the trigger runs to completion
  and prints `NO-DOUBLE-FREE`.

Either half missing = FAIL. The first free is `c:close()`; the second is the
userdata `__gc`. The fix nulls the boxed pointer so `__gc` frees `NULL`.
