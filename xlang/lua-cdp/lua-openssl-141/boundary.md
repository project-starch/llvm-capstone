# Boundary annotation — lua-openssl #141

### The object that crosses the boundary

A raw `EVP_CIPHER_CTX *`, allocated by OpenSSL (`libcrypto`), stored inside a Lua
full userdata created by `lua-openssl`'s cipher binding. The userdata **is** the
Lua-visible handle to the C object; the C pointer is what crosses.

### Owner vs. borrower

- **OpenSSL (native) owns the memory.** `EVP_CIPHER_CTX_new()` allocated it;
  `EVP_CIPHER_CTX_free()`/`_cleanup()` frees it.
- **Lua (managed) owns the handle lifetime.** The GC decides when the userdata
  is unreachable and runs its `__gc`.
- The bug: `lua-openssl` exposes **two** paths that both free the C object — the
  explicit `c:close()` and the userdata `__gc` — without the first invalidating
  the pointer the second reads.

### Free site (first)

`c:close()` → `openssl_cipher_ctx_free` → `EVP_CIPHER_CTX_free`
(`src/cipher.c:552`). The 168-byte C context is freed here (`CRYPTO_free`), but
the userdata's boxed pointer is **not** nulled.

### Stale-use site (second, one crossing later)

The Lua GC collects the userdata and runs its `__gc` metamethod (same
`openssl_cipher_ctx_free`). Its first line, `EVP_CIPHER_CTX_cleanup(ctx)` =
`EVP_CIPHER_CTX_reset(ctx)` (`src/cipher.c:551`), **reads the already-freed
ctx** → ASan `heap-use-after-free` (READ of size 8) in `EVP_CIPHER_CTX_reset`;
the following `EVP_CIPHER_CTX_free(ctx)` would re-free it (double free). The
finalizer stack is `openssl_cipher_ctx_free` (cipher.c:551) ← `GCTM` ←
`luaB_collectgarbage` — matching the upstream gdb backtrace, which ran on
OpenSSL 1.0.0 where the same function was named `EVP_CIPHER_CTX_cleanup`.

### The lifetime rule that is violated

A native resource wrapped by a Lua userdata must have **exactly one** owner of
its free. If an explicit close is offered, it must null/mark the wrapped pointer
so the finalizer becomes a no-op. Here the two owners disagree, so the object is
freed twice.

### Capability note (revoke-on-free)

On a revoke-on-free allocator the first free (`c:close()`) **revokes** the
capability to the `EVP_CIPHER_CTX` block. The `__gc` path then holds a revoked
capability: the second free — capability arithmetic / store through a revoked
cap — faults at the contract point instead of corrupting the allocator. This is
the boundary-only case: the free happens exactly at the C↔Lua handoff.
