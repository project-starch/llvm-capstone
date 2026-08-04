# Boundary annotation — luaossl #124

### The object that crosses the boundary

A raw `X509_STORE *`, allocated by OpenSSL (`libcrypto`, `X509_STORE_new`) and
wrapped by a Lua full userdata created by luaossl's `openssl.x509.store` binding.
`ctx:setStore(store)` passes that C pointer across into an `SSL_CTX`, storing it
as `ctx->cert_store`. The C pointer is what crosses; after the call it is
reachable from two owners.

### Owner vs. borrower

- **The Lua store userdata owns it (domain 1: Lua GC).** `openssl.x509.store.new()`
  allocated the `X509_STORE` with refcount 1; the userdata's `__gc` (`xs__gc`)
  calls `X509_STORE_free`.
- **The `SSL_CTX` owns it too (domain 2: OpenSSL aggregate).** `setStore`'s set0
  assignment made the `SSL_CTX` an owner of `ctx->cert_store`; `SSL_CTX_free`
  calls `X509_STORE_free` on it.
- The bug: `setStore` took ownership **without** an up-ref (`SSL_CTX_set_cert_store`,
  set0 semantics), so refcount stayed at 1 while two owners hold it. Each owner
  frees once → the block is freed twice.

### Free site (first)

The Lua store userdata becomes unreachable and the GC runs its `__gc`:
`xs__gc` (`src/openssl.c:7610`) → `X509_STORE_free` → refcount 1→0 →
`CRYPTO_free` releases the 152-byte block (`x509_lu.c:230`).

### Stale-use site (second, one crossing later)

The `SSL_CTX` userdata's `__gc` runs: `sx__gc` (`src/openssl.c:8493`) →
`SSL_CTX_free` (`ssl_lib.c:3262`) → `X509_STORE_free(ctx->cert_store)` →
`CRYPTO_DOWN_REF` **reads and writes the refcount word of the already-freed
store** (`x509_lu.c:212`) → ASan `heap-use-after-free` (WRITE of size 4); it
would then `free()` the block a second time. Both `__gc`s are driven by the Lua
GC (`GCTM` ← `luaB_collectgarbage`).

### The lifetime rule that is violated

A native resource shared between a Lua userdata and a native aggregate must have
its **refcount raised when a second owner adopts it**, or a single owner
designated. luaossl's set0 `setStore` adopted the store into the `SSL_CTX`
without raising the refcount, so the Lua GC domain and the OpenSSL `SSL_CTX`
domain both free the same block. The fix (`X509_STORE_up_ref` before the set0)
restores one-free-per-owner.

### Capability note (revoke-on-free)

On a revoke-on-free allocator the first free (the store userdata `__gc`) revokes
the capability to the `X509_STORE` block. `SSL_CTX_free` then holds a revoked
capability: its `CRYPTO_DOWN_REF` — a load/store through the revoked cap — faults
at the contract point instead of corrupting the allocator or racing on a stale
refcount. This is the shared-ownership case: two live capabilities to one block,
one of them held by the Lua-GC userdata, the other by the native `SSL_CTX`.
