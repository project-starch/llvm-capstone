-- luaossl #124 — cross-domain double-free of an X509_STORE.
--
-- ctx:setStore(store) hands the X509_STORE to the SSL_CTX. On the vulnerable
-- tree luaossl's compat shim for SSL_CTX_set1_cert_store expands to the
-- OWNERSHIP-taking SSL_CTX_set_cert_store() (set0: no refcount bump), so ONE
-- X509_STORE ends up owned by TWO domains:
--   * the Lua-GC store userdata  (its __gc = xs__gc -> X509_STORE_free)
--   * the SSL_CTX                (SSL_CTX_free -> X509_STORE_free of cert_store)
-- Each frees it once -> double free / heap-use-after-free.
--
-- Upstream issue #124 reproducer (3 lines, crashes at process exit):
--     local ctx = require"openssl.ssl.context".new("TLS", true)
--     ctx:setStore(require"openssl.x509.store".new())
-- We drive the two frees explicitly (below) only to make the ORDER
-- deterministic and both owner stacks appear in one ASan report; the
-- double-ownership is identical to the 3-line form.

local ctx_mod   = require "openssl.ssl.context"
local store_mod = require "openssl.x509.store"

local ctx = ctx_mod.new("TLS", true)

do
	local store = store_mod.new()   -- X509_STORE, refcount 1, owned by this userdata
	ctx:setStore(store)             -- vulnerable set0: SSL_CTX now co-owns it, refcount STILL 1
end                                     -- 'store' userdata is now unreachable

collectgarbage("collect")               -- FIRST free: store userdata __gc (xs__gc) -> X509_STORE_free
ctx = nil
collectgarbage("collect")               -- SECOND free: ctx __gc (sx__gc) -> SSL_CTX_free -> X509_STORE_free

-- Reached only if the store was NOT doubly-owned (i.e. the fixed build, which
-- up-refs the store in setStore so each free just decrements). run.sh treats
-- reaching here WITHOUT an ASan report as the clean control.
print("NO-DOUBLE-FREE")
