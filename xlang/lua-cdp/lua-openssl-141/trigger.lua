-- lua-openssl #141 — cross-domain double-free of an EVP_CIPHER_CTX.
--
-- Two owners of one C object: the explicit c:close() and the userdata __gc.
-- close() frees the EVP_CIPHER_CTX but leaves the userdata's pointer dangling;
-- the GC finalizer then frees it a second time.
--
-- Minimal reproducer from upstream issue #141 (verbatim shape).

local openssl = require('openssl')

local c = openssl.cipher.decrypt_new('bf-cbc', 'secret_key', 'iv')
c:update('msg')
c:final()

c:close()            -- FIRST free: EVP_CIPHER_CTX freed, pointer NOT nulled
c = nil
collectgarbage('collect')   -- __gc runs -> SECOND free of the same pointer

-- If we get here without a double-free abort, the bug did NOT reproduce
-- (e.g. a fixed build that nulls the pointer). run.sh treats that as FAIL.
print('NO-DOUBLE-FREE')
