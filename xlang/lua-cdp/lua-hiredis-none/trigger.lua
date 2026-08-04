-- lua-hiredis — strongest attempt at the described hard-tier CDP shape, which
-- DEMONSTRATES THE SHAPE IS ABSENT (this is a BLOCKED case, not a repro).
--
-- Shape we tried to trigger: a redisReply* (or nested reply->element[i]) held by
-- a Lua handle after the parent reply is freed by freeReplyObject -> UAF.
-- Result: agladysh/lua-hiredis DEEP-COPIES the reply into pure Lua values and
-- freeReplyObject()s it inside the C call (src/lua-hiredis.c:266,322,364), so the
-- value Lua holds owns no C memory. Under ASan, re-access after GC + heap churn
-- is a normal Lua table read, not a use-after-free.
local hiredis = require "hiredis"
local sock = assert(os.getenv("SOCK"), "SOCK env (redis unix socket) required")

-- (1) Reply path: get an ARRAY reply, drop every reference to its origin, force
--     GC + heavy heap churn to reuse any freed C block, then re-read it.
local c = assert(hiredis.connect(sock))
c:command("DEL", "lhk")
for i = 1, 5 do c:command("RPUSH", "lhk", "v" .. i) end
local reply = c:command("LRANGE", "lhk", 0, -1)      -- a Lua table copy
assert(type(reply) == "table" and #reply == 5, "reply must be a materialised Lua table")
collectgarbage("collect"); collectgarbage("collect")
local churn = {}; for i = 1, 5000 do churn[i] = string.rep("x", 64) end
-- If `reply` were a handle into a freed redisReply*, THIS would UAF. It is a copy:
assert(reply[1] == "v1" and reply[5] == "v5", "reply is a live Lua copy, not a dangling ptr")
io.write("OK-reply-is-a-lua-copy\n")

-- (2) Context path: explicit close() frees redisContext*; the userdata __gc is the
--     SAME null-guarded function (lua-hiredis.c:409,424) -> second free is a no-op.
c:close()                                            -- redisFree + pContext=NULL
c = nil
collectgarbage("collect"); collectgarbage("collect") -- __gc runs lconn_close again
io.write("OK-context-not-double-freed\n")
