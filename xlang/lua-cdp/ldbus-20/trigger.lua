-- ldbus #20 — an iterator userdata references a DBusMessage (reply) that is
-- freed when the reply's own wrapper is GC'd; the iterator has no refcount on it
-- (fixed by 5cc933b, which makes the iter hold a reference).
--
-- Detection: libdbus POOLS freed message headers, so a stale read stays readable
-- rather than trapping. We force the pool slot to be reused by fresh messages,
-- then read the iterator: on the vulnerable tree it reads the reused (empty)
-- message -> get_arg_type()=nil; the fixed tree keeps the reply -> 'a'.
local ldbus = require "ldbus"
collectgarbage("stop")
local conn = ldbus.bus.get("session")
local function mk() return ldbus.message.new_method_call(
  "org.freedesktop.DBus","/org/freedesktop/DBus","org.freedesktop.DBus","ListNames") end
local iter = conn:send_with_reply_and_block(mk()):iter_init()   -- reply is unstored
collectgarbage("collect")                     -- frees the reply -> dbus_message_unref
local churn = {}; for i=1,200 do churn[i] = mk() end            -- reuse the pooled slot
io.write("arg_type=", tostring(iter:get_arg_type()), "\n")
