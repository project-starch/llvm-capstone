-- LMDB value-lifetime CDP UAF (documented-contract reproduction).
--
-- lmdb.h:249-251 / :1275-1276 — "Values returned from the database are valid
-- only until a subsequent update operation, or the end of the transaction."
--
-- txn:get(k) returns a DEFERRED value handle that borrows a pointer into the
-- transaction's page (no copy). The Lua handle lives on across the txn end; one
-- crossing later (val:read()) it dereferences a page the txn already freed.
--
--   arg[1] == "control" : read the handle BEFORE the txn ends  -> clean.
--   otherwise (vuln)    : commit the txn, THEN read the handle  -> heap-UAF.
--
-- The value is multi-page on purpose: LMDB free()s a multi-page overflow buffer
-- outright at txn end (small values are pooled, which would mask the free).
local mdb = require "minilmdb"

local control = arg[1] == "control"
-- Deterministic single-file DB path (MDB_NOSUBDIR). Fixed per mode and cleaned
-- up front, so a vuln run (which ASan aborts before its own cleanup) leaves at
-- most one stale file that the next run overwrites, instead of accumulating.
local path = "/tmp/minilmdb-" .. (control and "control" or "vuln") .. ".mdb"
os.remove(path); os.remove(path .. "-lock")       -- start fresh; let LMDB create it

local KEY = "k"
local VAL = string.rep("LMDB-overflow-", 20000)   -- ~280 KB -> multi-page overflow

local env = mdb.open(path)
local txn = env:begin()
txn:put(KEY, VAL)
local h = txn:get(KEY)                             -- borrowed pointer into the page

local out
if control then
  out = h:read()                                  -- read while the txn is live: valid
  txn:commit()
else
  txn:commit()                                    -- END OF TRANSACTION: page freed
  out = h:read()                                  -- borrowed pointer now dangles -> UAF
end

-- Only reached when the read did not trap (always, for control).
io.write(string.format("read ok=%s len=%d head=%s\n",
  tostring(out == VAL), #out, out:sub(1, 14)))

os.remove(path); os.remove(path .. "-lock")
