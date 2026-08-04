-- LuaDBI #35 — cross-domain use-after-free of a native PostgreSQL PGconn*.
--
-- Two owners of one native handle:
--   * the connection userdata owns the PGconn* (frees it in db:close()).
--   * the statement userdata holds its OWN raw copy of that same PGconn*.
-- db:close() runs PQfinish() which free()s the PGconn, but the statement's
-- copy is left dangling; stmt:execute() then dereferences it via PQstatus().
--
-- Adapted to PostgreSQL from the upstream issue #35 repro (which was filed
-- against the SQLite3 driver): connect -> prepare -> close -> execute.
--
-- Connection parameters come from the ephemeral server run.sh starts.

local DBI = require "DBI"

local dbname = os.getenv("LUADBI_DB")   or "postgres"
local user   = os.getenv("LUADBI_USER") or "luadbi"
local host   = os.getenv("LUADBI_HOST")            -- unix-socket directory
local port   = tonumber(os.getenv("LUADBI_PORT") or "5432")

local db = assert(DBI.Connect("PostgreSQL", dbname, user, nil, host, port))
local stmt = assert(db:prepare("SELECT 1"))

db:close()          -- FIRST crossing: PQfinish() frees the PGconn;
                    -- the statement's raw copy is NOT nulled -> dangling.

-- SECOND crossing: statement_execute() reads the freed PGconn via PQstatus():
--   vulnerable (f562ccd~1) -> heap-use-after-free; ASan halts here.
--   fixed      (f562ccd)   -> reads conn->postgresql == NULL, so PQstatus(NULL)
--                             is CONNECTION_BAD and it raises "statement broken".
local ok, err = pcall(function() return stmt:execute() end)

-- Only reached on a build that did NOT use-after-free (the fixed control):
-- the vulnerable build aborts inside execute() under ASan before returning.
print("NO-UAF ok=" .. tostring(ok) .. " err=" .. tostring(err))
