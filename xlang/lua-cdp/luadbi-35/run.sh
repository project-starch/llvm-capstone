#!/usr/bin/env bash
# CHECK (not just a runner) for LuaDBI #35, by DIFFERENTIAL:
#   VULNERABLE f562ccd~1 -> ASan heap-use-after-free of the PGconn: db:close()
#                           runs PQfinish() (free), and the statement's stale raw
#                           copy is dereferenced by PQstatus() in stmt:execute().
#   FIXED      f562ccd    -> clean run (prints NO-UAF), no ASan report: the
#                           statement now refers to the connection object, so it
#                           reads conn->postgresql==NULL and raises cleanly.
# PASS only if BOTH hold. That differential is what makes this a verified
# reproduction of the cross-domain use-after-free, not an incidental crash.
#
# The UAF is entirely client-side; ASan (preloaded into the Lua process)
# intercepts libpq's free() of the PGconn, so an ordinary (-g, un-instrumented)
# libpq is enough to yield a labelled heap-use-after-free with symbolised
# PQfinish / PQstatus frames. run.sh spins up an ephemeral PostgreSQL on a unix
# socket (needed only so db:prepare() can create a real statement) and tears it
# down afterwards.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"
PG="$W/pg-install"
REPO="$W/luadbi"
LUA54=$(cd -- "$HERE/../_toolchain/.work/lua54" 2>/dev/null && pwd || true)
ASAN=$(cc -print-file-name=libasan.so)

VULN=$(git -C "$REPO" rev-parse f562ccd~1 2>/dev/null)   # raw PGconn* copy in the statement
FIX=f562ccdc93c4068db3350d76ed6baf0848c51aaf             # statement refers to the connection object

[ -x "$LUA54/lua-shared" ]        || { echo "run ./build.sh first (no shared lua)" >&2; exit 2; }
[ -x "$PG/bin/initdb" ]           || { echo "run ./build.sh first (no PostgreSQL)" >&2; exit 2; }
[ -f "$PG/include/libpq-fe.h" ]   || { echo "run ./build.sh first (no libpq headers)" >&2; exit 2; }
[ -d "$REPO/.git" ]               || { echo "run ./build.sh first (no LuaDBI)" >&2; exit 2; }
[ -n "$VULN" ]                    || { echo "cannot resolve vulnerable commit f562ccd~1" >&2; exit 1; }

# ---- minimal Lua 5.1 compat shim (build-level, not a source edit) ----------
# LuaDBI @2016 uses the 5.1-only name luaL_checkint (removed in Lua 5.3) at one
# connect-time site (the port argument, off the UAF path). Its own
# `#include <compat-5.1.h>` is #if LUA_VERSION_NUM<501-guarded, so on Lua 5.4 it
# is compiled out and nothing backfills the name. We force-inject the one macro
# with gcc -include, without touching the binding source. Its function
# *registration* is separately #if-guarded (luaL_register vs luaL_setfuncs), so
# nothing else is needed. This is the analogue of lua-openssl-141's
# -DOPENSSL_NO_SM2 build flag: it touches the build environment, not the code.
mkdir -p "$W/shim"
cat > "$W/shim/luadbi_compat.h" <<'EOF'
/* build-only shim, force-included via gcc -include (see run.sh) */
#include <lauxlib.h>
#ifndef luaL_checkint
#define luaL_checkint(L,n)   ((int)luaL_checkinteger((L),(n)))
#endif
#ifndef luaL_optint
#define luaL_optint(L,n,d)   ((int)luaL_optinteger((L),(n),(d)))
#endif
EOF

build_at() { # $1 = commit -> builds an ASan dbdpostgresql.so in $REPO
  git -C "$REPO" checkout -q "$1" || return 1
  rm -f "$REPO/dbdpostgresql.so"
  # -std=gnu17: gcc 15 defaults to C23; keep the 2016 driver on C17.
  cc -g -O0 -std=gnu17 -fsanitize=address -fno-omit-frame-pointer -shared -fPIC \
     -include "$W/shim/luadbi_compat.h" \
     -I"$LUA54" -I"$PG/include" -I"$REPO" \
     -Wno-error=implicit-function-declaration \
     "$REPO/dbd/common.c" "$REPO/dbd/postgresql/main.c" \
     "$REPO/dbd/postgresql/connection.c" "$REPO/dbd/postgresql/statement.c" \
     -o "$REPO/dbdpostgresql.so" \
     -L"$PG/lib-asan" -Wl,-rpath,"$PG/lib-asan" -lpq -fsanitize=address \
     >"$W/build.$1.log" 2>&1 \
     || { echo "build failed for $1 (see $W/build.$1.log)" >&2; return 1; }
  [ -f "$REPO/dbdpostgresql.so" ] || { echo "no dbdpostgresql.so for $1" >&2; return 1; }
}

# ---- ephemeral PostgreSQL on a unix socket --------------------------------
PGDATA="$W/pgdata"; SOCK="$W/s"; PGPORT=54329; PGLOG="$W/pg.log"
pg_stop() { [ -d "$PGDATA" ] && "$PG/bin/pg_ctl" -D "$PGDATA" -m immediate stop >/dev/null 2>&1 || true; }
pg_start() {
  pg_stop; rm -rf "$PGDATA" "$SOCK"; mkdir -p "$SOCK"
  LD_LIBRARY_PATH="$PG/lib" "$PG/bin/initdb" -D "$PGDATA" -A trust -U luadbi -N \
    >"$W/initdb.log" 2>&1 || { echo "initdb failed (see $W/initdb.log)" >&2; return 1; }
  LD_LIBRARY_PATH="$PG/lib" "$PG/bin/pg_ctl" -D "$PGDATA" -l "$PGLOG" -w -t 60 \
    -o "-p $PGPORT -k $SOCK -c listen_addresses=''" start \
    >"$W/pgctl.log" 2>&1 || { echo "pg_ctl start failed (see $PGLOG)" >&2; return 1; }
}
trap pg_stop EXIT

run_trigger() { # -> stdout+stderr of the trigger under ASan (client uses ASan libpq)
  LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$PG/lib-asan:$LUA54" \
    LUA_CPATH="$REPO/?.so;;" LUA_PATH="$REPO/?.lua;;" \
    LUADBI_DB=postgres LUADBI_USER=luadbi LUADBI_HOST="$SOCK" LUADBI_PORT="$PGPORT" \
    ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" \
    "$LUA54/lua-shared" "$HERE/trigger.lua" 2>&1
}

echo "== starting ephemeral PostgreSQL =="; pg_start || exit 1

VOUT="$W/vuln.out"; FOUT="$W/fix.out"
echo "== vulnerable ($VULN) =="; build_at "$VULN" || exit 1; run_trigger >"$VOUT" 2>&1; head -20 "$VOUT"
echo "== fixed ($FIX) ==";      build_at "$FIX"  || exit 1; run_trigger >"$FOUT" 2>&1; tail -3 "$FOUT"
git -C "$REPO" checkout -q "$VULN"   # leave the case on the vulnerable tree
pg_stop

# Vulnerable: heap-use-after-free of the PGconn, USED by statement_execute
# (PQstatus) and FREED by connection_close (PQfinish) — our own captured trace.
vuln_uaf=0
if grep -qa 'heap-use-after-free' "$VOUT" \
   && grep -qa 'statement_execute' "$VOUT" \
   && grep -qa 'PQfinish' "$VOUT" \
   && grep -qa 'connection_close' "$VOUT"; then vuln_uaf=1; fi

# Fixed: no ASan report AND the trigger ran to completion.
fix_clean=0
if grep -qa 'NO-UAF' "$FOUT" && ! grep -qa 'AddressSanitizer' "$FOUT"; then fix_clean=1; fi

echo "--- verdict: vuln_uaf=$vuln_uaf fix_clean=$fix_clean ---"
if [ "$vuln_uaf" = 1 ] && [ "$fix_clean" = 1 ]; then
  echo "PASS: LuaDBI #35 reproduced (heap-use-after-free of the PGconn on $VULN via stmt:execute() after db:close(); clean on $FIX)"
  exit 0
fi
echo "FAIL: differential not satisfied (vuln_uaf=$vuln_uaf fix_clean=$fix_clean)" >&2
exit 1
