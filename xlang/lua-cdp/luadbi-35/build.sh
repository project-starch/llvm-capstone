#!/usr/bin/env bash
# Build the prerequisites for LuaDBI #35 (PostgreSQL driver native-handle UAF):
#   1. PostgreSQL from source into ./.work/pg-install, built with -g (debug
#      info) so the ASan report's free/alloc frames through libpq are symbolised.
#      This one source build yields BOTH the client library libpq AND the server
#      binaries (initdb/postgres/pg_ctl) that run.sh spins up on a unix socket.
#   2. a clone of LuaDBI into ./.work/luadbi.
# The reference Lua 5.4 comes from the shared toolchain (read-only), not here.
# Idempotent; everything lands under ./.work.
#
# WHY PostgreSQL from source (not the system package / not apt):
#   this box ships no postgres server, no pg_config and no libpq-fe.h (only the
#   bare libpq.so.5 runtime).  Building from source is the same discipline the
#   corpus's lua-openssl-141 case uses for OpenSSL 1.1.1 — a self-contained C
#   dependency, no apt-install, no docker in the reproduction path.
#
# WHY NOT ASan inside postgres: the UAF is entirely CLIENT-side — db:close()
#   runs PQfinish() which free()s the PGconn in the Lua process, and the stale
#   statement then reads it via PQstatus() in the same process.  ASan (preloaded
#   into the client by run.sh) intercepts that free() globally, so an
#   uninstrumented-but-debug (-g) libpq is enough to get a labelled
#   heap-use-after-free with symbolised PQfinish/PQstatus frames.  The server is
#   a separate, ordinary process and stays out of ASan entirely.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"; mkdir -p "$W"

LUA54=$(cd -- "$HERE/../_toolchain/.work/lua54" 2>/dev/null && pwd || true)
[ -n "$LUA54" ] && [ -x "$LUA54/lua-shared" ] || {
  echo "BLOCKED: shared Lua 5.4 toolchain not found at ../_toolchain/.work/lua54/lua-shared" >&2
  exit 2; }
echo "using shared Lua 5.4 at: $LUA54"

PG_VER=16.4
PG_TAR="$W/postgresql-$PG_VER.tar.gz"
PG_SRC="$W/postgresql-$PG_VER"
PG_PREFIX="$W/pg-install"
LIBPQ="$PG_PREFIX/lib/libpq.so"

have_pg() {
  [ -x "$PG_PREFIX/bin/initdb" ] && [ -x "$PG_PREFIX/bin/postgres" ] \
    && [ -x "$PG_PREFIX/bin/pg_ctl" ] && [ -f "$LIBPQ" ] \
    && [ -f "$PG_PREFIX/include/libpq-fe.h" ] \
    && [ -f "$PG_PREFIX/lib-asan/libpq.so" ]
}

# ---- 1. PostgreSQL (server + libpq, debug info) ---------------------------
if have_pg; then
  echo "PostgreSQL $PG_VER already built at $PG_PREFIX"
else
  echo "== fetching + building PostgreSQL $PG_VER (a few minutes) =="
  if [ ! -f "$PG_TAR" ]; then
    wget -q -O "$PG_TAR" \
      "https://ftp.postgresql.org/pub/source/v$PG_VER/postgresql-$PG_VER.tar.gz" \
      || curl -fsSL -o "$PG_TAR" \
         "https://ftp.postgresql.org/pub/source/v$PG_VER/postgresql-$PG_VER.tar.gz" \
      || { echo "BLOCKED: cannot download PostgreSQL $PG_VER (no network?)" >&2; exit 2; }
  fi
  rm -rf "$PG_SRC"; tar -C "$W" -xzf "$PG_TAR"
  ( cd "$PG_SRC"
    # readline/zlib/icu are not present on this box; none is needed for the
    # driver, the server, or the UAF path.
    # -std=gnu17: gcc 15 defaults to C23 where `bool` is a keyword, which breaks
    # PostgreSQL 16.4's `typedef unsigned char bool;` (src/include/c.h) — pin C17.
    ./configure --prefix="$PG_PREFIX" \
      --without-readline --without-zlib --without-icu \
      CFLAGS="-g -O1 -fno-omit-frame-pointer -std=gnu17" >config.log 2>&1
    make -j"$(nproc)" >build.log 2>&1
    make install >install.log 2>&1 )
  # Additionally build an ASan libpq for the CLIENT into $PG_PREFIX/lib-asan.
  # WHY: the UAF read is PQstatus(), which lives *inside* libpq — an
  # uninstrumented libpq means that read is never shadow-checked, so ASan sees
  # nothing (verified: it silently returns CONNECTION_BAD). Instrumenting libpq
  # makes PQstatus's read of the freed PGconn a labelled heap-use-after-free with
  # PQfinish/makeEmptyPGconn frames. The server keeps the ordinary libpq in
  # $PG_PREFIX/lib (initdb links it), so only the client library is ASan-built —
  # exactly the split lua-openssl-141 uses (ASan libcrypto, ordinary everything).
  # src/port + src/common static archives (already built above, un-instrumented)
  # are linked as-is; the PGconn alloc/free/use all live in libpq's fe-connect.c.
  ( cd "$PG_SRC/src/interfaces/libpq" && make clean >/dev/null 2>&1
    make -j"$(nproc)" \
      CFLAGS="-g -O1 -fno-omit-frame-pointer -std=gnu17 -fsanitize=address" \
      >"$PG_SRC/libpq-asan.log" 2>&1 )
  mkdir -p "$PG_PREFIX/lib-asan"
  cp -a "$PG_SRC/src/interfaces/libpq/"libpq.so* "$PG_PREFIX/lib-asan/"
  have_pg || { echo "BLOCKED: PostgreSQL build incomplete (see $PG_SRC/*.log)" >&2; exit 1; }
  echo "built PostgreSQL $PG_VER at $PG_PREFIX (server: lib/, ASan client: lib-asan/)"
fi

# ---- 2. LuaDBI checkout ---------------------------------------------------
if [ ! -d "$W/luadbi/.git" ]; then
  echo "== cloning LuaDBI =="
  git clone --quiet https://github.com/mwild1/luadbi "$W/luadbi"
fi
echo "LuaDBI at: $W/luadbi"

echo "OK: prerequisites ready. Now run ./run.sh"
