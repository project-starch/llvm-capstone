#!/usr/bin/env bash
# Build luv #503 at vulnerable (e2d3d18) and fixed (ba4589c), module build with
# shared libuv, against the shared Lua toolchain. Same cmake-module recipe as
# luv-696, but WITHOUT -fsanitize=address: this UAF is a read of a freed
# lua_State that happens *inside* uninstrumented liblua5.4.so (lua_settop/
# luaL_unref), which ASan-on-luv.so cannot see (and ASan's quarantine keeps the
# freed bytes intact, so the stale read silently "succeeds"). Issue #503 was
# diagnosed under valgrind; run.sh uses valgrind, which instruments liblua too.
# ponytail: plain (non-ASan) build on purpose — ASan is both blind here and
# incompatible with the valgrind run. Upgrade path: an ASan-built liblua5.4
# would let ASan catch it, but the shared toolchain lua is prebuilt read-only.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
pkg-config --exists libuv || { echo "libuv missing (expected system libuv 1.51)" >&2; exit 2; }
command -v valgrind >/dev/null || { echo "valgrind required for this case" >&2; exit 2; }
LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = SYSTEM ]; then LI=$(pkg-config --variable=includedir lua5.4); LL=$(pkg-config --variable=libdir lua5.4)/liblua5.4.so;
else LI="$LUA"; LL="$LUA/liblua5.4.so"; fi
[ -d "$W/luv" ] || git clone https://github.com/luvit/luv "$W/luv"
b(){ git -C "$W/luv" checkout -q "$1"; rm -rf "$W/$2"
  cmake -S "$W/luv" -B "$W/$2" -DWITH_SHARED_LIBUV=ON -DBUILD_MODULE=ON \
    -DLUA_BUILD_TYPE=System -DWITH_LUA_ENGINE=Lua -DLUA_INCLUDE_DIR="$LI" -DLUA_LIBRARIES="$LL" \
    -DCMAKE_C_FLAGS="-g -O0 -fno-omit-frame-pointer" >/dev/null 2>&1
  cmake --build "$W/$2" -j4 >/dev/null 2>&1; }
b e2d3d18 vuln; b ba4589c fixed; git -C "$W/luv" checkout -q e2d3d18
echo "built: $W/{vuln,fixed}/luv.so"
