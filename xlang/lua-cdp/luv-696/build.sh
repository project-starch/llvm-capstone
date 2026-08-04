#!/usr/bin/env bash
# Build luv #696 at vulnerable (0e4a895) and fixed (3e39f98) with ASan + shared
# libuv, module build, against the shared Lua toolchain.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
pkg-config --exists libuv || sudo apt-get install -y libuv1-dev
LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = SYSTEM ]; then LI=$(pkg-config --variable=includedir lua5.4); LL=$(pkg-config --variable=libdir lua5.4)/liblua5.4.so;
else LI="$LUA"; LL="$LUA/liblua5.4.so"; fi
[ -d "$W/luv" ] || git clone https://github.com/luvit/luv "$W/luv"
b(){ git -C "$W/luv" checkout -q "$1"; rm -rf "$W/$2"
  cmake -S "$W/luv" -B "$W/$2" -DWITH_SHARED_LIBUV=ON -DBUILD_MODULE=ON \
    -DLUA_BUILD_TYPE=System -DWITH_LUA_ENGINE=Lua -DLUA_INCLUDE_DIR="$LI" -DLUA_LIBRARIES="$LL" \
    -DCMAKE_C_FLAGS="-g -O0 -fsanitize=address -fno-omit-frame-pointer" \
    -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address" >/dev/null 2>&1
  cmake --build "$W/$2" -j4 >/dev/null 2>&1; }
b 0e4a895 vuln; b 3e39f98 fixed; git -C "$W/luv" checkout -q 0e4a895
echo "built: $W/{vuln,fixed}/luv.so"
