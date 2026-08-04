#!/usr/bin/env bash
# Build the sol2 #1080 harness with ASan against the shared Lua toolchain.
# sol2 is header-only; v3.5.0 builds on gcc 15 and the reference/value push
# semantics behind #1080 are version-stable.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"

LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = "SYSTEM" ]; then LUA_INC=$(pkg-config --cflags lua5.4); LUA_LIB=$(pkg-config --libs lua5.4); LUADIR="";
else LUA_INC="-I$LUA"; LUA_LIB="-L$LUA -llua5.4"; LUADIR="$LUA"; fi

[ -d "$W/sol2" ] || git clone --depth 1 --branch v3.5.0 https://github.com/ThePhD/sol2 "$W/sol2"
c++ -std=c++17 -g -O0 -fsanitize=address -fno-omit-frame-pointer \
  -I"$W/sol2/include" $LUA_INC "$HERE/harness.cpp" -o "$W/harness" $LUA_LIB
echo "built: $W/harness  (lua: ${LUADIR:-system})"
