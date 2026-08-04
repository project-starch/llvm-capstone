#!/usr/bin/env bash
# Build LuaBridge #319 harness with ASan against the shared Lua toolchain.
# LuaBridge is header-only (vinniefalco/LuaBridge, Source/).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = "SYSTEM" ]; then LUA_INC=$(pkg-config --cflags lua5.4); LUA_LIB=$(pkg-config --libs lua5.4);
else LUA_INC="-I$LUA"; LUA_LIB="-L$LUA -llua5.4"; fi
[ -d "$W/lb" ] || git clone --depth 1 https://github.com/vinniefalco/LuaBridge "$W/lb"
c++ -std=c++17 -g -O0 -fsanitize=address -fno-omit-frame-pointer \
  -I"$W/lb/Source" $LUA_INC "$HERE/harness.cpp" -o "$W/harness" $LUA_LIB
echo "built: $W/harness"
