#!/usr/bin/env bash
# Build a shared PUC Lua 5.4 + pkg-config into _toolchain/.work, once, for all
# cases (unless the system already provides lua5.4-dev). Prints LUA_PC_DIR.
set -euo pipefail
T=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); W="$T/.work"
if pkg-config --exists lua5.4 2>/dev/null; then echo "SYSTEM"; exit 0; fi
mkdir -p "$W"
if [ ! -x "$W/lua54/lua-shared" ]; then
  [ -d "$W/lua54" ] || git clone --depth 1 --branch v5.4.7 https://github.com/lua/lua "$W/lua54"
  cd "$W/lua54"; rm -f ./*.o liblua5.4.so lua-shared
  for f in ./*.c; do case "$(basename "$f")" in lua.c|luac.c|onelua.c|ltests.c) continue;; esac
    cc -DLUA_USE_LINUX -DLUA_USE_DLOPEN -fPIC -O2 -w -c "$f"; done
  cc -shared -o liblua5.4.so ./*.o -lm -ldl
  cc -DLUA_USE_LINUX -DLUA_USE_DLOPEN -O2 -w -c lua.c -o lua_main.o
  cc -rdynamic -o lua-shared lua_main.o -L. -llua5.4 -lm -ldl
  printf 'extern "C" {\n#include "lua.h"\n#include "lualib.h"\n#include "lauxlib.h"\n}\n' > lua.hpp
  mkdir -p "$W/pc"; cat > "$W/pc/lua5.4.pc" <<PC
prefix=$W/lua54
Name: Lua
Version: 5.4.7
Cflags: -I\${prefix}
Libs: -L\${prefix} -llua5.4
PC
fi
echo "$W/lua54"
