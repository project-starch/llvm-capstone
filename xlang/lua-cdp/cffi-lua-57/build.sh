#!/usr/bin/env bash
# Build the toolchain for cffi-lua #57: a shared PUC Lua 5.4 (if the system has
# no lua5.4-dev) plus a pkg-config file the cffi-lua meson build can find.
# Idempotent; leaves everything under ./.work. Reproduced on 2026-08-03.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"; mkdir -p "$W"

# If the system already provides lua5.4 + libffi via pkg-config, use them.
if PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}" pkg-config --exists lua5.4 libffi; then
  echo "system lua5.4 + libffi present; no from-source lua needed"; exit 0
fi

echo "== building shared PUC Lua 5.4 from source =="
[ -d "$W/lua54" ] || git clone --depth 1 --branch v5.4.7 https://github.com/lua/lua "$W/lua54"
cd "$W/lua54"
rm -f ./*.o liblua5.4.so lua-shared
for f in ./*.c; do
  case "$(basename "$f")" in lua.c|luac.c|onelua.c|ltests.c) continue;; esac
  cc -DLUA_USE_LINUX -DLUA_USE_DLOPEN -fPIC -O2 -w -c "$f"
done
cc -shared -o liblua5.4.so ./*.o -lm -ldl
cc -DLUA_USE_LINUX -DLUA_USE_DLOPEN -O2 -w -c lua.c -o lua_main.o
cc -rdynamic -o lua-shared lua_main.o -L. -llua5.4 -lm -ldl
# PUC source repo ships no lua.hpp; cffi-lua's meson probe needs it.
printf 'extern "C" {\n#include "lua.h"\n#include "lualib.h"\n#include "lauxlib.h"\n}\n' > lua.hpp

mkdir -p "$W/pc"
cat > "$W/pc/lua5.4.pc" <<EOF
prefix=$W/lua54
Name: Lua
Description: Lua
Version: 5.4.7
Cflags: -I\${prefix}
Libs: -L\${prefix} -llua5.4
EOF
echo "built: $W/lua54/{liblua5.4.so,lua-shared}, pc at $W/pc"
