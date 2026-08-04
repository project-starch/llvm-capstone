#!/usr/bin/env bash
# Build agladysh/lua-hiredis @ the surveyed HEAD with ASan against the shared Lua
# toolchain + system hiredis. This is a BLOCKED case: the point of building is to
# let run.sh EMPIRICALLY confirm the binding is CDP-safe by construction (the
# reply is a Lua copy; the context free is null-guarded), not to reproduce a bug.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
COMMIT=5df62990fdec196dc88a1037aa809afd1f57f1e4   # HEAD; builds on Lua 5.4
pkg-config --exists hiredis || { echo "need libhiredis-dev (installed on this box)" >&2; exit 2; }
LUA=$("$LC/_toolchain/build-lua.sh"); [ "$LUA" = SYSTEM ] && { LI=$(pkg-config --cflags lua5.4); } || { LI="-I$LUA"; }
[ -d "$W/lua-hiredis/.git" ] || git clone -q https://github.com/agladysh/lua-hiredis "$W/lua-hiredis"
git -C "$W/lua-hiredis" checkout -q "$COMMIT"
cc -shared -fPIC -g -O0 -fsanitize=address -fno-omit-frame-pointer \
  $LI $(pkg-config --cflags hiredis) \
  "$W"/lua-hiredis/src/lua-hiredis.c $(pkg-config --libs hiredis) \
  -o "$W/hiredis.so"
echo "built: $W/hiredis.so (agladysh/lua-hiredis @ ${COMMIT:0:9}, ASan)"
