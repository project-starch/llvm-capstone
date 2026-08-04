#!/usr/bin/env bash
# Build the deliberately-unsafe minilmdb.so binding with ASan against the shared
# Lua toolchain and the system liblmdb 0.9.31. lua-shared itself is not ASan-built,
# so run.sh LD_PRELOADs libasan (interceptors are process-global, so liblmdb's
# malloc/free are tracked even though liblmdb is uninstrumented).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd)
W="$HERE/.work"; mkdir -p "$W"
LUA=$("$LC/_toolchain/build-lua.sh")
if [ "$LUA" = "SYSTEM" ]; then LI=$(pkg-config --cflags lua5.4); LL=$(pkg-config --libs lua5.4);
else LI="-I$LUA"; LL="-L$LUA -llua5.4"; fi
[ -e /usr/include/lmdb.h ] || { echo "liblmdb-dev missing (do NOT apt-install here)" >&2; exit 2; }
cc -shared -fPIC -g -O0 -fsanitize=address -fno-omit-frame-pointer \
  $LI "$HERE/minilmdb.c" -llmdb $LL -o "$W/minilmdb.so"
echo "built: $W/minilmdb.so"
