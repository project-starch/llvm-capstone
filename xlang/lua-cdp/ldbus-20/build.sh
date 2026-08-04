#!/usr/bin/env bash
# Build ldbus #20 at the vulnerable (2571a9b) and fixed (5cc933b) commits with
# ASan against the shared Lua toolchain. Needs libdbus-1-dev.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"; mkdir -p "$W"
pkg-config --exists dbus-1 || sudo apt-get install -y libdbus-1-dev
LUA=$("$LC/_toolchain/build-lua.sh"); [ "$LUA" = SYSTEM ] && { LI=$(pkg-config --cflags lua5.4); LL=$(pkg-config --libs lua5.4); } || { LI="-I$LUA"; LL="-L$LUA -llua5.4"; }
[ -d "$W/ldbus" ] || git clone https://github.com/daurnimator/ldbus "$W/ldbus"
b(){ git -C "$W/ldbus" checkout -q "$1"; mkdir -p "$W/$2"; cc -shared -fPIC -g -O0 -fsanitize=address -fno-omit-frame-pointer \
  $LI $(pkg-config --cflags dbus-1) -I"$W/ldbus/vendor/compat-5.3/c-api" \
  "$W"/ldbus/src/*.c "$W"/ldbus/vendor/compat-5.3/c-api/compat-5.3.c $(pkg-config --libs dbus-1) -o "$W/$2/ldbus.so"; }
b 2571a9b vuln; b 5cc933b fixed; git -C "$W/ldbus" checkout -q 2571a9b
echo "built: $W/{vuln,fixed}/ldbus.so"
