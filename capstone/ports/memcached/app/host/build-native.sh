#!/usr/bin/env bash
# The oracle's reference: memcached 1.6.45 on the host, unpatched, against libevent 2.1.12 built from
# the same pinned tarball with the same options as the domain's. Output: $MC_WORK/native/bin/memcached.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
source "$MC_DEPS_DIR/fetch.sh"
NAT=$MC_WORK/native; rm -rf "$NAT"; mkdir -p "$NAT/src" "$NAT/prefix"
LTAR=$(mc_fetch libevent)
tar -xf "$LTAR" -C "$NAT/src"
( cd "$NAT/src"/libevent-* && env -u CC -u AR -u RANLIB ./configure --prefix="$NAT/prefix" --disable-shared --enable-static \
    --disable-openssl --disable-thread-support --disable-samples --disable-debug-mode > "$NAT/libevent-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j"${JOBS:-8}" install > "$NAT/libevent-build.log" 2>&1 )
MSRC=$(CAPSTONE_TMP_ROOT=$MC_WORK bash "$APP/../fetch-memcached.sh" | tail -1)
rm -rf "$NAT/memcached"; cp -a "$MSRC" "$NAT/memcached"
( cd "$NAT/memcached" && env -u CC -u AR -u RANLIB ./configure --prefix="$NAT" --with-libevent="$NAT/prefix" \
    --disable-extstore --disable-proxy --disable-tls --disable-sasl --disable-docs > "$NAT/memcached-configure.log" 2>&1 &&
  env -u CC -u AR -u RANLIB make -j"${JOBS:-8}" memcached > "$NAT/memcached-build.log" 2>&1 )
mkdir -p "$NAT/bin"; cp "$NAT/memcached/memcached" "$NAT/bin/memcached"
echo "native memcached: $("$NAT/bin/memcached" -V) sha256 $(sha256sum < "$NAT/bin/memcached" | cut -c1-16)"
