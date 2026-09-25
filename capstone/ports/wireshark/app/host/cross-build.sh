#!/usr/bin/env bash
# The minimal tshark cross-built for capstone64 with Wireshark's own CMake, against the libraries
# deps/ built (M-deps).
#
#   cross-build.sh          -> $TS_WORK/xbuild: static libwsutil, libwiretap and libwireshark,
#                              plus tshark linked by capstone-cc as a first link check
#
# Source: the pinned tarball (upstream.json), plus every patch in patches/ except 0007, plus
# src/capstone-stubs.c copied into epan/dissectors/ (the whitelist lists it as a dissector file).
# 0007 (wmem's block size) belongs to the sublet heap arm alone: host/build-domain.sh applies it
# to its own copies of the two wmem files. Applied here as well, it would land in every arm's
# libwsutil (moving wmem's asserts' __LINE__ even where its define is off) and then fail to apply
# a second time in the sublet arm.
#
# What makes it a cross build, each from reading the tree (the M0 plan's Step C3):
# - BUILD_SHARED_LIBS=OFF: capstone-cc links domains, never shared objects;
# - LEMON_C_COMPILER=clang: lemon is the only program the build runs, so it is built for the host;
# - BUILD_dcerpcidl2wrs=OFF: it defaults on and would be built for capstone64;
# - the find hints for what the deps prefix lacks or names differently:
#   - GThread is libglib;
#   - libgcrypt's error library;
#   - iconv and libm are in musl;
# - the options the native minimal build had, and the optional libraries it did not find (xxhash,
#   minizip-ng, libnl) turned off;
# - CAPSTONE_SINGLE_THREAD_DOMAIN (patch 0004), CAPSTONE_HF_PREALLOC=4096 (patch 0005), NDEBUG;
# - -fno-stack-protector. Wireshark adds -fstack-protector-strong whenever the compiler accepts it,
#   and every stack-protector mode crashes capstone64's clang in "Insert stack protectors"
#   (ISSUES.md C-60, open: the guard slot is built in address space 0). The first cross build
#   failed 82 files on it.
#
# TIME: about two hours with the tree's debug clang, most of it four generated tables compiled
# in parallel at the end (manuf.c alone about 90 minutes: its pointer initialisers, plan item 5).
# Do not wrap it in a short timeout; one at 50 minutes killed a build at step 409 of 424.
# `ninja -C $TS_WORK/xbuild -k 0 tshark` resumes one.
# EVERY later ninja invocation in the tree costs the same again: CMake's glob check is always
# dirty, so ninja re-runs CMake, and prefs.c, proto.c and manuf.c then recompile (seen on three
# invocations; what the regeneration touches was not pinned down). host/build-domain.sh therefore
# compiles tshark.c itself and never builds through ninja.
set -euo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$APP/deps/env.sh"
SRC=$TS_WORK/xsrc B=$TS_WORK/xbuild LOG=$TS_WORK/xbuild-logs P=$TS_DEPS_PREFIX
read -r URL SHA VER < <(python3 -c 'import json,sys; u=json.load(open(sys.argv[1])); print(u["url"], u["sha256"], u["version"])' "$APP/upstream.json")
TAR=$TS_WORK/wireshark-$VER.tar.xz
[ -f "$TAR" ] || curl -sSfL --retry 5 -o "$TAR" "$URL"
echo "$SHA  $TAR" | sha256sum -c --quiet -
rm -rf "$SRC" "$B" "$LOG"; mkdir -p "$SRC" "$B" "$LOG"
tar -xf "$TAR" -C "$SRC" --strip-components=1
for p in "$APP"/patches/0*.patch; do
  case $p in */0007-*) continue ;; esac
  patch -s -d "$SRC" -p1 < "$p"
done
cp "$APP/src/capstone-stubs.c" "$SRC/epan/dissectors/"

cat > "$B/capstone64.cmake" <<TCEOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_C_COMPILER $CC)
set(CMAKE_AR $AR CACHE FILEPATH "")
set(CMAKE_RANLIB $RANLIB CACHE FILEPATH "")
set(CMAKE_FIND_ROOT_PATH $P)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
TCEOF
OFF_TOOLS=(wireshark stratoshark logray dumpcap capinfos captype editcap mergecap reordercap
  text2pcap randpkt dftest rawshark sharkd tfshark fuzzshark androiddump sshdump ciscodump udpdump
  randpktdump etwdump dpauxmon sdjournal falcodump sshdig strato mmdbresolve dcerpcidl2wrs corbaidl2wrs
  wifidump xxx2deb)
ARGS=(-G Ninja -S "$SRC" -B "$B" -DCMAKE_TOOLCHAIN_FILE="$B/capstone64.cmake"
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -DNDEBUG"
  -DCMAKE_C_FLAGS="-DCAPSTONE_SINGLE_THREAD_DOMAIN -DCAPSTONE_HF_PREALLOC=4096 -fno-stack-protector"
  -DBUILD_SHARED_LIBS=OFF -DBUILD_tshark=ON
  -DENABLE_PLUGINS=OFF -DENABLE_PCAP=OFF -DENABLE_LUA=OFF -DENABLE_CAP=OFF -DENABLE_GNUTLS=OFF
  -DENABLE_NETLINK=OFF -DENABLE_MINIZIPNG=OFF -DENABLE_XXHASH=OFF -DENABLE_COMPILER_COLOR_DIAGNOSTICS=OFF
  -DCAPSTONE_DISSECTOR_WHITELIST="$APP/dissector-whitelist.txt"
  -DLEMON_C_COMPILER=/usr/bin/clang
  -DGTHREAD2_LIBRARY="$P/lib/libglib-2.0.a" -DGTHREAD2_INCLUDE_DIR="$P/include/glib-2.0"
  -DGCRYPT_INCLUDE_DIR="$P/include" -DGCRYPT_LIBRARY="$P/lib/libgcrypt.a" -DGCRYPT_ERROR_LIBRARY="$P/lib/libgpg-error.a"
  -DICONV_INCLUDE_DIR="$TS_MUSL/include" -DM_INCLUDE_DIR="$TS_MUSL/include")
for t in "${OFF_TOOLS[@]}"; do ARGS+=(-DBUILD_$t=OFF); done
PKG_CONFIG_LIBDIR="$P/lib/pkgconfig" PKG_CONFIG_SYSROOT_DIR= cmake "${ARGS[@]}" > "$LOG/configure.log" 2>&1 \
  || { tail -30 "$LOG/configure.log"; exit 1; }
echo "cross-build: configured"
grep -q 'COMMAND = /usr/bin/clang\|/usr/bin/clang ' "$B/build.ninja" && echo "cross-build: lemon is built by host clang"
( TS_CENSUS=1 TS_CENSUS_LOG="$LOG/cast-log.txt" ninja -C "$B" -k 0 tshark > "$LOG/build.log" 2>&1 ) || true
grep -E "^FAILED: " "$LOG/build.log" | sed 's/^FAILED: //' > "$LOG/failed.txt" || true
echo "cross-build: $(wc -l < "$LOG/failed.txt") failed build steps; tshark $( [ -f "$B/run/tshark" ] && echo LINKED || echo 'NOT linked')"
