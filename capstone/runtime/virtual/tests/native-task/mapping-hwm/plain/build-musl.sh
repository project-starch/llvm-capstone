#!/bin/bash
# build-musl.sh (perf/plainhwm) <variant> <extra CFLAGS>: perf/alean/plain/build-musl.sh (llvm-c4, -O1, rv64gc, lp64d,
# static) on perf/plainhwm/musl-src, out of tree in b-<variant>, into musl-<variant>.
main() {
  set -eu
  local F=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/perf/plainhwm L=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}/llvm-c4/bin v=$1 x=${2:-}
  mkdir -p $F/b-$v && cd $F/b-$v
  $F/musl-src/configure --target=riscv64-linux-musl --prefix=$F/musl-$v --disable-shared \
    CC="$L/clang --no-default-config --target=riscv64-unknown-linux-musl -march=rv64gc -mabi=lp64d" CFLAGS="-O1 $x" \
    AR=$L/llvm-ar RANLIB="$L/llvm-ar s" > $F/b-$v.configure.log 2>&1
  taskset -c 16-31,48-63 nice -n 10 make -j12 > $F/b-$v.build.log 2>&1
  make install > $F/b-$v.install.log 2>&1
  sha256sum $F/musl-$v/lib/libc.a
}
main "$@"; exit $?
