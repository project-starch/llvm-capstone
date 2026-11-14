#!/usr/bin/env bash
# build-mc.sh: slabs.c and cache.c of the fixed and of the original patch 0006, rebuilt with inject.h,
# linked with mc-test.c and the prebuilt protected server's other objects (its main renamed away).
set -euo pipefail
L=$HOME/asr-work/life-c T=$L/test O=$L/mc-test
PB=$HOME/asr-work/eval/prebuilt/reuse/memcached
S=$PB/protected/src E=$HOME/asr-work/eval/tree
mkdir -p "$O"
export CAPSTONE_CLANG=$HOME/lanes/gatefix/llvm/bin/clang CAPSTONE_LD_LLD=$HOME/lanes/gatefix/llvm/bin/ld.lld
export MC_MUSL=$(ls -d $PB/musl-src/musl-*/ | head -1) MC_DEPS_PREFIX=$PB/deps-cap
export MC_RUNTIME_DIR=$(ls -d $PB/deps-runtime-*) MC_LIBC_ARCHIVE=$(ls $PB/musl-capstone-build-*/libc-capstone.a)
export CAPSTONE_SDK=$MC_RUNTIME_DIR MC_LINKER_SCRIPT=$E/capstone/my_first_domain/link.ld
CC=$E/capstone/ports/memcached/app/deps/capstone-cc
FLAGS="-DHAVE_CONFIG_H -I$S -DNDEBUG -I$PB/deps-cap/include -I$E/capstone/runtime/include -DMC_CAPSTONE_SUBLET -g -O2 -w"
"$HOME/lanes/gatefix/llvm/bin/llvm-objcopy" --redefine-sym main=mc_upstream_main "$S/memcached-memcached.o" "$O/memcached-memcached.o"
$CC $FLAGS -c -o "$O/mc-test.o" "$T/mc-test.c"
objs=$(cd "$S" && ls memcached-*.o | grep -vE 'memcached-(memcached|slabs|cache)\.o' | sed "s|^|$S/|")
for v in fixed orig; do
  src=$L/mc-src-$v
  $CC $FLAGS -include "$T/inject.h" -c -o "$O/slabs-$v.o" "$src/slabs.c"
  $CC $FLAGS -include "$T/inject.h" -c -o "$O/cache-$v.o" "$src/cache.c"
  $CC -g -O2 -L$PB/deps-cap/lib -o "$O/mc-test-$v.dom" "$O/mc-test.o" "$O/memcached-memcached.o" \
     "$O/slabs-$v.o" "$O/cache-$v.o" $objs -levent
  echo "built $O/mc-test-$v.dom"
done
