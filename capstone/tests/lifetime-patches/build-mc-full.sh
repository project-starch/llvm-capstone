#!/usr/bin/env bash
# build-mc-full.sh: the protected memcached server from the patched tree (0001-0006 and the reuse
# hooks), configured as experiments/study/reuse-hooks/memcached/build-reuse.sh does, with the
# prebuilt libevent and SDK. Prints the CDERIVE/CREVOKE/CRENEW counts of the image.
set -euo pipefail
L=$HOME/asr-work/life-c PB=$HOME/asr-work/eval/prebuilt/reuse/memcached E=$HOME/asr-work/eval/tree
export CAPSTONE_CLANG=$HOME/lanes/gatefix/llvm/bin/clang CAPSTONE_LD_LLD=$HOME/lanes/gatefix/llvm/bin/ld.lld
export MC_MUSL=$(ls -d $PB/musl-src/musl-*/ | head -1) MC_DEPS_PREFIX=$PB/deps-cap
export MC_RUNTIME_DIR=$(ls -d $PB/deps-runtime-*) MC_LIBC_ARCHIVE=$(ls $PB/musl-capstone-build-*/libc-capstone.a)
export CAPSTONE_SDK=$MC_RUNTIME_DIR MC_LINKER_SCRIPT=$E/capstone/my_first_domain/link.ld
export CC=$E/capstone/ports/memcached/app/deps/capstone-cc
X=$L/mc-full; rm -rf "$X"; cp -a "$L/mcs/memcached-1.6.45" "$X"; cp "$E/capstone/ports/memcached/app/src/mcapp-safety.c" "$X/"
cd "$X"
./configure --host=riscv64-unknown-linux-musl --with-libevent="$MC_DEPS_PREFIX" --disable-extstore --disable-proxy \
  --disable-tls --disable-sasl --disable-docs CPPFLAGS="-I$E/capstone/runtime/include -DMC_CAPSTONE_SUBLET" > configure.log 2>&1
make -j8 memcached > build.log 2>&1 || { grep -m5 error build.log; exit 1; }
"$HOME/lanes/gatefix/llvm/bin/llvm-objcopy" -O binary --only-section=.text memcached /tmp/life-c-mc.text
python3 - /tmp/life-c-mc.text <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read(); w = struct.unpack('<%dI' % (len(d) // 4), d[:len(d) // 4 * 4])
c = lambda f7: sum(1 for x in w if x & 0xfe00707f == (f7 << 25) | (1 << 12) | 0x5b)
print('memcached protected built: cderive=%d crevoke=%d crenew=%d' % (c(0x51), c(0x52), c(0x54)))
PY
rm -f /tmp/life-c-mc.text
