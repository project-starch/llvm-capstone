#!/usr/bin/env bash
# B2 (docs/plans/b0-silicon-delegated-runtime.md): memcached 1.6.45 as a SILICON-ABI delegated application.
#
# build-b0-hello.sh's many-source mode over the port's configured trees: the 25 memcached sources and the 18
# libevent sources of its libevent_core, LTO-linked with musl, the runtime and B1's contexts. Three contexts, a
# 20 MiB data region with a 16 MiB arena (memcached -t 1 -m 8). The board runs of B2a and B3 used this build
# (ba7e6921cf27f2b6).
#
# Required: what build-b0-hello.sh requires (CAPSTONE_LLVM_BIN, B0_MUSL_ARCHIVE), and the port's configured trees
#           under MC_WORK (default $CAPSTONE_TMP_ROOT/memcached-app, ports/memcached/app/deps/env.sh): the patched,
#           configured memcached in domain/src, libevent-cap in deps-build, its installed headers in deps-cap. The
#           port's own SDK build creates them; this script only reads them.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MC_WORK=${MC_WORK:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/memcached-app}
MC=$MC_WORK/domain/src
LE=$MC_WORK/deps-build/libevent-cap
[ -f "$MC/config.h" ] && [ -f "$LE/config.h" ] || { echo "no configured memcached/libevent under $MC_WORK" >&2; exit 2; }

SRCS=""
for f in assoc authfile base64 bipbuffer cache crawler daemon hash items itoa_ljust jenkins_hash logger \
         vendor/mcmc/mcmc memcached murmur3_hash proto_bin proto_parser proto_text restart slab_automove \
         slabs_mover slabs stats_prefix thread util; do
  SRCS="$SRCS $MC/$f.c"
done
for f in buffer bufferevent bufferevent_filter bufferevent_pair bufferevent_ratelim bufferevent_sock event evmap \
         evthread evutil evutil_rand evutil_time listener log select poll epoll signal; do
  SRCS="$SRCS $LE/$f.c"
done
for f in $SRCS; do [ -f "$f" ] || { echo "missing source $f" >&2; exit 2; }; done

export B0_APP=memcached B0_APP_SRCS="$SRCS"
export B0_APP_CFLAGS="-DHAVE_CONFIG_H -DNDEBUG -fno-strict-aliasing -I$LE/compat -I$LE/include \
-I$MC_WORK/deps-cap/include -I$MC -I$MC/vendor -Wno-error"
export B0_CONTEXT_BYTES=${B0_CONTEXT_BYTES:-131072} B0_CONTEXTS=${B0_CONTEXTS:-3}
export B0_DATA=${B0_DATA:-$((20 << 20))} B0_ARENA=${B0_ARENA:-$((16 << 20))}
export OUT_DIR=${OUT_DIR:-${CAPSTONE_TMP_ROOT:-/tmp/capstone}/b2/memcached}
exec bash "$HERE/build-b0-hello.sh"
