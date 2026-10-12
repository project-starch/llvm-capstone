#!/usr/bin/env bash
# memcached 1.6.45, both arms from one build: domain/memcached.dom (stock) and sublet/memcached.dom (patch 0006).
set -u
. $HOME/sa-mem-mm/env.sh
s=$(date +%s)
JOBS=16 bash $R/ports/memcached/app/host/build-virtual.sh $S/mc-build > $S/mc-build.log 2>&1
echo "MC-BUILD rc=$? $(( $(date +%s) - s ))s" > $S/mc-build.status
