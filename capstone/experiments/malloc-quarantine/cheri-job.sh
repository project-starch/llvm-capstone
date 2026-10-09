#!/usr/bin/env bash
# One measurement in its own CheriBSD VM: boot from the snapshot image, copy the guest kit in,
# run one program under one arm and one instrument, power off. Several jobs run side by side,
# each with its own slot (port, work dir, serial socket).
#   cheri-job.sh <slot> <results dir> <arm> <instrument> <program args...>
# arm         on (CheriBSD default: revocation, async, 1/4) or off (revocation disabled)
# instrument  plain     mqstat.so only: exit-time epochs, jemalloc ledger, sweep counters,
#                       time -l (max RSS); the footprint measurement
#             trace<N>  mqtrace.so with MQ_TRACK=<N> (and MQ_WIN=0) plus mqstat.so;
#                       MQ_ADDR_SAMPLE and MQ_SAMPLE pass through from the environment
#             maps      run-maps.sh: resident pages per mapping owner every MAPS_EVERY s
# <stdin>     MQ_STDIN names a file in the guest kit fed to the program (barnes.input)
# Result: <results dir>/<arm>-<instrument>-<tag>.txt, ending in MQ-EXIT rc=<status>.
set -u
SLOT=$1 RES=$2 ARM=$3 INST=$4; shift 4
KIT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CHERI=${CHERI:-$HOME/cheri-mq/output} MQ_WORK=$HOME/cheri-mq/work-c$SLOT MQ_PORT=$((10480 + SLOT))
export PY=${PY:-/usr/bin/python3}
mkdir -p "$MQ_WORK" "$RES"
tag=$(printf '%s_' "$@" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
f="$RES/$ARM-$INST-$tag.txt"
case $ARM in
  off) envs="_RUNTIME_REVOCATION_DISABLE=1" ;;
  on)  envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
in=/dev/null; [ -n "${MQ_STDIN:-}" ] && in=/root/mq/$MQ_STDIN
case $INST in
  plain)   cmd="env -i PATH=/bin:/usr/bin HOME=/root $envs LD_PRELOAD=/root/mq/mqstat.so /usr/bin/time -l $* < $in" ;;
  trace[123]) cmd="env -i PATH=/bin:/usr/bin HOME=/root $envs MQ_TRACK=${INST#trace} MQ_WIN=0 ${MQ_ADDR_SAMPLE:+MQ_ADDR_SAMPLE=$MQ_ADDR_SAMPLE} ${MQ_SAMPLE:+MQ_SAMPLE=$MQ_SAMPLE} LD_PRELOAD=/root/mq/mqtrace.so:/root/mq/mqstat.so /usr/bin/time -l $* < $in" ;;
  maps)    [ "$ARM" = on ] || { echo "maps runs the on arm only" >&2; exit 2; }
           cmd="MAPS_EVERY=${MAPS_EVERY:-20} MAPS_STDIN=$in sh /root/mq/run-maps.sh /root/mq/maps.out $*; cat /root/mq/maps.out /root/mq/maps.out.prog" ;;
  *) echo "unknown instrument $INST" >&2; exit 2 ;;
esac
cd "$KIT"
{
  echo "MQ-JOB slot=$SLOT arm=$ARM instrument=$INST argv=$* start=$(date -u +%FT%TZ)"
  echo "MQ-KIT $(cat guest/SHA256SUMS | tr '\n' ' ')"
  if ./vm.sh up > "$MQ_WORK/up.log" 2>&1 && ./vm.sh put guest/* > "$MQ_WORK/put.log" 2>&1; then
    ./vm.sh ssh "cd /root/mq && chmod +x * && $cmd" 2>&1
    echo "MQ-EXIT rc=$?"
  else
    echo "MQ-EXIT rc=boot-failed"
  fi
  echo "MQ-JOB end=$(date -u +%FT%TZ)"
  ./vm.sh down > /dev/null 2>&1
} > "$f"
grep -q '^MQ-EXIT rc=0$' "$f" && echo "ok   $f" || echo "FAIL $f"
