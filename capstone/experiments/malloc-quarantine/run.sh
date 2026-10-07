#!/usr/bin/env bash
# Run one program under several MRS policy arms in the guest.
#   run.sh OUTDIR ARMS -- PROGRAM ARGS...
# ARMS is a comma list of: off on sync q8 q2 q1 (on = default async, 1/4).
# Every arm starts from an empty environment, so only the named knobs differ.
# MQ_PRELOAD=1 adds mqstat.so (exit-time epochs and jemalloc ledger) for
# unmodified programs; MQ_TRACK=1 turns on churn's address tracking.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
out=$1; arms=$2; shift 2; [ "$1" = "--" ] && shift
mkdir -p "$out"
tag=$(printf '%s_' "$@" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_*$//')
for arm in ${arms//,/ }; do
  case $arm in
    off)  envs="_RUNTIME_REVOCATION_DISABLE=1" ;;
    on)   envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1" ;;
    sync) envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_SYNC_REVOKE=1" ;;
    q8)   envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1 _RUNTIME_QUARANTINE_NUMERATOR=1 _RUNTIME_QUARANTINE_DENOMINATOR=8" ;;
    q2)   envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1 _RUNTIME_QUARANTINE_NUMERATOR=1 _RUNTIME_QUARANTINE_DENOMINATOR=2" ;;
    q1)   envs="_RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1 _RUNTIME_QUARANTINE_NUMERATOR=1 _RUNTIME_QUARANTINE_DENOMINATOR=1" ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  f="$out/$arm-$tag.txt"
  {
    echo "MQ-ARM arm=$arm env=$envs track=${MQ_TRACK:-0} argv=$*"
    pre=; [ "${MQ_PRELOAD:-0}" = 1 ] && pre=LD_PRELOAD=/root/mq/mqstat.so
    "$HERE/vm.sh" ssh "cd /root/mq && env -i PATH=/bin:/usr/bin HOME=/root MQ_TRACK=${MQ_TRACK:-0} $pre $envs /usr/bin/time -l $*" 2>&1
    echo "MQ-EXIT rc=$?"
  } > "$f" || true
  grep -q '^MQ-EXIT rc=0$' "$f" && echo "ok   $f" || echo "FAIL $f"
done
