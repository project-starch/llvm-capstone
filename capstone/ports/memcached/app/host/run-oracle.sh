#!/usr/bin/env bash
# The memcached oracle. Native reference runs on the host (two null runs, two perturbed), then one
# domain run in one guest boot, and every comparison the plan names:
#   null      two native runs: transcripts identical (the script is deterministic);
#   positive  a perturbed value byte, and a perturbed cas value: each transcript must differ;
#   identity  `stats pointer_size` reads 64 natively and 128 in the domain;
#   oracle    the domain's normalised transcript is byte-identical to the native one, its exit status
#             (capstone-job's record) matches the native one, and its stderr is empty like native's.
#   run-oracle.sh <results-dir> [--stop TERM|USR1]
# Env: MC_WORK (build-native.sh, build-domain.sh outputs), CAPSTONE_VM_UP_ARGS (capstone-vm up platform
# arguments), capstone-vm on PATH, CAPSTONE_BUILDROOT_DIR (the guest cross compiler).
# Port 21299, not memcached's 11211: on a shared host 11211 may belong to someone else's server.
set -uo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
R=${1:?results dir}; STOP=TERM; [ "${2:-}" = "--stop" ] && STOP=${3:?}
: "${MC_WORK:?}" "${CAPSTONE_VM_UP_ARGS:?}" "${CAPSTONE_BUILDROOT_DIR:?}"
PORT=21299; FLAGS=(-l 127.0.0.1 -p "$PORT" -U 0 -m 64 -t 4)
NAT=$MC_WORK/native/bin/memcached; DOM=$MC_WORK/domain/memcached.dom
[ -x "$NAT" ] && [ -f "$DOM" ] || { echo "need $NAT and $DOM (build-native.sh, build-domain.sh)" >&2; exit 2; }
mkdir -p "$R/bin" "$R/share"
gcc -O1 -Wall -o "$R/bin/mc-harness-host" "$APP/host/mc-harness/mc-harness.c" -lpthread || exit 2
"$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" -O1 -Wall -o "$R/share/mc-harness" \
  "$APP/host/mc-harness/mc-harness.c" -lpthread || exit 2
cp "$DOM" "$R/share/memcached.dom"
{ echo "native memcached $(sha256sum < "$NAT" | cut -c1-16)"; echo "domain memcached.dom $(sha256sum < "$DOM" | cut -c1-16)";
  echo "harness host $(sha256sum < "$R/bin/mc-harness-host" | cut -c1-16) guest $(sha256sum < "$R/share/mc-harness" | cut -c1-16)"; } | tee "$R/inputs.txt"

for v in null1 null2 value cas; do
  rm -rf "$R/native-$v"; mkdir -p "$R/native-$v"
  p=none; case $v in value|cas) p=$v ;; esac
  "$R/bin/mc-harness-host" --out "$R/native-$v" --port "$PORT" --stop "$STOP" --perturb "$p" -- "$NAT" "${FLAGS[@]}" \
    > "$R/native-$v/harness.log" 2>&1 || { echo "native $v: harness failed"; cat "$R/native-$v/harness.log"; exit 3; }
done
verdict() { printf '%-9s %-4s %s\n' "$1" "$2" "$3" | tee -a "$R/verdicts.txt"; }
: > "$R/verdicts.txt"
cmp -s "$R/native-null1/transcript.norm" "$R/native-null2/transcript.norm" \
  && verdict null PASS "two native runs identical ($(wc -c < "$R/native-null1/transcript.norm") bytes)" \
  || verdict null FAIL "two native runs differ: the script is not deterministic"
for v in value cas; do
  cmp -s "$R/native-null1/transcript.norm" "$R/native-$v/transcript.norm" \
    && verdict positive FAIL "perturbing the $v left the transcript unchanged: the comparison cannot see it" \
    || verdict positive PASS "perturbing the $v changes the transcript"
done

VM=$R/vm
trap 'capstone-vm --state "$VM" down > /dev/null 2>&1' EXIT
for attempt in $(seq 1 2400); do
  rm -rf "$VM"
  # shellcheck disable=SC2086
  capstone-vm --state "$VM" up $CAPSTONE_VM_UP_ARGS --share "$R/share" --boot-timeout 600 > "$R/up.log" 2>&1 && break
  grep -q "Another Capstone VM owns" "$R/up.log" || { tail -3 "$R/up.log"; exit 4; }
  sleep 3
done
rm -rf "$R/share/domain-run"
capstone-vm --state "$VM" exec sh -c "
  mkdir -p /tmp/mc && /mnt/host/mc-harness --out /tmp/mc --port $PORT --stop $STOP -- \
    /usr/bin/capstone-job /tmp/mc/job.json --user 65534:65534 -- /usr/bin/capstone-exec /mnt/host/memcached.dom ${FLAGS[*]}
  echo harness rc=\$?
  cp -r /tmp/mc /mnt/host/domain-run" > "$R/domain-exec.log" 2>&1
echo "domain exec rc=$?"; cat "$R/domain-exec.log"
D=$R/share/domain-run
[ -f "$D/transcript.norm" ] || { verdict oracle FAIL "the domain run left no transcript"; exit 5; }
cmp -s "$R/native-null1/transcript.norm" "$D/transcript.norm" \
  && verdict oracle PASS "domain transcript identical to native ($(wc -c < "$D/transcript.norm") bytes)" \
  || { verdict oracle FAIL "domain transcript differs from native"; diff "$R/native-null1/transcript.norm" "$D/transcript.norm" | head -20; }
n_id=$(cat "$R/native-null1/identity.txt"); d_id=$(cat "$D/identity.txt")
[ "$n_id" = $'STAT pointer_size 64\r' ] && [ "$d_id" = $'STAT pointer_size 128\r' ] \
  && verdict identity PASS "native 64, domain 128" || verdict identity FAIL "native '$n_id' domain '$d_id'"
n_st=$(sed -E 's/ stop_seconds=.*//' "$R/native-null1/status.txt"); d_job=$(cat "$D/job.json" 2>/dev/null)
want="{\"version\":1,\"kind\":\"$(echo "$n_st" | sed -E 's/.* (exit|signal)=.*/\1/')\",\"value\":$(echo "$n_st" | sed -E 's/.*=([0-9]+)$/\1/')}"
[ "$d_job" = "$want" ] && verdict status PASS "native '$n_st', domain $d_job" || verdict status FAIL "native '$n_st' (want $want), domain '$d_job'"
[ ! -s "$R/native-null1/server.err" ] && [ ! -s "$D/server.err" ] && verdict stderr PASS "both empty" \
  || verdict stderr FAIL "native $(wc -c < "$R/native-null1/server.err") bytes, domain $(wc -c < "$D/server.err") bytes"
echo "stop: native $(cat "$R/native-null1/status.txt") domain $(cat "$D/status.txt")"
