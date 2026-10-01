#!/usr/bin/env bash
# The memcached oracle. Native reference runs on the host (two null runs, two perturbed), then one
# domain run in one guest boot, and every comparison the plan names:
#   null      two native runs: transcripts identical (the script is deterministic);
#   positive  a perturbed value byte, and a perturbed cas value: each transcript must differ;
#   identity  `stats pointer_size` reads 64 natively and 128 in the domain;
#   oracle    the domain's normalised transcript is byte-identical to the native one, its exit status
#             (capstone-job's record) matches the native one, and its stderr is empty like native's.
#   workers   --marker only (M3): the images of host/build-marker.sh, which print "MC-WORKER <i> conn"
#             per connection a worker sets up; per-worker counts equal native's, every worker present,
#             the stderr otherwise compared as above, and a control that the gate fails on no markers.
#   run-oracle.sh <results-dir> [--stop TERM|USR1] [--arm level0|shrink|sublet] [--runs N] [--marker]
#             [--alternate STOCK-EXEC FIXED-EXEC] [--image DOM] [--server-opts "OPTS"] [--env NAME=VALUE]...
#             (N domain runs, one boot)
#   --image runs that domain image in place of the arm's; --server-opts appends OPTS to the server's
#   flags, native and domain alike (a later -m overrides the default); --env sets NAME=VALUE for the
#   server, native and domain alike (the slabsublet arm's MC_SLAB_SUBLET_MODE).
#   --alternate runs odd-numbered runs under STOCK-EXEC and even-numbered ones under FIXED-EXEC (two
#   capstone-exec binaries copied to the share), so a launcher pair is compared within one boot.
# Env: MC_WORK (build-native.sh, build-domain.sh outputs), CAPSTONE_VM_UP_ARGS (capstone-vm up platform
# arguments), capstone-vm on PATH, CAPSTONE_BUILDROOT_DIR (the guest cross compiler).
# Port 21299, not memcached's 11211: on a shared host 11211 may belong to someone else's server.
set -uo pipefail
APP=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
R=${1:?results dir}; shift; STOP=TERM; ARM=shrink; RUNS=1; MARKER=0; ALT_STOCK=; ALT_FIXED=; IMAGE=; OPTS=; ENVS=()
while [ $# -gt 0 ]; do case $1 in --stop) STOP=$2; shift 2 ;; --arm) ARM=$2; shift 2 ;; --runs) RUNS=$2; shift 2 ;;
  --marker) MARKER=1; shift ;;
  --alternate) ALT_STOCK=$2; ALT_FIXED=$3; shift 3 ;;
  --image) IMAGE=$2; shift 2 ;;
  --server-opts) OPTS=$2; shift 2 ;;
  --env) ENVS+=("$2"); shift 2 ;;
  *) echo "unknown option $1" >&2; exit 2 ;; esac; done
: "${MC_WORK:?}" "${CAPSTONE_VM_UP_ARGS:?}" "${CAPSTONE_BUILDROOT_DIR:?}"
PORT=21299; NTHREADS=4; FLAGS=(-l 127.0.0.1 -p "$PORT" -U 0 -m 64 -t "$NTHREADS")
# shellcheck disable=SC2206
[ -n "$OPTS" ] && FLAGS+=($OPTS)
NAT=$MC_WORK/native/bin/memcached; DOM=$MC_WORK/domain/memcached-$ARM.dom
[ -n "$IMAGE" ] && DOM=$IMAGE
if [ "$MARKER" = 1 ]; then
  [ "$ARM" = shrink ] || { echo "--marker: the marker domain image is built on the shrink runtime only" >&2; exit 2; }
  NAT=$MC_WORK/marker/memcached-native; DOM=$MC_WORK/marker/memcached.dom
fi
[ -x "$NAT" ] && [ -f "$DOM" ] || { echo "need $NAT and $DOM (build-native.sh, build-domain.sh)" >&2; exit 2; }
mkdir -p "$R/bin" "$R/share"
gcc -O1 -Wall -o "$R/bin/mc-harness-host" "$APP/host/mc-harness/mc-harness.c" -lpthread || exit 2
"$CAPSTONE_BUILDROOT_DIR/build/host/bin/riscv64-buildroot-linux-gnu-gcc" -O1 -Wall -o "$R/share/mc-harness" \
  "$APP/host/mc-harness/mc-harness.c" -lpthread || exit 2
cp "$DOM" "$R/share/memcached.dom"
if [ -n "$ALT_STOCK" ]; then
  cp "$ALT_STOCK" "$R/share/exec-stock"; cp "$ALT_FIXED" "$R/share/exec-fixed"; chmod 755 "$R/share/exec-stock" "$R/share/exec-fixed"
  echo "alternate: stock $(sha256sum < "$ALT_STOCK" | cut -c1-16) fixed $(sha256sum < "$ALT_FIXED" | cut -c1-16)" | tee -a "$R/inputs.txt"
fi
{ echo "native memcached $(sha256sum < "$NAT" | cut -c1-16)"; echo "domain memcached.dom $(sha256sum < "$DOM" | cut -c1-16)";
  echo "server flags: ${FLAGS[*]}"; echo "server env: ${ENVS[*]:-none}";
  echo "harness host $(sha256sum < "$R/bin/mc-harness-host" | cut -c1-16) guest $(sha256sum < "$R/share/mc-harness" | cut -c1-16)"; } | tee "$R/inputs.txt"

for v in null1 null2 value cas; do
  rm -rf "$R/native-$v"; mkdir -p "$R/native-$v"
  p=none; case $v in value|cas) p=$v ;; esac
  env ${ENVS[@]+"${ENVS[@]}"} "$R/bin/mc-harness-host" --out "$R/native-$v" --port "$PORT" --stop "$STOP" --perturb "$p" -- "$NAT" "${FLAGS[@]}" \
    > "$R/native-$v/harness.log" 2>&1 || { echo "native $v: harness failed"; cat "$R/native-$v/harness.log"; exit 3; }
done
verdict() { printf '%-9s %-4s %s\n' "$1" "$2" "$3" | tee -a "$R/verdicts.txt"; }
: > "$R/verdicts.txt"
# M3: per-worker connection counts from two stderr files; PASS when equal and every worker present.
workers() { python3 - "$1" "$2" "$NTHREADS" <<'EOF'
import collections, re, sys
def counts(p):
    return collections.Counter(int(w) for w in re.findall(rb'^MC-WORKER (\d+) conn$', open(p, 'rb').read(), re.M))
n, d = counts(sys.argv[1]), counts(sys.argv[2])
fmt = lambda c: ' '.join(f'{k}:{c[k]}' for k in sorted(c)) or 'none'
ok = n == d and set(d) == set(range(int(sys.argv[3])))
print(f"native {fmt(n)}, domain {fmt(d)}")
sys.exit(0 if ok else 1)
EOF
}
nomark() { sed -E '/^MC-WORKER [0-9]+ conn$/d' "$1"; }
cmp -s "$R/native-null1/transcript.norm" "$R/native-null2/transcript.norm" \
  && verdict null PASS "two native runs identical ($(wc -c < "$R/native-null1/transcript.norm") bytes)" \
  || verdict null FAIL "two native runs differ: the script is not deterministic"
if [ "$MARKER" = 1 ]; then
  out=$(workers "$R/native-null1/server.err" "$R/native-null2/server.err") && verdict wnull PASS "two native runs: $out" \
    || verdict wnull FAIL "two native runs: $out"
  out=$(workers "$R/native-null1/server.err" /dev/null) && verdict wcontrol FAIL "the gate passes with no marker line: $out" \
    || verdict wcontrol PASS "the gate fails with no marker line ($out)"
fi
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
# capstone-job forwards SIGINT, SIGTERM and SIGHUP and nothing else (runtime/linux/job.c): a SIGUSR1 sent
# to it kills the helper and never reaches the domain (the first M4 SIGUSR1 run, 2026-10-01). So for USR1
# the harness signals capstone-job's child, capstone-exec, itself; TERM keeps the path M5 ran.
SIGCHILD=; [ "$STOP" = TERM ] || SIGCHILD=--signal-child
# the launcher and job helper the guest runs: capstone-vm up --launcher/--job-helper decide them
capstone-vm --state "$VM" exec sh -c 'sha256sum /usr/bin/capstone-exec /usr/bin/capstone-job' 2>&1 \
  | sed 's/^/guest /' | tee -a "$R/inputs.txt"
for n in $(seq 1 "$RUNS"); do
EXEC=/usr/bin/capstone-exec; LBL=$ARM
if [ -n "$ALT_STOCK" ]; then if [ $((n % 2)) = 1 ]; then EXEC=/mnt/host/exec-stock; LBL=$ARM/stock; else EXEC=/mnt/host/exec-fixed; LBL=$ARM/fixed; fi; fi
rm -rf "$R/share/domain-run$n"
capstone-vm --state "$VM" exec sh -c "
  rm -rf /tmp/mc /tmp/mc-fault.txt; mkdir -p /tmp/mc && CAPSTONE_FAULT_RECORD=/tmp/mc-fault.txt ${ENVS[*]} /mnt/host/mc-harness --out /tmp/mc --port $PORT --stop $STOP $SIGCHILD -- \
    /usr/bin/capstone-job /tmp/mc/job.json --user 65534:65534 -- $EXEC /mnt/host/memcached.dom ${FLAGS[*]}
  echo harness rc=\$?
  if [ -s /tmp/mc-fault.txt ]; then cp /tmp/mc-fault.txt /tmp/mc/fault.txt; fi
  cp -r /tmp/mc /mnt/host/domain-run$n; true" > "$R/domain-exec$n.log" 2>&1
echo "domain run $n ($LBL): exec rc=$?"; cat "$R/domain-exec$n.log"
D=$R/share/domain-run$n
[ -s "$D/fault.txt" ] && { echo "--- domain fault record"; cat "$D/fault.txt"; }
[ -f "$D/transcript.norm" ] || { verdict "oracle$n" FAIL "[$LBL] the domain run left no transcript"; continue; }
cmp -s "$R/native-null1/transcript.norm" "$D/transcript.norm" \
  && verdict "oracle$n" PASS "[$LBL] domain transcript identical to native ($(wc -c < "$D/transcript.norm") bytes)" \
  || { verdict "oracle$n" FAIL "[$LBL] domain transcript differs from native"; diff "$R/native-null1/transcript.norm" "$D/transcript.norm" | head -20; }
n_id=$(cat "$R/native-null1/identity.txt"); d_id=$(cat "$D/identity.txt")
[ "$n_id" = $'STAT pointer_size 64\r' ] && [ "$d_id" = $'STAT pointer_size 128\r' ] \
  && verdict "identity$n" PASS "native 64, domain 128" || verdict "identity$n" FAIL "native '$n_id' domain '$d_id'"
n_st=$(sed -E 's/ stop_seconds=.*//' "$R/native-null1/status.txt"); d_job=$(cat "$D/job.json" 2>/dev/null)
want="{\"version\":1,\"kind\":\"$(echo "$n_st" | sed -E 's/.* (exit|signal)=.*/\1/')\",\"value\":$(echo "$n_st" | sed -E 's/.*=([0-9]+)$/\1/')}"
[ "$d_job" = "$want" ] && verdict "status$n" PASS "native '$n_st', domain $d_job" || verdict "status$n" FAIL "native '$n_st' (want $want), domain '$d_job'"
cmp -s <(nomark "$R/native-null1/server.err") <(nomark "$D/server.err") \
  && verdict "stderr$n" PASS "identical ($(nomark "$D/server.err" | wc -c) bytes$([ "$MARKER" = 1 ] && echo ' besides the marker lines'))" \
  || verdict "stderr$n" FAIL "native $(nomark "$R/native-null1/server.err" | wc -c) bytes, domain $(nomark "$D/server.err" | wc -c) bytes"
if [ "$MARKER" = 1 ]; then
  out=$(workers "$R/native-null1/server.err" "$D/server.err") && verdict "workers$n" PASS "$out" || verdict "workers$n" FAIL "$out"
fi
echo "stop: native $(cat "$R/native-null1/status.txt") domain $(cat "$D/status.txt")"
done
