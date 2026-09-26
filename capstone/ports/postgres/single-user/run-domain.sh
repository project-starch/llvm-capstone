#!/usr/bin/env bash
# Replay one of initdb's backend invocations in a domain under QEMU.
#
#   run-domain.sh <call> [seconds]
#   PGSU_CLUSTER=<data directory> [PGSU_SQL=<file>] run-domain.sh work [seconds]
#
# <call> is a call number from record-initdb.sh (4 is `--boot` over the catalog
# script, 5 the `--single` session over the setup SQL). The run stages, on the
# 9p share the runner mounts at /mnt/host:
#
#   bin/postgres.dom  the image build-domain.sh linked ($PG_SU_ROOT/link/); it
#                  is argv[0], and the backend takes its share directory from
#                  it: <dir>/../share when the directory is named bin
#   share/         the native install's share/postgresql (timezonesets, the
#                  setup SQL, and timezone/, which the guest links to the
#                  /usr/share/zoneinfo the image was configured with)
#   pgdata/        the recorded data directory the call started from, mode 0700
#   pg-input       the call's standard input; the backend reads it from the
#                  file PG_BOOT_INPUT / PG_SINGLE_INPUT names (patch 0003),
#                  since a domain has no stdin
#   pg-args        the call's arguments, one per line, plus PGSU_EXTRA_ARGS
#                  (default: shared_buffers and dynamic_shared_memory_type a
#                  domain can serve); pg-env its environment, with PGDATA
#                  pointing at the staged copy (toolchain/domain_entry.c)
#   lt.user        the host process, musl-capstone's libc_test_host.c
#
# The catalog script is regenerated from the template rather than taken as
# initdb wrote it: initdb substitutes sizeof(Pointer) and its alignment for
# the one pointer-sized type (pg_ddl_command), 8 and "d" natively, 16 and "d"
# here; every other substitution is checked equal to the recording.
#
# Prints the domain's output and the host's LT-RESULT line; exit 0 when the
# backend exited 0. A boot that ends with neither a result nor a domain halt is
# retried, PGSU_ATTEMPTS times (default 3), as the CPython runner does for the
# guest stalls it met. PGSU_QEMU_MONITOR=<socket> adds a QEMU monitor on that
# UNIX socket, so a spinning run can be asked where the vCPU is:
#   echo 'info registers' | socat - UNIX-CONNECT:<socket> The image needs a domain block larger than the buddy
# allocator's 4 MiB: the runner boots CAPSTONE_BUILDROOT_DIR (default the
# snapshot whose rootfs module carries CMA-backed blocks) with cma= sized for
# one block per run plus one spare, as the CPython port's runner does.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh" >/dev/null

CALL=${1:?usage: run-domain.sh <call> [seconds]}
SECONDS_LIMIT=${2:-900}
ROOT=${PG_SU_ROOT:-$CAPSTONE_TMP_ROOT/pg-single-user}
RT=${RUNTIME_REPO:-$CAPSTONE_REPO_ROOT}
MUSL_PORT=$RT/capstone/ports/musl-capstone
MODSRC=${PGSU_MODCAPSTONE_SRC:-$RT/capstone/caplifive-buildroot/package/modcapstone}
BR=${CAPSTONE_BUILDROOT_DIR:?}/build
IMAGE=${PGSU_IMAGE:-$ROOT/link/postgres.dom}
REC=$ROOT/initdb-rec
WORK=$ROOT/run
SHARE=$WORK/share
EXTRA=${PGSU_EXTRA_ARGS-"-c shared_buffers=4MB -c max_connections=10 -c dynamic_shared_memory_type=sysv"}

[[ -f "$IMAGE" ]] || { echo "no image $IMAGE; run build-domain.sh" >&2; exit 2; }
if [[ $CALL == work ]]; then
  CLUSTER=${PGSU_CLUSTER:?PGSU_CLUSTER=<a data directory initdb finished: the share/pgdata of a passing call-5 run>}
  INPUT=${PGSU_SQL:-$SCRIPT_DIR/work.sql}
  [[ -f "$CLUSTER/PG_VERSION" ]] || { echo "PGSU_CLUSTER=$CLUSTER is not a data directory" >&2; exit 2; }
  [[ -f "$INPUT" ]] || { echo "no SQL file $INPUT" >&2; exit 2; }
else
  [[ -f "$REC/call-$CALL.args" ]] || { echo "no recorded call $CALL under $REC; run record-initdb.sh" >&2; exit 2; }
  [[ -f "$REC/call-$CALL.stdin" ]] || { echo "call $CALL took no input; only --boot and --single calls are replayed" >&2; exit 2; }
  CLUSTER=$REC/pgdata-before-$CALL INPUT=$REC/call-$CALL.stdin
fi

rm -rf "$WORK"; mkdir -p "$SHARE/bin"
"$CAPSTONE_LLVM_BIN/llvm-objcopy" --strip-debug "$IMAGE" "$SHARE/bin/postgres.dom"
cp -a "$ROOT/pg-native/share/postgresql" "$SHARE/share"
cp -a "$CLUSTER" "$SHARE/pgdata"
chmod 0700 "$SHARE/pgdata"
# "work": a single-user session over a SQL file (default work.sql) in the database
# postgres, as survey-native.sh runs it natively.
ARGS=$REC/call-$CALL.args
if [[ $CALL == work ]]; then ARGS=$WORK/work.args; printf '%s\n' --single postgres > "$ARGS"; fi

# The input. For --boot, the template with this target's substitutions; the
# recording must then differ from it in the pointer-sized rows alone (the
# pg_ddl_command and internal types: 8 there, 16 here), or initdb substituted
# something this script does not know about.
if grep -qx -- '--boot' "$ARGS"; then
  python3 - "$ROOT/pg-native/share/postgresql/postgres.bki" "$INPUT" "$SHARE/pg-input" <<'PY'
import sys
template, recorded, out = sys.argv[1:]
subst = [("NAMEDATALEN", "64"), ("SIZEOF_POINTER", "16"), ("ALIGNOF_POINTER", "d"),
         ("FLOAT8PASSBYVAL", "true"), ("POSTGRES", "'pg'"), ("ENCODING", "6"),
         ("LC_COLLATE", "'C'"), ("LC_CTYPE", "'C'"), ("DATLOCALE", "_null_"),
         ("ICU_RULES", "_null_"), ("LOCALE_PROVIDER", "c")]   # strings as escape_quotes_bki writes them
text = open(template).read()
for token, value in subst:            # initdb's replace_token: every occurrence, in this order
    text = text.replace(token, value)
open(out, "w").write(text)
rec = open(recorded).read().splitlines()
new = text.splitlines()
diff = [(a, b) for a, b in zip(rec, new) if a != b]
bad = [(a, b) for a, b in diff if a.replace(" 8 ", " 16 ", 1) != b]
if len(rec) != len(new) or bad:
    print("the regenerated catalog script differs from initdb's beyond the pointer size:", file=sys.stderr)
    for a, b in bad[:5]:
        print("  recorded:", a[:120], "\n  here:    ", b[:120], file=sys.stderr)
    sys.exit(2)
print(f"boot input: {len(new)} lines, {len(diff)} row(s) differ from the native recording (16-byte pointer types)")
PY
else
  # The setup SQL names files of the install initdb ran from (COPY ... FROM
  # '<prefix>/share/postgresql/sql_features.txt'); the domain sees that directory
  # as /mnt/host/share. Each named file must be in the staged share.
  python3 - "$INPUT" "$SHARE/pg-input" "$SHARE/share" <<'PY'
import os, re, sys
recorded, out, share = sys.argv[1:]
text = open(recorded).read()
pat = re.compile(r"/[^'\s]*/share/postgresql/([^'\s]+)")
names = pat.findall(text)
missing = [n for n in names if not os.path.isfile(os.path.join(share, n))]
if missing:
    sys.exit(f"setup SQL names install files the share lacks: {missing}")
open(out, "w").write(pat.sub(lambda m: "/mnt/host/share/" + m.group(1), text))
print(f"single input: {len(names)} install path(s) moved to /mnt/host/share: {sorted(set(names))}")
PY
fi

# Arguments and environment, as recorded, with the paths moved to the share.
# For --single the extra options go before the database name, which comes last; an option
# after it is rejected ("invalid command-line argument: -c"). Only there: --boot's last
# argument is the value of -X, and splitting the pair breaks it.
{
  mapfile -t ARGV < "$ARGS"
  last=${ARGV[${#ARGV[@]}-1]}
  if [[ ${ARGV[0]:-} == --single && $last != -* ]]; then
    printf '%s\n' "${ARGV[@]:0:${#ARGV[@]}-1}"; [[ -n $EXTRA ]] && printf '%s\n' $EXTRA; printf '%s\n' "$last"
  else
    printf '%s\n' "${ARGV[@]}"; [[ -n $EXTRA ]] && printf '%s\n' $EXTRA
  fi
} > "$SHARE/pg-args"
{
  echo "PGDATA=/mnt/host/pgdata"
  echo "PGSU_DOMAIN=1"
  if grep -qx -- '--boot' "$ARGS"; then echo "PG_BOOT_INPUT=/mnt/host/pg-input"
  else echo "PG_SINGLE_INPUT=/mnt/host/pg-input"; fi
  [[ -n ${PGSU_EXTRA_ENV:-} ]] && printf '%s\n' $PGSU_EXTRA_ENV
} > "$SHARE/pg-env"
echo "call $CALL: $(tr '\n' ' ' < "$SHARE/pg-args")"
echo "  env: $(tr '\n' ' ' < "$SHARE/pg-env")"

"$BR/host/bin/riscv64-buildroot-linux-gnu-gcc" -O2 \
  -I"$MUSL_PORT/libc-test" -I"$MUSL_PORT/runtime" -I"$MODSRC/userspace/lib" \
  -I"$RT/capstone/tests/runtime-qemu/hostcall-stdout-probe" -I"$RT/capstone/tests/runtime-qemu" \
  -o "$SHARE/lt.user" "$MUSL_PORT/libc-test/libc_test_host.c" "$MODSRC/userspace/lib/libcapstone.c"

# The image goes to the guest's tmpfs first; the domain's output goes straight
# to the console, so what a halting domain printed before is kept. The sed is
# for run-domain-smoke.py, which takes "# " for the shell prompt.
#
# PGSU_STALL_DUMP=<seconds> adds a guest-side watchdog: every that many seconds while
# lt.user lives, each of its threads' state, wait channel and kernel stack. A run
# that stops with the vCPU in the idle task (neither the domain nor its helper
# runnable) then says what the helper is blocked on. It polls every 5 s so that it
# ends with lt.user, and writes to /dev/console, not to the pipe into sed: the first
# version slept the whole period with the pipe open, and a finished run then waited
# for it (the guest's date read the same second at the start and end of a whole run).
STALL_DUMP=
if [[ -n ${PGSU_STALL_DUMP:-} ]]; then
  STALL_DUMP="( sleep 5; n=0; while p=\$(pidof lt.user); do sleep 5; n=\$((n + 5)); \
[ \$n -ge $PGSU_STALL_DUMP ] || continue; n=0; \
for t in /proc/\$p/task/*; do echo \"PGSU-STALL-DUMP t=\$(date +%s) task=\${t##*/} \$(grep State: \$t/status) wchan=\$(cat \$t/wchan)\"; \
sed 's/^/PGSU-STACK /' \$t/stack 2>&1; done; done ) > /dev/console 2>&1 < /dev/null &"
fi
cat > "$SHARE/pg-run.sh" <<EOF
mkdir -p /usr/share && ln -sfn /mnt/host/share/timezone /usr/share/zoneinfo && echo PGSU-ZONEINFO-LINKED
cp /mnt/host/bin/postgres.dom /tmp/postgres.dom && echo PGSU-IMAGE-COPIED
$STALL_DUMP
echo PGSU-RUN-BEGIN call=$CALL t=\$(date +%s)
/tmp/lt.user /tmp/postgres.dom $SECONDS_LIMIT 2>&1
echo PGSU-RUN-END call=$CALL rc=\$? t=\$(date +%s)
ls -la /mnt/host/pgdata /mnt/host/pgdata/base/1 2>&1 | head -40
EOF
GUEST="echo __BOOT_OK__; cp /mnt/host/lt.user /tmp/lt.user && chmod 0755 /tmp/lt.user; \
sh /mnt/host/pg-run.sh 2>&1 | sed 's/#[ ]/#_/g'; dmesg | grep -iE 'Domain block|beyond the buddy|Failed to allocate' | tail -3; echo __ALL_DONE__"
[[ $GUEST != *"# "* ]] || { echo "the guest command contains '# '" >&2; exit 2; }
(( ${#GUEST} < 1000 )) || { echo "guest command is ${#GUEST} bytes" >&2; exit 2; }

python3 -c "import pexpect" 2>/dev/null \
  || { echo "run-domain-smoke.py needs pexpect in this python3 (activate the venv)" >&2; exit 2; }
# See the CPython runner for both: gp is re-fabricated LINEAR at every cjalr
# without CAPSTONE_GP_NONLIN=1, and the module never frees a domain's block.
export CAPSTONE_GP_NONLIN=${CAPSTONE_GP_NONLIN:-1}
CMA_MB=${PGSU_CMA_MB:-256}
for attempt in $(seq 1 "${PGSU_ATTEMPTS:-3}"); do
  rc=0
  CAPSTONE_QEMU_LOGIN_TIMEOUT=${CAPSTONE_QEMU_LOGIN_TIMEOUT:-240} \
  CAPSTONE_GUEST_COMMAND_TIMEOUT=$((SECONDS_LIMIT + 330)) \
  capstone_with_qemu_lock python3 "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$SHARE" --log-file "$WORK/qemu.log" --timeout-multiplier 8 \
    --kernel-arg cma=${CMA_MB}M --guest-command "$GUEST" \
    ${PGSU_QEMU_MONITOR:+--qemu-extra-arg=-monitor --qemu-extra-arg=unix:$PGSU_QEMU_MONITOR,server,nowait} \
    --success-marker __BOOT_OK__ --success-marker __ALL_DONE__ > "$WORK/runner.out" 2>&1 || rc=$?
  echo "attempt $attempt: runner rc=$rc (log $WORK/qemu.log)"
  grep -aqE '^LT-RESULT|domain halted' "$WORK/qemu.log" && break
  echo "attempt $attempt: neither a result nor a halt; last lines:"
  grep -avE 'remote fence' "$WORK/qemu.log" | tail -3 | sed 's/^/    /'
  cp "$WORK/qemu.log" "$WORK/qemu-stalled-$attempt.log"
done
sed -n '/PGSU-RUN-BEGIN/,/PGSU-RUN-END/p' "$WORK/qemu.log" | grep -av 'remote fence extension' \
  | grep -av 'echo PGSU-RUN' || true
grep -aE 'Domain block|beyond the buddy|Failed to allocate|capstone-domain:' "$WORK/qemu.log" | grep -v 'grep -iE' || true
result=$(grep -a -m1 "^LT-RESULT postgres.dom " "$WORK/qemu.log" || true)
[[ -n "$result" ]] || { echo "no LT-RESULT line: the domain did not finish (see $WORK/qemu.log)" >&2; exit 1; }
[[ "$result" == *" PASS"* ]]
