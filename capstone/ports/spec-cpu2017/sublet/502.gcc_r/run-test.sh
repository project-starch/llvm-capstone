#!/bin/bash
# Run 502.gcc_r images on SPEC's test input in one guest boot and print one result line per run.
# usage: run-test.sh PLATFORM INPUT WORK NAME=IMAGE [NAME=IMAGE...]
#   PLATFORM  a platform kit: qemu-system-riscv64, images/ and adapter/capstone-vexec
#   INPUT     the test input t1.c of the licensed tree (data/test/input/t1.c)
#   WORK      a new directory for the stage and the run
#   NAME=IMAGE  the executables to compare, for example stock=.../cpugcc_r sublet=.../cpugcc_r
# Source capstone/tests/capstone-test-env.sh first, from the checkout whose
# capstone/runtime/virtual/run-staged.py matches PLATFORM.
# environment:
#   ROUNDS     rounds over all images (default 3). The first round is cold.
#   REFERENCE  SPEC's expected output (data/test/output/t1.opts-O3_-finline-limit_50000.s)
#   ICOUNT     a static guest build of icount.c. When set, QEMU runs with -icount shift=0 and each
#              line carries the instructions of the whole job. Guest seconds are then virtual.
#   TIMEOUT    seconds for the boot (default 900)
# The command is the benchmark's own for this input: t1.c -O3 -finline-limit=50000.
set -euo pipefail

[ $# -ge 4 ] || { sed -n '2,6p' "$0" >&2; exit 2; }
PLATFORM=$(cd "$1" && pwd); INPUT=$2; WORK=$3; shift 3
REPO=${CAPSTONE_REPO_ROOT:?source capstone/tests/capstone-test-env.sh first}
: "${CAPSTONE_QEMU_LOCK:?source capstone/tests/capstone-test-env.sh first}"
ROUNDS=${ROUNDS:-3}
TIMEOUT=${TIMEOUT:-900}
[ -e "$WORK" ] && { echo "run-test: $WORK exists" >&2; exit 2; }
mkdir -p "$WORK/stage"; WORK=$(cd "$WORK" && pwd)

NAMES=()
for pair in "$@"; do
  name=${pair%%=*}; image=${pair#*=}
  [ -f "$image" ] || { echo "run-test: no image $image" >&2; exit 2; }
  cp "$image" "$WORK/stage/gcc.$name"
  sha256sum "$image" | sed "s|  .*|  $name|" >> "$WORK/SHA256SUMS"
  NAMES+=("$name")
done
cp "$INPUT" "$WORK/stage/t1.c"
cp "$PLATFORM/adapter/capstone-vexec" "$WORK/stage/"
PREFIX=""; EXTRA=()
if [ -n "${ICOUNT:-}" ]; then
  cp "$ICOUNT" "$WORK/stage/icount"; PREFIX="./icount"; EXTRA=(--icount 0)
fi

# A native task opens files beneath its start directory only, and the stage is read-only,
# so the gate works in a copy under /tmp.
cat > "$WORK/stage/gate.sh" <<GATE
#!/bin/sh
mkdir -p /tmp/gcc-test && cp /mnt/vm/gcc.* /mnt/vm/t1.c /mnt/vm/capstone-vexec /tmp/gcc-test/ || exit 1
[ -f /mnt/vm/icount ] && cp /mnt/vm/icount /tmp/gcc-test/
cd /tmp/gcc-test || exit 1
round=1
while [ \$round -le $ROUNDS ]; do
  for name in ${NAMES[*]}; do
    rm -f out.s
    start=\$(cut -d' ' -f1 /proc/uptime)
    counted=\$($PREFIX ./capstone-vexec ./gcc.\$name t1.c -O3 -finline-limit=50000 -o out.s 2> err.txt)
    rc=\$?
    end=\$(cut -d' ' -f1 /proc/uptime)
    echo "GCC_TEST name=\$name round=\$round rc=\$rc start=\$start end=\$end \$counted sha256=\$(sha256sum out.s 2>/dev/null | cut -c1-64)"
  done
  round=\$((round + 1))
done
dmesg | grep 'capstone_task: fault' | sed 's/^.*capstone_task: fault/GCC_FAULT/'
echo VIRTUAL_STAGED_DONE
GATE
chmod +x "$WORK/stage/gate.sh"

python3 "$REPO/capstone/runtime/virtual/run-staged.py" --qemu "$PLATFORM/qemu-system-riscv64" \
  --images "$PLATFORM/images" --stage "$WORK/stage" --work "$WORK/run" --timeout "$TIMEOUT" \
  "${EXTRA[@]}" > "$WORK/runner.log" 2>&1 || { echo "run-test: the boot did not complete, see $WORK/runner.log" >&2; exit 1; }

expected=""
[ -n "${REFERENCE:-}" ] && expected=$(sha256sum "$REFERENCE" | cut -c1-64)
grep -aE '^GCC_(TEST|FAULT)' "$WORK/run/serial.log" | tr -d '\r' | awk -v expected="$expected" '
  /^GCC_TEST/ {
    split("", field)
    for (i = 2; i <= NF; i++) { split($i, kv, "="); field[kv[1]] = kv[2] }
    verdict = expected == "" ? "unchecked" : (field["sha256"] == expected ? "MATCH" : "MISMATCH")
    counted = ("instructions" in field) ? " instructions=" field["instructions"] " status=" field["status"] : ""
    printf "%s %s %s %s seconds=%.2f%s output=%s\n", $1, $2, $3, $4, field["end"] - field["start"], counted, verdict
    next
  }
  { print }' | tee "$WORK/result-lines.txt"
