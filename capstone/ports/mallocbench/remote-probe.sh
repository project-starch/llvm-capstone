#!/bin/bash
# Run each named program of the mallocbench gate in its own QEMU, all in parallel.
#   remote-probe.sh RUNID SECONDS NAME...
# Each run gets its own stage (gate.sh with "only" = NAME) and its own lock file, so the
# runs do not serialise. Logs: runs/RUNID/NAME.log (rc line) and runs/RUNID/work-NAME/serial.log.
set -u
K=$(cd "$(dirname "$0")" && pwd); R=$K/runs/$1; T=$2; shift 2
mkdir -p "$R"
# What ran: the stage's checksums, QEMU and images, and the kit's own record of its commits.
{ date -u +%FT%TZ; cat "$K/SHA256SUMS"; sha256sum "$K/qemu-system-riscv64" "$K"/images/*;
  cat "$K/PROVENANCE" 2>/dev/null; } > "$R/provenance.txt"
for n in "$@"; do
  s=$R/stage-$n; mkdir -p "$s"; cp -a "$K/stage-base/." "$s/"; echo "$n" > "$s/only"; echo "$T" > "$s/timeout"
  ( CAPSTONE_QEMU_LOCK=$R/lock-$n LD_LIBRARY_PATH=$K/lib python3 "$K/run-staged.py" --qemu "$K/qemu-system-riscv64" \
      --images "$K/images" --stage "$s" --work "$R/work-$n" --timeout $((T + 900)) > "$R/$n.log" 2>&1
    echo "rc=$?" >> "$R/$n.log" ) &
done
wait
echo PROBE-DONE > "$R/DONE"
