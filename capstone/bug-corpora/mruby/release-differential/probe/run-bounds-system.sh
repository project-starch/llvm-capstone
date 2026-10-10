#!/usr/bin/env bash
# The system-allocator cases on the bounds-only Capstone application domain (sysalloc-bounds:
# the first-fit heap with each allocation bounded and no revocation), one boot, with each
# fault's cause and pc kept so that the image's symbols can say where it lies.
#
#   run-bounds-system.sh <KIT> <OUT> <case>...      case: NN, from this corpus
#
# KIT holds the platform the 2026-10-06 sysalloc-bounds arm ran on and the arm's image:
#   qemu-system-riscv64 (+ libslirp.so.0), Image, fw_jump.elf, rootfs.ext2, capstone.ko,
#   capstone-exec, host/ (runtime/host of llvm-capstone 1b7d47d39732, the dev of that day),
#   mruby-bounds.dom (sha256 0f208270...), capi-NN.dom for a C-API case (probe/build-capi.sh
#   against that arm's build), smoke.rb and d40.rb/d500.rb
# Controls first, as on 2026-10-06: smoke.rb must print SMOKE_DONE and both recursion depths
# must complete, or no case is a reading. Each case runs under a 25 s watchdog (the guest's
# busybox has no timeout(1)) with CAPSTONE_EXEC_DIAGNOSTICS=1, without which the launcher
# reports a fault's cause only to a tty.
set -u
K=$(cd "${1:?KIT}" && pwd); OUT=${2:?OUT}; shift 2
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORPUS=$(cd -- "$HERE/.." && pwd)
mkdir -p "$OUT"; S=$K/share; rm -rf "$S"; mkdir -p "$S/cases" "$S/files"; chmod 777 "$S/files"
cp "$K/mruby-bounds.dom" "$K/smoke.rb" "$K/d40.rb" "$K/d500.rb" "$S/"
cat > "$S/one.sh" <<'GUEST'
#!/bin/sh
export CAPSTONE_EXEC_DIAGNOSTICS=1
cd /mnt/host
"$@" > /tmp/case.out 2>&1 &
pid=$!
( sleep 25; kill -9 $pid 2>/dev/null ) > /dev/null 2>&1 &
dog=$!
wait $pid
st=$?
kill $dog 2>/dev/null
head -c 6000 /tmp/case.out
echo "STATUS=$st"
GUEST
runs=()
for n in "$@"; do
  d=$(ls -d "$CORPUS"/$(printf %02d "$n")_*); trig=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trigger"])' "$d/case.json")
  nn=$(printf %02d "$n")
  if [[ $trig == *.c ]]; then cp "$K/capi-$nn.dom" "$S/"; runs+=("$nn:/mnt/host/capi-$nn.dom")
  else cp "$d/$trig" "$S/cases/$nn.rb"; runs+=("$nn:/mnt/host/mruby-bounds.dom /mnt/host/cases/$nn.rb"); fi
done
export PYTHONPATH=$K/host CAPSTONE_QEMU_LOCK=$K/qemu.lock LD_LIBRARY_PATH=$K
V=$K/vm; vm() { python3 -m capstone_vm --state "$V" "$@"; }
vm down > /dev/null 2>&1; rm -rf "$V"
vm up --qemu "$K/qemu-system-riscv64" --kernel "$K/Image" --firmware "$K/fw_jump.elf" --rootfs "$K/rootfs.ext2" \
  --share "$S" --module "$K/capstone.ko" --launcher "$K/capstone-exec" --cma-mib 1024 --process-cache-mib 768 \
  > "$OUT/up.log" 2>&1 || { echo "up FAILED"; tail -5 "$OUT/up.log"; exit 1; }
ex() { timeout 900 python3 -m capstone_vm --state "$V" exec sh /mnt/host/one.sh "$@"; }
ex /mnt/host/mruby-bounds.dom /mnt/host/smoke.rb > "$OUT/control-smoke.txt" 2>&1
for d in d40 d500; do ex /mnt/host/mruby-bounds.dom /mnt/host/$d.rb > "$OUT/control-$d.txt" 2>&1; done
echo "control smoke $(grep -c SMOKE_DONE "$OUT/control-smoke.txt") d40 $(grep -c '^DEEP' "$OUT/control-d40.txt") d500 $(grep -c '^DEEP' "$OUT/control-d500.txt")"
for r in "${runs[@]}"; do
  nn=${r%%:*}; cmd=${r#*:}
  # shellcheck disable=SC2086
  ex $cmd > "$OUT/case-$nn.txt" 2>&1
  echo "case $nn $(grep -o 'STATUS=[0-9]*' "$OUT/case-$nn.txt") $(grep -m1 -oE 'cause=[0-9]+[^\n]{0,80}' "$OUT/case-$nn.txt")"
done
vm down > /dev/null 2>&1
echo "BOUNDS-DONE"
