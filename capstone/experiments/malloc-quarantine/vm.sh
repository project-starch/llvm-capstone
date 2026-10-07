#!/usr/bin/env bash
# Own CheriBSD purecap guest for the malloc/quarantine experiments.
#   vm.sh up      boot (snapshot: the disk image is never written), install a temporary SSH key
#   vm.sh ssh CMD run CMD in the guest
#   vm.sh put F.. copy files to /root/mq/ in the guest
#   vm.sh down    power off
set -euo pipefail
CHERI=${CHERI:-$HOME/cheri/output}
WORK=${MQ_WORK:-/tmp/capstone/malloc-quarantine}
PORT=${MQ_PORT:-10461}
MEM=${MQ_MEM:-2048}
PY=${PY:-$HOME/.venvs/capstone/bin/python3}
HERE=$(cd "$(dirname "$0")" && pwd)
KEY=$WORK/vm/key
SOCK=$WORK/vm/serial.sock
SSHOPT=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

case "${1:-}" in
up)
  mkdir -p "$WORK/vm"
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N '' -f "$KEY"
  rm -f "$SOCK"
  nohup "$CHERI/sdk/bin/qemu-system-riscv64cheri" -M virt -m "$MEM" -smp 1 -nographic -snapshot \
    -bios "$CHERI/sdk/share/qemu/bbl-riscv64cheri-virt-fw_jump.bin" \
    -kernel "$CHERI/rootfs-riscv64-purecap/boot/kernel/kernel" \
    -drive if=none,file="$CHERI/cheribsd-riscv64-purecap.img",id=drv,format=raw \
    -device virtio-blk-device,drive=drv \
    -device virtio-net-device,netdev=net0 -netdev user,id=net0,hostfwd=tcp:127.0.0.1:$PORT-:22 \
    -device virtio-rng-pci -monitor none \
    -chardev socket,id=s0,path="$SOCK",server=on,wait=off,logfile="$WORK/vm/serial.log" -serial chardev:s0 \
    ${MQ_QEMU_EXTRA:-} > "$WORK/vm/qemu.out" 2>&1 &
  echo $! > "$WORK/vm/qemu.pid"
  "$PY" "$HERE/vm-login.py" "$SOCK" "$KEY.pub"
  ssh "${SSHOPT[@]}" root@127.0.0.1 'uname -a; mkdir -p /root/mq'
  ;;
ssh) shift; ssh "${SSHOPT[@]}" root@127.0.0.1 "$@" ;;
put) shift; scp -q -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$@" root@127.0.0.1:/root/mq/ ;;
down)
  ssh "${SSHOPT[@]}" root@127.0.0.1 'shutdown -p now' || true
  sleep 20; kill "$(cat "$WORK/vm/qemu.pid")" 2>/dev/null || true ;;
*) echo "usage: $0 up|ssh|put|down" >&2; exit 2 ;;
esac
