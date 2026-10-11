#!/usr/bin/env bash
# Resolve the QEMU binary, bbl firmware, kernel and disk image that provisioning
# actually produced, and write a qemu argv file (one token per line) for
# cheri-run.py.
#
# WHY THIS EXISTS RATHER THAN USING provision-cheri-vehicle.sh's OWN ARGV:
# cheribuild rewrites the target `riscv64cheri-softmmu` to `riscv64xcheri-softmmu`
# (pycheribuild/projects/build_qemu.py:427), so the installed emulator is
# qemu-system-riscv64Xcheri. provision-cheri-vehicle.sh tests for, and writes
# into its argv, `qemu-system-riscv64cheri` without the x -- so on this
# cheribuild it both rebuilds QEMU every run and emits an argv naming a binary
# that does not exist. The sibling run-cheri-baseline.sh already greps for the
# x-form, which is the correct one.
#
# Rather than edit that shared script, resolve everything here by globbing and
# ASSERT each artifact exists, so a missing one is a loud error rather than a
# pexpect spawn failure 20 minutes later.
set -uo pipefail
CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
IMG=${IMG:-$CHERI_ROOT/output/cheribsd-riscv64-purecap.img}
OUT=${1:?usage: make-qemu-argv.sh <out-argv-file>}
QEMU_MEM=${QEMU_MEM:-2048}

die() { echo "make-qemu-argv: $*" >&2; exit 1; }

# QEMU: prefer the x-form, accept either, and refuse ambiguity silently resolved.
QEMU=""
for c in "$SDK/bin/qemu-system-riscv64xcheri" "$SDK/bin/qemu-system-riscv64cheri"; do
  [ -x "$c" ] && { QEMU="$c"; break; }
done
[ -n "$QEMU" ] || die "no purecap QEMU in $SDK/bin (looked for qemu-system-riscv64{x,}cheri)"

# bbl purecap firmware. CHERI-QEMU's -bios default IS this file; stock OpenSBI is
# not CHERI-aware and the purecap kernel then produces NO console output at all,
# which looks like a hang and is actually the firmware.
FW=$(ls "$SDK"/share/qemu/bbl-riscv64*cheri*-virt-fw_jump.bin 2>/dev/null | head -1)
[ -n "$FW" ] || FW=$(ls "$SDK"/share/qemu/bbl-riscv64*fw_jump*.bin 2>/dev/null | head -1)
[ -n "$FW" ] && [ -f "$FW" ] || die "no bbl purecap firmware under $SDK/share/qemu"

KERNEL="$ROOTFS/boot/kernel/kernel"
[ -f "$KERNEL" ] || die "no kernel at $KERNEL"
[ -f "$IMG" ]    || die "no disk image at $IMG"

printf '%s\n' "$QEMU" \
  -M virt -m "$QEMU_MEM" -nographic -bios "$FW" \
  -kernel "$KERNEL" \
  -drive "if=none,file=$IMG,id=drv,format=raw" \
  -device virtio-blk-device,drive=drv \
  -append "vfs.root.mountfrom=ufs:/dev/vtbd0" > "$OUT"

echo "[*] qemu:     $QEMU"
echo "[*] firmware: $FW"
echo "[*] kernel:   $KERNEL"
echo "[*] image:    $IMG ($(du -h "$IMG" | cut -f1))"
echo "[*] argv ->   $OUT ($(wc -l < "$OUT") tokens)"
