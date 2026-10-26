#!/usr/bin/env bash
# Build a QEMU kernel with the options of PoisonCap's FPGA kernel, for their speedtest1 runs.
#
#   bash in-container.sh bash .../build-kernel-poison-nodebug.sh
#
# Their artifact ships two purecap POISON kernels: CHERI-PURECAP-QEMU-POISON (a debug kernel:
# WITNESS, INVARIANTS, DEADLKRES, which the platform boots) and CHERI-PURECAP-GFE-POISON-NODEBUG (the
# FPGA board's, std.nodebug). The GFE board's kernel does not boot on QEMU's virt machine, so this
# composes the FPGA kernel's options onto their QEMU board from their own files only:
#
#   include "CHERI-PURECAP-QEMU-NODEBUG"        their file: CHERI-PURECAP-QEMU + std.nodebug
#   nooptions CHERI_CAPREVOKE_STATS             }
#   nooptions CHERI_CAPREVOKE_FAST_COPYIN       } the three lines of CHERI-PURECAP-GFE-POISON-NODEBUG
#   options   CHERI_CAPREVOKE_POISON            }
#
# The config is a NEW file added to their sys/riscv/conf (cheribuild only finds configs there); no file
# of theirs is changed. Built kernel-only (--skip-world, --skip-install), so the rootfs and kernel the
# platform boots are untouched; the kernel is copied to $P/kernels/poison-nodebug/boot/kernel/kernel,
# the layout guest.py reads a kernel from.
set -euo pipefail
P=${POISONCAP_PLATFORM:?run through in-container.sh}
W=$P/work
NAME=CHERI-PURECAP-QEMU-POISON-NODEBUG
CONF=$W/source/cheribsd/sys/riscv/conf
for f in CHERI-PURECAP-QEMU-NODEBUG CHERI-PURECAP-GFE-POISON-NODEBUG; do
  [ -f "$CONF/$f" ] || { echo "missing their config $f" >&2; exit 2; }
done
printf '%s\n' "include \"CHERI-PURECAP-QEMU-NODEBUG\"" "ident $NAME" \
  "nooptions CHERI_CAPREVOKE_STATS" "nooptions CHERI_CAPREVOKE_FAST_COPYIN" \
  "options CHERI_CAPREVOKE_POISON" > "$CONF/$NAME"
# The three option lines must be exactly theirs.
diff <(grep -E '^(no)?options' "$CONF/CHERI-PURECAP-GFE-POISON-NODEBUG" | tr -s ' \t' ' ') \
     <(grep -E '^(no)?options' "$CONF/$NAME" | tr -s ' \t' ' ') > /dev/null \
  || { echo "option lines differ from their GFE POISON NODEBUG config" >&2; exit 3; }

OUT=$P/kernels/poison-nodebug
mkdir -p "$OUT/boot/kernel"
python3 "$P/cheribuild/cheribuild.py" cheribsd-riscv64-purecap --skip-update --skip-world --skip-install \
  --source-root "$W/source" --build-root "$W/build" --tools-root "$W" --output-root "$W/output" \
  --cheribsd/source-directory "$W/source/cheribsd" \
  --clang-path "$(command -v clang-18)" --clang++-path "$(command -v clang++-18)" \
  --cheribsd/toolchain custom --cheribsd/toolchain-path "$W/sdk" --make-jobs "${BUILD_JOBS:-64}" \
  --kernel-config "$NAME" --cheribsd/no-build-tests > "$OUT/cheribuild.log" 2>&1 \
  || { tail -30 "$OUT/cheribuild.log"; exit 4; }
K=$(find "$W/build" -path "*/sys/$NAME/kernel" -type f | head -1)
[ -n "$K" ] || { echo "no built kernel for $NAME under $W/build" >&2; exit 5; }
cp "$K" "$OUT/boot/kernel/kernel"
cp "$CONF/$NAME" "$OUT/$NAME.conf"
sha256sum "$OUT/boot/kernel/kernel" > "$OUT/SHA256SUMS"
echo "kernel $NAME: $(cut -c1-16 "$OUT/SHA256SUMS") from $K"
