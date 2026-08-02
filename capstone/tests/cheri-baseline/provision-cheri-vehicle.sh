#!/usr/bin/env bash
# Provision a bootable CHERI-RISC-V purecap vehicle for the corpus baselines,
# WITHOUT building CheriBSD world.
#
# This is the recipe that actually worked (2026-07-31), and it is VERIFIED
# from an empty CHERI_ROOT (2026-08-01): a clean run produces SDK clang-17,
# CHERI-QEMU, the bbl firmware, a 1.6 GB purecap image and the QEMU argv with
# no intervention, and the measurement run against it reproduces all 14 row
# verdicts identically.
#
# DO NOT "fix" this by building CheriBSD world instead. That was tried at
# length and is blocked on a modern Linux host for a reason no compiler flag
# can address: the C++ host tool usr.bin/dtc pulls in libstdc++ headers whose
# gthread machinery needs glibc's pthread types, while FreeBSD's own legacy
# include dir defines pthread_mutex_t as a POINTER, so glibc's brace-shaped
# PTHREAD_MUTEX_INITIALIZER fails with "excess elements in scalar
# initializer". Confirmed identically against libstdc++ 12, 14 and 15, and
# with both clang and g++. Downloading the world sidesteps all of it.
#
#   CHERI-QEMU              built from source (small, builds fine)
#   bbl purecap firmware    built from source (~5 seconds)
#   CheriBSD purecap world  DOWNLOADED as published distribution sets
#   disk image              assembled with the packaged NetBSD makefs
#
# Everything lands under $CHERI_ROOT; no sudo, nothing system-wide.
# `rm -rf $CHERI_ROOT` undoes it.
#
#   ./provision-cheri-vehicle.sh                 # build/refresh the vehicle
#   OVERLAY_SRC=<dir> ./provision-cheri-vehicle.sh   # stage <dir> into /root
#
# Output: $CHERI_ROOT/output/cheribsd-riscv64-purecap.img plus
#         $CHERI_ROOT/xlang-run/qemu-argv.txt for cheri-run.py.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CHERI_ROOT=${CHERI_ROOT:-$HOME/cheri}
CHERI_WS=${CHERI_WS:-$HOME/cheri-ws}
CHERIBUILD=${CHERIBUILD:-$CHERI_WS/cheribuild/cheribuild.py}
SDK=${SDK:-$CHERI_ROOT/output/sdk}
LOCAL_DEPS=${LOCAL_DEPS:-$CHERI_WS/local-deps}
ROOTFS=${ROOTFS:-$CHERI_ROOT/rootfs-purecap}
IMG=${IMG:-$CHERI_ROOT/output/cheribsd-riscv64-purecap.img}
RUNDIR=${RUNDIR:-$CHERI_ROOT/xlang-run}
JOBS=${JOBS:-$(( $(nproc) - 8 ))}; [ "$JOBS" -ge 1 ] || JOBS=1
# CheriBSD release to fetch. riscv64c == purecap.
REL=${REL:-26.07}
DIST_URL=${DIST_URL:-https://download.cheribsd.org/releases/riscv/riscv64c/$REL/ftp}
mkdir -p "$CHERI_ROOT" "$RUNDIR"

# --- host compiler wrappers ---------------------------------------------------
# Two independent lessons are baked in here, both learned by reading the actual
# failing command rather than guessing:
#  (a) SPECIFICITY: QEMU passes -Werror=incompatible-pointer-types-discards-
#      qualifiers. A blanket -Wno-error does NOT override a specific -Werror=;
#      only the matching specific -Wno-error= does.
#  (b) ORDER: clang resolves these left to right, so the negation must come
#      after the build's own -Werror. Passing them both before and after "$@"
#      covers builds that inject flags at either end.
# cheribuild keeps its OWN source/build/output roots and does NOT read
# CHERI_ROOT. Left implicit these default to ~/cheri, so a run with
# CHERI_ROOT elsewhere silently built into ~/cheri instead — the script was
# not relocatable. Pass the roots explicitly on every invocation.
CB_ROOTS=(--source-root "$CHERI_ROOT" --output-root "$CHERI_ROOT/output"
          --build-root "$CHERI_ROOT/build")

WRAP="$CHERI_ROOT/host-cc-wrappers"; mkdir -p "$WRAP"
W='-Wno-error -Wno-error=incompatible-pointer-types-discards-qualifiers'
for drv in clang clang++; do
  real=$(command -v /usr/bin/$drv) || continue
  printf '#!/bin/sh\nexec %s %s "$@" %s\n' "$real" "$W" "$W" > "$WRAP/$drv"
  chmod +x "$WRAP/$drv"
done

step() { echo "== $* =="; }

# --- 0. preflight -------------------------------------------------------------
# Fail here with a specific message rather than halfway through a multi-hour
# build. Every one of these was an actual stumble during bring-up.
preflight() {
  local missing=0 t
  for t in curl tar python3 git fakeroot dpkg-deb apt-get; do
    command -v "$t" >/dev/null || { echo "MISSING TOOL: $t"; missing=1; }
  done
  [ -x "$CHERIBUILD" ] || {
    echo "MISSING: $CHERIBUILD"
    echo "  git clone https://github.com/CTSRD-CHERI/cheribuild $CHERI_WS/cheribuild"
    missing=1; }
  # Disk: measured 26 GB for a full working tree, of which ~13 GB is the
  # CHERI-LLVM build and ~5.5 GB the installed SDK. 40 GB gives headroom.
  local avail
  avail=$(df -Pk "$(dirname "$CHERI_ROOT")" 2>/dev/null | awk 'NR==2{print int($4/1048576)}')
  if [ -n "$avail" ] && [ "$avail" -lt 40 ]; then
    echo "LOW DISK: ${avail} GiB free at $(dirname "$CHERI_ROOT"); want >= 40"
    missing=1
  fi
  [ "$missing" -eq 0 ] || { echo "preflight failed"; exit 2; }
  echo "[*] preflight ok (${avail:-?} GiB free, jobs=$JOBS)"
}
preflight

# --- 1. CHERI-LLVM ------------------------------------------------------------
# Needed by bbl (step 2) and by the shims (compiled purecap against this SDK).
# QEMU does NOT need it — it builds with the host compiler — so this step is
# skipped on a tree that already has an SDK. This is the long pole: ~13 GB of
# build tree and the better part of an hour at high parallelism.
if [ ! -x "$SDK/bin/clang" ]; then
  step "building CHERI-LLVM (long: this is the SDK bbl and the shims need)"
  # Target is llvm-native (the host-side CHERI toolchain). "cheri-llvm" is
  # NOT a target; cheribuild rejects it and suggests unrelated names.
  python3 "$CHERIBUILD" llvm-native -f --make-jobs "$JOBS" "${CB_ROOTS[@]}" \
    --clang-path "$WRAP/clang" --clang++-path "$WRAP/clang++" \
    || { echo "cheri-llvm build failed"; exit 1; }
fi

# --- 2. CHERI-QEMU ------------------------------------------------------------
if [ ! -x "$SDK/bin/qemu-system-riscv64cheri" ]; then
  step "building CHERI-QEMU"
  python3 "$CHERIBUILD" qemu -f --make-jobs "$JOBS" "${CB_ROOTS[@]}" \
    --clang-path "$WRAP/clang" --clang++-path "$WRAP/clang++" \
    || { echo "qemu build failed"; exit 1; }
fi

# --- 3. bbl purecap firmware --------------------------------------------------
# CHERI-QEMU's -bios default IS this file. Stock OpenSBI is not CHERI-aware:
# it loads and hands off to S-mode, and then the purecap kernel produces no
# console output at all. Symptom looks like a hang; cause is the firmware.
FW="$SDK/share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"
if [ ! -f "$FW" ]; then
  step "building bbl purecap firmware"
  python3 "$CHERIBUILD" bbl-baremetal-riscv64-purecap -f --make-jobs "$JOBS" "${CB_ROOTS[@]}" \
    --clang-path "$WRAP/clang" --clang++-path "$WRAP/clang++" \
    || { echo "bbl build failed"; exit 1; }
fi

# --- 4. CheriBSD purecap distribution sets ------------------------------------
# base.txz + kernel.txz replace the entire world build. Verified that this
# kernel exports the revocation knobs the three configs need:
#   runtime_revocation_default, runtime_revocation_every_free_default,
#   runtime_revocation_async
#
# EVERYTHING from here on runs under ONE fakeroot session (state file carried
# with -i/-s). Without it the tree is owned by the invoking user, and the
# image boots but login refuses:
#   init: _secure_path: /etc/login.conf is not owned by root
# and the console sits at "login:" forever. fakeroot makes the recorded
# ownership root:wheel, which is what makefs then writes into the image.
FR_STATE="$CHERI_ROOT/fakeroot.state"
command -v fakeroot >/dev/null || { echo "MISSING: fakeroot (apt-get download fakeroot)"; exit 2; }

if [ ! -d "$ROOTFS/boot/kernel" ]; then
  step "fetching CheriBSD $REL purecap distribution sets"
  mkdir -p "$CHERI_ROOT/dist" "$ROOTFS"
  for f in base.txz kernel.txz; do
    [ -f "$CHERI_ROOT/dist/$f" ] || curl -sL -o "$CHERI_ROOT/dist/$f" "$DIST_URL/$f" \
      || { echo "download failed: $DIST_URL/$f"; exit 1; }
  done
  ( cd "$ROOTFS" && fakeroot -s "$FR_STATE" -- sh -c \
      "tar -xf '$CHERI_ROOT/dist/base.txz' && tar -xf '$CHERI_ROOT/dist/kernel.txz'" )

  step "trimming rootfs"
  # Not just to save space: the packaged makefs fails writing large files
  # ("Writing inode ...: Invalid argument", first hit on the 6 MB zfs.ko).
  # Trimming 1.3 GB -> ~470 MB removes every such file we do not need.
  # Trim + configure inside the SAME fakeroot session, and force root:wheel.
  # The distribution sets ship no fstab and the kernel otherwise drops to
  # mountroot>; on QEMU virt + virtio-blk the root device is vtbd0.
  ( cd "$ROOTFS" && fakeroot -i "$FR_STATE" -s "$FR_STATE" -- sh -c '
      rm -rf rescue usr/tests usr/share/doc usr/share/man usr/share/examples \
             usr/share/locale usr/lib/debug 2>/dev/null
      find boot/kernel -name "*.ko" -delete 2>/dev/null
      # Delete static archives EXCEPT the ones needed to LINK purecap
      # binaries. Deleting libgcc.a broke every link with
      # "unable to find library -lgcc"; the rootfs doubles as our sysroot.
      find usr/lib -name "*.a" \
           ! -name "libgcc*.a" ! -name "libcompiler_rt*.a" \
           ! -name "libc_nonshared.a" -delete 2>/dev/null
      printf "/dev/vtbd0\t/\tufs\trw\t1\t1\n" > etc/fstab
      printf "autoboot_delay=\"0\"\nboot_verbose=\"NO\"\n" > boot/loader.conf
      chown -R 0:0 .
    ' )
  # root already has an EMPTY password in the stock sets, and ttyu0 is
  # onifconsole, so the pexpect driver can log in unattended.
fi

# --- 5. makefs ----------------------------------------------------------------
MAKEFS="$LOCAL_DEPS/usr/sbin/makefs"
if [ ! -x "$MAKEFS" ]; then
  step "fetching makefs (no sudo, into the local prefix)"
  mkdir -p "$LOCAL_DEPS" "$CHERI_ROOT/debs"
  ( cd "$CHERI_ROOT/debs" && apt-get download makefs \
    && for d in makefs*.deb; do dpkg-deb -x "$d" "$LOCAL_DEPS"; done ) >/dev/null 2>&1 \
    || { echo "could not fetch makefs"; exit 1; }
fi

# --- 6. stage the corpus overlay ---------------------------------------------
OVERLAY_SRC=${OVERLAY_SRC:-}
if [ -n "$OVERLAY_SRC" ] && [ -d "$OVERLAY_SRC" ]; then
  step "staging overlay $OVERLAY_SRC -> /root/$(basename "$OVERLAY_SRC")"
  dst="$ROOTFS/root/$(basename "$OVERLAY_SRC")"
  fakeroot -i "$FR_STATE" -s "$FR_STATE" -- sh -c \
    "mkdir -p '$dst' && cp -a '$OVERLAY_SRC'/. '$dst'/ && rm -f '$dst'/*.build.log && chown -R 0:0 '$dst'"
fi

# --- 7. disk image ------------------------------------------------------------
step "building UFS image"
# UFS **1**, deliberately. The packaged makefs is NetBSD-derived and writes a
# UFS2 superblock at 0x2000, but FreeBSD expects SBLOCK_UFS2 at 0x10000, so a
# version=2 image is rejected at mountroot with
#   "fs->fs_sblockloc (0x2000) != SBLOCK_UFS2 (0x10000)".
# UFS1 puts its superblock at 8192 on both, and CheriBSD mounts it fine.
# Size stays well under 2 GB: at 4 GB makefs hits a 32-bit lseek overflow.
rm -f "$IMG"
fakeroot -i "$FR_STATE" -- "$MAKEFS" -t ffs -B little -s "${IMG_SIZE:-1600m}" \
  -o version=1,bsize=32768,fsize=4096 "$IMG" "$ROOTFS" >"$CHERI_ROOT/makefs.log" 2>&1 \
  || { echo "makefs failed:"; tail -3 "$CHERI_ROOT/makefs.log"; exit 1; }

# --- 8. qemu argv for cheri-run.py -------------------------------------------
printf '%s\n' "$SDK/bin/qemu-system-riscv64cheri" \
  -M virt -m "${QEMU_MEM:-2048}" -nographic -bios "$FW" \
  -kernel "$ROOTFS/boot/kernel/kernel" \
  -drive "if=none,file=$IMG,id=drv,format=raw" \
  -device virtio-blk-device,drive=drv \
  -append "vfs.root.mountfrom=ufs:/dev/vtbd0" > "$RUNDIR/qemu-argv.txt"

step "vehicle ready"
echo "  image:     $IMG"
echo "  firmware:  $FW"
echo "  qemu argv: $RUNDIR/qemu-argv.txt"
echo "  boot+run:  python3 $SCRIPT_DIR/cheri-run.py $RUNDIR/qemu-argv.txt $RUNDIR/serial.log <guest-dir>"
