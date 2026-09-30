#!/usr/bin/env bash
# Populate the two submodules the build needs. RUN THIS ON THE HOST, not in the container.
#
#   capstone/container/fetch-submodules.sh
#
# Why on the host: project-starch/capstone-qemu is PRIVATE, and this machine authenticates
# to GitHub with `credential.helper = store`, i.e. a token sitting in ~/.git-credentials.
# Mounting that token into the container would hand a GitHub credential to a container that
# then compiles and runs an entire Linux distribution's worth of third-party source
# (Buildroot). Fetching source is not building, so it stays outside.
#
# If you would rather do it in the container anyway, run.sh honours
# CAPSTONE_GIT_CREDS=1, which bind-mounts ~/.git-credentials read-only.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
cd "$ROOT"

QEMU_PIN=deb7d757565dd22f950c565579987e4864891ec9

# If a standalone capstone-qemu clone happens to sit next to llvm-capstone, borrow its
# objects instead of pulling ~500 MB again. --dissociate copies what is borrowed into the
# new repo and drops the alternates link afterwards, so the result does NOT depend on that
# directory continuing to exist.
SIBLING="$(dirname -- "$ROOT")/capstone-qemu"
REF=()
if [ -d "$SIBLING/.git" ] && git -C "$SIBLING" cat-file -e "$QEMU_PIN^{commit}" 2>/dev/null; then
  echo "==> borrowing objects from $SIBLING (then dissociating)"
  REF=(--reference "$SIBLING" --dissociate)
fi

# NON-recursive: capstone-qemu's 15 nested roms/* submodules are ~1 GB and a
# riscv64-softmmu build uses none of them.
echo "==> capstone/capstone-qemu (pin ${QEMU_PIN:0:10})"
git submodule update --init "${REF[@]}" capstone/capstone-qemu
have=$(git -C capstone/capstone-qemu rev-parse HEAD)
[ "$have" = "$QEMU_PIN" ] && echo "    at the recorded pin" || echo "    NOTE: at ${have:0:10}, not the pin"

# caplifive-buildroot: init its submodules BY NAME, not with --recursive.
#
# --recursive fails outright on the pinned commit d04bd83b. That tree records a gitlink at
# components/opensbi-qemu with NO matching stanza in .gitmodules, so git has no URL for it
# and aborts the whole recursion:
#
#   fatal: No url found for submodule path '.../components/opensbi-qemu' in .gitmodules
#
# It is an orphaned entry, not something we are skipping at our peril: configs/
# qemu_capstone_defconfig selects BR2_PACKAGE_OVERRIDE_FILE=local-qemu.mk, and that file
# points OPENSBI_OVERRIDE_SRCDIR at components/opensbi -- its own comment says "OpenSBI
# comes from the same components/opensbi as the board (platform/generic is the QEMU
# target)". Nothing reads components/opensbi-qemu.
#
# The four below are the ones qemu_capstone_defconfig actually needs:
#   buildroot                                 the upstream tree this BR2_EXTERNAL extends
#   components/linux                          LINUX_OVERRIDE_SRCDIR (kernel 6.1.26)
#   components/opensbi                        OPENSBI_OVERRIDE_SRCDIR (OpenSBI 1.2)
#   package/capstone-sbi-domain/capstone-sbi  BR2_PACKAGE_CAPSTONE_SBI_DOMAIN=y
echo "==> capstone/caplifive-buildroot"
git submodule update --init capstone/caplifive-buildroot
# --recursive HERE is both safe and necessary. Safe: naming the four paths keeps the
# recursion away from components/opensbi-qemu, the entry with no URL. Necessary:
# components/opensbi has its OWN nested submodule, lib/sbi/capstone-sbi, and without it
# OpenSBI dies late in the Buildroot run with
#   lib/sbi/sbi_capstone_init.S:1:10: fatal error: capstone-sbi/sbi_capstone.h
# -- roughly an hour in, after the entire cross toolchain has already been compiled.
( cd capstone/caplifive-buildroot
  git submodule update --init --recursive \
    buildroot \
    components/linux \
    components/opensbi \
    package/capstone-sbi-domain/capstone-sbi
)
for d in buildroot components/linux components/opensbi components/opensbi/lib/sbi/capstone-sbi \
         package/capstone-sbi-domain/capstone-sbi; do
  [ -n "$(ls -A "capstone/caplifive-buildroot/$d" 2>/dev/null)" ] ||
    { echo "!! caplifive-buildroot/$d is empty after init" >&2; exit 1; }
  echo "    ok  $d"
done

# capstone-c: the compiler that generates the OpenSBI monitor assembly. Required --
# caplifive-buildroot/Makefile runs `cargo run` in here for $(CAPSTONE_S_OUTPUT), and its
# check-cc target refuses to build without CAPSTONE_CC_PATH pointing at a real checkout.
echo "==> capstone/capstone-c"
git submodule update --init capstone/capstone-c
[ -d capstone/capstone-c/src ] || { echo "!! capstone/capstone-c is empty" >&2; exit 1; }
echo "    ok  $(git -C capstone/capstone-c rev-parse --short HEAD)"

echo
echo "==> done. Next: capstone/container/run.sh capstone/container/setup.sh llvm qemu buildroot"
