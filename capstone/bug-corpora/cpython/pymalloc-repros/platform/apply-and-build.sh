#!/usr/bin/env bash
# Apply the libc fix to a prepared PoisonCap platform and rebuild it.
#
# The published platform's libc poisons a block on free and never takes the
# poison off again, so the next allocation that lands in that block traps on its
# first write. Any program that frees and reallocates the same size class dies;
# this corpus dies in pym_lifetime_init, whose feature_present() does an
# asprintf. See README.md in this directory for the evidence.
#
# Without the fix the corpus can only run with libc revocation OFF. With it,
# both configurations run.
#
#   platform/apply-and-build.sh WORK            apply, rebuild, verify
#   platform/apply-and-build.sh WORK --revert    put the published libc back
#
# WORK is the prepared platform directory, e.g. /tmp/capstone/poisoncap-work.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
WORK=${1:?usage: apply-and-build.sh WORK [--revert]}
MODE=${2:-apply}
WORK=$(cd "$WORK" && pwd)

MRS="$WORK/source/cheribsd/lib/libc/stdlib/malloc/mrs/mrs.c"
LIBC="$WORK/output/rootfs-riscv64-purecap/lib/libc.so.7"
IMAGE="$WORK/output/cheribsd-riscv64-purecap.img"
SAVED="$WORK/published-platform"
PLATFORM="$REPO/capstone/ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/platform.sh"
PREPARE="$REPO/capstone/ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/prepare.py"

poison_counts() {
  "$WORK/sdk/bin/llvm-objdump" -d "$LIBC" 2>/dev/null |
    grep -oE '\b(cpoison|cclearpoison)\b' | sort | uniq -c | tr -s ' ' | tr '\n' ' '
}

if [ "$MODE" = "--revert" ]; then
  [ -d "$SAVED" ] || { echo "no saved platform under $SAVED" >&2; exit 2; }
  cp -f "$SAVED/libc.so.7" "$LIBC".new && chmod --reference="$LIBC" "$LIBC".new 2>/dev/null || true
  mv -f "$LIBC".new "$LIBC"
  cp -f "$SAVED/cheribsd-riscv64-purecap.img" "$IMAGE"
  cp -f "$SAVED/mrs.c" "$MRS"
  echo "reverted to the published platform; poison instructions: $(poison_counts)"
  exit 0
fi

# Patch the published state, not whatever is there: the provenance guard says
# which one we have, and a second apply would fail anyway.
echo "== verifying the prepared sources against the pinned artifact"
python3 "$PREPARE" "$WORK" --verify

echo "== saving the published libc and image"
mkdir -p "$SAVED"
cp -f "$LIBC" "$SAVED/libc.so.7"
cp -f "$IMAGE" "$SAVED/cheribsd-riscv64-purecap.img"
cp -f "$MRS" "$SAVED/mrs.c"
sha256sum "$SAVED/libc.so.7" "$SAVED/cheribsd-riscv64-purecap.img" > "$SAVED/SHA256SUMS"

echo "== applying $HERE/mrs-poison-retire.patch"
patch -p1 -d "$WORK/source/cheribsd" --forward < "$HERE/mrs-poison-retire.patch"

echo "== rebuilding libc and the image (about 90s on a warm tree)"
: "${CHERIBUILD:?set CHERIBUILD to the cheribuild.py entry point}"
bash "$PLATFORM" "$WORK" cheribsd
bash "$PLATFORM" "$WORK" image

# A rebuild that produced no cclearpoison did not apply the fix, whatever the
# build said. Make that an error rather than a silent pass.
counts=$(poison_counts)
echo "== poison instructions in the new libc: $counts"
case "$counts" in
*cclearpoison*) ;;
*) echo "the rebuilt libc has no cclearpoison: the fix did not take" >&2; exit 1;;
esac
sha256sum "$LIBC" "$IMAGE"
echo "patched platform ready; --revert puts the published one back"
