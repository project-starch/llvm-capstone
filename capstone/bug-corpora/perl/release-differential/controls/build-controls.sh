#!/usr/bin/env bash
# Link svhead-controls.c against one Perl build's own libperl.a, with that build's SDK and that
# build's C flags (Makefile.config's CFLAGS without -DPERL_CORE), as a domain image.
#
#   CAPSTONE_LLVM_BUILD_DIR=<toolchain> bash build-controls.sh PERLD_ROOT OUT.dom
#
# PERLD_ROOT is a virtual-profile build of ports/perl/musl/build-perl-domain.sh: PERLD_SUBLET=0
# for the release arm, PERLD_SUBLET=1 for the protected arm. The interpreter in the image is that
# build's libperl.a, byte for byte the archive its perl image was linked from.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$1 OUT=$2
S=$ROOT/src/perl-5.36.3
SDK=$ROOT/runtime-virtual/sdk
[[ -f $S/libperl.a && -x $SDK/capstone-cc ]] || { echo "no libperl.a or SDK under $ROOT" >&2; exit 2; }
CFLAGS=$(sed -n 's/^CFLAGS = //p' "$S/Makefile.config")
[[ -n $CFLAGS ]] || { echo "no CFLAGS in $S/Makefile.config" >&2; exit 2; }
CFLAGS=${CFLAGS//-DPERL_CORE/}
export CAPSTONE_SDK=$SDK PATH=$SDK:$CAPSTONE_LLVM_BUILD_DIR/bin:$PATH
O=$(dirname "$OUT")/$(basename "$OUT" .dom).o
# shellcheck disable=SC2086  # CFLAGS is a flag list
capstone-cc $CFLAGS -g -I"$S" -c "$HERE/svhead-controls.c" -o "$O"
capstone-cc -Wl,-E -o "$OUT" "$O" "$S/libperl.a" -lm
echo "cflags: $CFLAGS -g"
sha256sum "$OUT" "$S/libperl.a" "$S/perl" "$HERE/svhead-controls.c"
