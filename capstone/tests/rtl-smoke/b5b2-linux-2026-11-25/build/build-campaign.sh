#!/usr/bin/env bash
# The campaign's SPEC test images: every qualification overlay (SP-B5TL1-<g>/ovl, byte-identical to its B5TL1 source)
# relinked on the final kernels with the fast-rdtime keep-tags firmware, the overlay unchanged.
#   CF-B5TL1-<g>: Full, TREE=fpga-final (seal b42c8c644)
#   CS-B5TL1-<g>: +Spatial, TREE=fpga-final-spatial (seal 14f2ef81a)
# Arms with other userland (Full + ports, LSRH, ...) use the same call on their own overlay directory.
# CAPSTONE_HDD and CAPSTONE_REPO_ROOT as make-payload-final.sh. Each image's out/ holds SHA256SUMS and
# overlay-files.sha256; compare the latter with the source overlay before use.
main() {
: "${CAPSTONE_HDD:?}" "${CAPSTONE_REPO_ROOT:?}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
S=${CAPSTONE_HDD}/trapk/fpga/spec
for g in g2i g3 g4mk g1b x264b; do
  for a in CF:fpga-final CS:fpga-final-spatial; do
    p=${a%%:*} tree=${a##*:}
    out=$S/$p-B5TL1-$g; mkdir -p "$out"
    TREE=$tree FW_EXTRA="FW_FAST_RDTIME=y" bash "$here/make-payload-final.sh" "$S/SP-B5TL1-$g/ovl" "$out/out" > "$out/make.log" 2>&1
    echo "$p $g rc=$? $(tail -1 "$out/make.log")"
  done
done
}
main "$@"; exit
