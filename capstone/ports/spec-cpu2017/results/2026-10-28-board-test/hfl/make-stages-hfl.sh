#!/usr/bin/env bash
# make-stages-hfl.sh <SPEC17 tree> <out root> [train|test]: the board stages of the SPEC train (or test) set, one arm
# per stage, with the per-benchmark make-stage.py options of the campaign stages they replace (spec/stages-train,
# stages-trainT1 for train; stages-board for test). Only the tree, and so the images, differ.
#   train: perlbench --keep-lib perl-train-lib.txt; x264 --ldecod ldecod-rv64; xz --link-inputs (fw33 kit);
#          mcf, lbm, imagick, nab: no options (imagick and x264 keep their target-side imagevalidate).
#   test (the QEMU twin's check, both arms in one stage, as census-q34's): perlbench --strip --native-harness <natperl> --keep-lib perl-test-lib.txt; x264 --ldecod --no-target-validation;
#          the others no options.
# Needs CAPSTONE_WORK (the directory holding spec/ and cc-prmerged/) and CAPSTONE_SPEC_KIT (the kit's
# capstone/ports/spec-cpu2017). Exit status: the number of stages that failed.
main() {
  set -u
  local TREE=${1:?tree} OUT=${2:?out root} SIZE=${3:-train} W=${CAPSTONE_WORK:?} K=${CAPSTONE_SPEC_KIT:?}
  local S=$W/spec MS="python3 $K/board/make-stage.py" fails=0 arm b opts name
  mkdir -p "$OUT"
  local sets="capstone-spatial capstone-lifetimes"
  [ "$SIZE" = test ] && sets=capstone-spatial,capstone-lifetimes
  for arm in $sets; do
    for b in 500.perlbench_r 505.mcf_r 519.lbm_r 525.x264_r 538.imagick_r 544.nab_r 557.xz_r; do
      opts=(); name=$b-$SIZE-$arm; [ "$SIZE" = test ] && name=$b-$SIZE
      case $SIZE/$b in
        train/500.perlbench_r) opts=(--keep-lib "$S/perl-train-lib.txt");;
        train/525.x264_r) opts=(--ldecod "$S/ldecod-rv64");;
        train/557.xz_r) opts=(--link-inputs); name=$name-link;;
        test/500.perlbench_r) opts=(--strip "$W/cc-prmerged/bin/llvm-objcopy"
                                    --native-harness "$S/stages-board/500.perlbench_r-test-capstone-spatial/stage/natperl"
                                    --keep-lib "$S/perl-test-lib.txt");;
        test/525.x264_r) opts=(--ldecod "$S/ldecod-rv64" --no-target-validation);;
      esac
      if $MS --tree "$TREE" --bench "$b" --size "$SIZE" --arms "$arm" "${opts[@]}" --out "$OUT/$name" > "$OUT/$name.mk.log" 2>&1; then
        echo "STAGE $name $(tail -1 "$OUT/$name.mk.log")"
      else
        echo "STAGE-FAIL $name: $(tail -1 "$OUT/$name.mk.log")"; fails=$((fails + 1))
      fi
    done
  done
  return $fails
}
main "$@"; exit $?
