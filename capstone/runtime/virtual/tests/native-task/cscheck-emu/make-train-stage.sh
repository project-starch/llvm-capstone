#!/bin/bash
# make-train-stage.sh <native train stage dir> <out dir>: the board stage for one native SPEC train run on this kit:
# the native smoke (make-stage.sh: the boot gate, whose pass lines are run-musl-io.py io's) plus the native train stage
# under train/ (capstone/runtime/virtual/tests/native-task/cscheck-emu/make-native-stage.py: SPEC's own train commands,
# each a direct exec of the native image, no launcher), run by spec.sh after a passing smoke (ctl.sh).
# The native stage is copied with cp -rL; its run.sh runs unchanged, as its QEMU twin (run-stage-twin.py) ran it.
main() {
  set -eu
  local KIT; KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local NS; NS=$(realpath "${1:?native train stage dir}"); local OUT; OUT=$(realpath -m "${2:?out dir}")
  [ -e "$OUT" ] && { echo "STOP: $OUT exists"; return 2; }
  [ -f "$NS/stage/run.sh" ] && [ -d "$NS/stage/bin/native" ] || { echo "STOP: $NS is not a native stage"; return 2; }
  bash "$KIT/make-stage.sh" "$OUT" > /dev/null
  cp -rL "$NS/stage" "$OUT/train"
  cp -L "$NS/plan.json" "$OUT/train/plan.json"
  # outputs in-band only up to 64 KiB, as every board stage: larger ones are checked by sha256 against the twin
  printf '#!/bin/sh\nexport SPEC_INBAND_MAX=65536\nsh /test-domains/vbatch/train/run.sh\n' > "$OUT/spec.sh"
  echo "stage $OUT: $(find "$OUT" -type f | wc -l) files, $(du -sb "$OUT" | cut -f1) bytes, gzip $(cd "$OUT" && tar cf - . | gzip -9 | wc -c)"
}
main "$@"; exit
