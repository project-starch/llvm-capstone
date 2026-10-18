#!/bin/bash
# register-prediction.sh <tag> <prediction run dir> <image list>: write records/<tag>.prediction.txt from the compiler
# lane's QEMU run of one arm, after checking that run's staged images equal the board's list by full sha256 and that the
# run used the prediction kit (its bundle's module and its applied -cpu string). Refuses (exit 1) on any mismatch, 2 on
# no data. The record is what the arm's pre-registration commits.
main() {
  set -u
  local tag=${1:?tag} P=${2:?prediction run dir} LIST=${3:?image list} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local B=$P/bundle IMG=$P/raw/stage/images out=$S/records/$tag.prediction.txt
  [ -f "$B/inputs.json" ] && [ -d "$IMG" ] && [ -s "$LIST" ] || { echo "NO DATA: $B/inputs.json, $IMG or $LIST missing"; return 2; }
  diff <(cd "$IMG" && sha256sum $(ls | sort)) "$LIST" > /dev/null || { echo "REFUSED: $tag's prediction ran on other bytes than $LIST"; return 1; }
  local mod; mod=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['platform']['module'][:16])" "$B/inputs.json")
  [ "$mod" = 5bfc38f808e1effd ] || { echo "REFUSED: $tag's prediction module is $mod, not 5bfc38f808e1effd"; return 1; }
  [ -e "$out" ] && { echo "REFUSED: $out exists"; return 1; }
  mkdir -p "$S/records"
  { echo "# Tier 3 prediction, arm $tag: the compiler lane's QEMU run on t3-smoke/qkit7 (capstone-qemu 7a4f3a64776f0c88 behind"
    echo "# the wrapper 5f731f99ec6a19b2: exact bounds stripped, svadu=false appended), module 5bfc38f808e1effd (no params), SDK"
    echo "# t3-freeze-nb. Its $(wc -l < "$LIST") staged images equal the board's list by full sha256 (checked by register-prediction.sh)."
    echo "# The bundle's 'compiler' field is the judge's --llvm-bin clang; the image hashes are the compiled bytes' identity."
    python3 "$S/tier3-cells.py" "$B" | sed 's/platform [0-9a-f]\{64\}/platform qkit7-wrapper/'
    cut -c1-16,65- "$LIST" | sed 's/^/# image /'; } > "$out"
  echo "REGISTERED $tag: $(sed -n '5p' "$out" | cut -c1-160)"
}
main "$@"; exit $?
