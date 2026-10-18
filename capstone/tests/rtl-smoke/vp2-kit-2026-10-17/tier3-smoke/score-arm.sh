#!/bin/bash
# score-arm.sh <tag> <prediction bundle> <board run dir> <seq out>: write records/<tag>.board.txt from one arm's board run:
# the bundle's cells, compare-pilot.py against the prediction (cell by cell, by image), the boot lines, and, when the boot
# did not complete, the wedge apertures; then copies the board bundle into its corpus's results/2026-10-18-vp2/<arm>/
# (never over an existing one). Prints the comparison's last line; exit = compare-pilot.py's (0 all same).
# KIT_DESC names another kit in the record's header; RESULTS_DIR files the bundle under another results/ directory (a
# different module's bundles must not share a directory with, or be refused by, the bkit7 ones).
main() {
  set -u
  local tag=${1:?tag} P=${2:?prediction bundle} D=${3:?board run dir} Q=${4:?seq out} S; S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  local out=$S/records/$tag.board.txt B=$D/out G
  G=$(dirname "$(find "$D" -name board.txt -path '*guest*' | head -1)")
  [ -f "$B/inputs.json" ] || { echo "NO DATA: $B/inputs.json missing"; return 2; }
  [ -e "$out" ] && { echo "REFUSED: $out exists"; return 1; }
  local rc
  { echo "# Tier 3 board reading, arm $tag: ${KIT_DESC:-kit bkit7 (module 3005ed52, pin off)}, SDK t3-freeze-nb. Result lines only."
    python3 "$S/tier3-cells.py" "$B"
    echo "## compare-pilot.py <prediction bundle> <board bundle>"
    python3 "$S/compare-pilot.py" "$P" "$B" | sed 's/^/  /'
    echo "## the boot"
    grep -a "T3SEQ $tag \|T3PG $tag " "$Q" | grep -v IMAGE | sed 's/^/  /'
    [ -f "$G/SEQ" ] && sed 's/^/  /' "$G/SEQ"
    if grep -a -q "wedge apertures =====" "$G/board.txt" 2>/dev/null; then
      echo "## the wedge apertures"
      awk '/wedge apertures =====/{f=1;next} f&&/^(sw=|commit pc|trap-log|rev_node)/{print "  "$0} f&&/^$/{exit}' "$G/board.txt"
    fi; } > "$out"
  python3 "$S/compare-pilot.py" "$P" "$B" > /dev/null; rc=$?
  # the board bundle into its corpus, beside the QEMU bundles (results/2026-10-18-vp2/<recorded arm>/), for the fold
  local corpus arm dst
  read -r corpus arm < <(python3 -c "import json,sys; j=json.load(open(sys.argv[1])); print(j['corpus'], j['arm'])" "$B/inputs.json")
  dst=$(cd "$S/../../../.." && pwd)/bug-corpora/$corpus/results/${RESULTS_DIR:-2026-10-18-vp2}/$arm
  if [ -n "$corpus" ] && [ -d "$(dirname "$(dirname "$dst")")" ] && [ ! -e "$dst" ]; then
    # a runner may record a build path under the account's scratch disk: the committed copy names it <mnt-hdd>, as the
    # corpora's older bundles name theirs <scratch> (precommit-scan blocks the account name)
    mkdir -p "$dst" && sed "s|/mnt/hdd/$USER/|<mnt-hdd>/|g; s|/home/$USER/|<home>/|g" "$B/inputs.json" > "$dst/inputs.json" \
      && sed "s|/mnt/hdd/$USER/|<mnt-hdd>/|g; s|/home/$USER/|<home>/|g" "$B/verdicts.jsonl" > "$dst/verdicts.jsonl" \
      && echo "$tag: bundle -> ${dst#*/capstone/}"
  fi
  echo "$tag: $(grep -a 'cells the same\|CELLS DIFFER\|NO DATA' "$out" | tail -1 | sed 's/^ *//') (rc $rc)"
  return $rc
}
main "$@"; exit $?
