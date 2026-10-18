#!/bin/bash
# Tier 3 on the board, run-virtual-cases.py arms one boot each, in the order given, into <out root> (not the corpus's
# results/). Usage: tier3-seq.sh <board kit> <out root> <sdk> <llvm-bin> <spec>...; a spec is <tag>|<corpus>|<arm>[|<extra>],
# <extra> being further runner arguments separated by single spaces (--cc-arg=..., --prebuilt ..., none containing a space).
# Each arm is pilot.sh's call (the runner builds, stages, bakes, boots and writes a verdicts-v1 bundle); then its payload
# and images are printed by hash. A run whose kit, SDK or llvm-bin is missing stops before any boot. A boot that does
# not complete stops the sequence: the next arm would boot behind a wedge only if the recipe failed to reset, and the
# first failure is the reading (board-run skill, the ordering rule). Ends by itself.
# The outer timeout is a backstop above the recipe's own driver timeout (the runner's 300 + 60 s per step, plus 1800):
# GNU timeout signals its whole process group, so the recipe, the driver and the watchdog go with it, and the bake's
# EXIT trap still restores the shared buildroot (tested: a grandchild bash's EXIT trap runs on that SIGTERM).
main() {
  set -u
  local KIT=${1:?board kit} R=${2:?out root} SDK=${3:?sdk} LB=${4:?llvm-bin}; shift 4
  local G=/tmp/capstone/wt-gate spec tag corpus arm extra d rc; local -a xa
  for f in "$KIT/board.env" "$KIT/board-run.sh" "$KIT/adapter/capstone-vexec" "$KIT/adapter/module/capstone_vm.ko" \
           "$SDK" "$LB/llvm-nm" "$LB/llvm-readelf"; do
    [ -e "$f" ] || { echo "STOP: $f is missing"; return 2; }
  done
  [ $# -gt 0 ] || { echo "STOP: no arms"; return 2; }
  mkdir -p "$R"
  echo "T3SEQ START $(date +%T) kit $KIT module $(sha256sum "$KIT/adapter/module/capstone_vm.ko" | cut -c1-16) params [$(grep -v '^#' "$KIT/adapter/module/params" 2>/dev/null | tr '\n' ' ')]"
  for spec in "$@"; do
    IFS='|' read -r tag corpus arm extra <<<"$spec"; read -r -a xa <<<"${extra:-}"
    d=$R/$tag
    [ -e "$d" ] && { echo "T3SEQ $tag SKIPPED: $d exists"; continue; }
    mkdir -p "$d"
    echo "T3SEQ $tag BEGIN $(date +%T) $corpus $arm ${extra:+(${#xa[@]} extra args)}"
    ( cd "$G" || exit 1
      timeout 10800 python3 capstone/bug-corpora/tools/run-virtual-cases.py --corpus "capstone/bug-corpora/$corpus" \
        --arm "$arm" --board-kit "$KIT" --sdk "$SDK" --llvm-bin "$LB" "${xa[@]}" --out "$d/out" --raw "$d/raw" ) > "$d/run.log" 2>&1
    rc=$?
    echo "T3SEQ $tag END $(date +%T) rc=$rc :: $(grep -a -o 'virtual boot: [A-Z a-z]*' "$d/run.log" | head -1) :: $(grep -a '^--- ' "$d/run.log" | tail -1)"
    local st=$d/raw/stage
    [ -d "$st" ] && echo "T3SEQ $tag PAYLOAD stage $(du -sb "$st" | cut -f1) B, $(tar -C "$st" -cf - . | gzip -6 | wc -c) B gzip; firmware $(stat -c %s "$d/raw/guest/fw/fw_payload.bin" 2>/dev/null) B"
    for f in "$st"/images/*.dom; do [ -f "$f" ] && echo "T3SEQ $tag IMAGE $(basename "$f") $(sha256sum "$f" | cut -c1-16)"; done
    grep -a -q 'virtual boot: completed' "$d/run.log" || { echo "T3SEQ STOP after $tag: the boot did not complete"; return 1; }
  done
  echo "T3SEQ DONE $(date +%T)"
}
main "$@"; exit $?
