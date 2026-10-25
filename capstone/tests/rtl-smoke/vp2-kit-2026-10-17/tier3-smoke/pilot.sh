#!/bin/bash
# The Tier 3 runner pilot (not a record): one case.c corpus arm through run-virtual-cases.py --board-kit end to end,
# into scratch only. Usage: pilot.sh <kit> <out root>. Ends by itself.
main() {
  set -u
  local KIT=${1:?board kit} R=${2:?out root}
  local G=/tmp/capstone/wt-gate SDK=/mnt/hdd/$USER/capstone/v9-kit/out/sdk LB=/tmp/capstone/vp2-prep/llvm-bin-v6
  [ -e "$R" ] && { echo "STOP: $R exists"; return 2; }
  mkdir -p "$R"
  cd "$G" || return 1
  python3 capstone/bug-corpora/tools/run-virtual-cases.py --corpus capstone/bug-corpora/memcached/plain-temporal-repros \
    --arm virtual-malloc --board-kit "$KIT" --sdk "$SDK" --llvm-bin "$LB" --out "$R/out" --raw "$R/raw" > "$R/run.log" 2>&1
  echo "run-virtual-cases rc=$?"
  tail -12 "$R/run.log"
  local st=$R/raw/stage
  echo "PAYLOAD stage $(du -sb "$st" | cut -f1) B raw, $(tar -C "$st" -cf - . | gzip -6 | wc -c) B gzip; firmware $(stat -c %s "$R/raw/guest/fw/fw_payload.bin" 2>/dev/null) B against the cap $(grep -o '^BOARD_FW_CAP=[0-9]*' "$KIT/board.env" | cut -d= -f2)"
  for f in "$st"/images/*.dom; do echo "IMAGE $(basename "$f") $(stat -c %s "$f") B, gzip $(gzip -6c "$f" | wc -c) B, sha256 $(sha256sum "$f" | cut -c1-16)"; done
}
main "$@"; exit $?
