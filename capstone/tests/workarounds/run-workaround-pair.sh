#!/usr/bin/env bash
# Tier 2b: one matched pair for one workaround knob -- the corpus built with the knob
# OFF and ON, run on QEMU, and the two results compared.
#
#   run-workaround-pair.sh <W-id> <rv8|coremark|beebs|slt> <-O0|-O2>
#
# QEMU has no silicon defects, so a knob whose removal changes QEMU output is
# COMPILER-DEBT by construction; identical output makes it a silicon-debt candidate to
# be settled on the board.  The knob's off/on spellings come from CLASSIFICATION.tsv.
# Compiler flags reach every suite through a clang WRAPPER exported as CAPSTONE_CLANG
# (the build scripts take the compiler from the environment, not the flags), defines
# the same way, env knobs directly.
#
# THE PAIR MUST DIFFER: the two builds' .dom images are hashed and the run aborts with
# exit 2 if any suite produced identical images for OFF and ON -- a pair that tested
# nothing must not produce a verdict.  Results: one row per benchmark in
# results/<date>.tsv with verdicts from twins/compare-twins.py (labels off/on), plus a
# summary row naming the knob, the level, the image hashes and the QEMU binary.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../capstone-test-env.sh"
TWINS="$SCRIPT_DIR/../twins"

ID=${1:?usage: run-workaround-pair.sh <W-id> <suite> <level>}
SUITE=${2:?usage: run-workaround-pair.sh <W-id> <suite> <level>}
LEVEL=${3:?usage: run-workaround-pair.sh <W-id> <suite> <level>}
ROW=$(awk -F'\t' -v id="$ID" '$1 == id' "$SCRIPT_DIR/CLASSIFICATION.tsv")
[[ -n "$ROW" ]] || { echo "ERROR: no row $ID in CLASSIFICATION.tsv" >&2; exit 2; }
MECH=$(cut -f3 <<< "$ROW"); OFF=$(cut -f4 <<< "$ROW"); ON=$(cut -f5 <<< "$ROW")
case "$MECH" in mllvm|define|cflag|env) ;; *) echo "ERROR: $ID has mechanism '$MECH', which this runner cannot drive" >&2; exit 2 ;; esac

QEMU=$CAPSTONE_QEMU_BINARY
[[ -x "$QEMU" ]] || { echo "ERROR: QEMU binary $QEMU missing" >&2; exit 2; }
QID="qemu=$(sha256sum "$QEMU" | cut -c1-12)@$(date -r "$QEMU" +%Y-%m-%dT%H:%M)"
BASE="$CAPSTONE_TMP_ROOT/workarounds/$ID-$SUITE$LEVEL"
rm -rf "$BASE"; mkdir -p "$BASE"
REAL_CLANG=$CAPSTONE_CLANG
RESULTS=${WORKAROUND_RESULTS:-$SCRIPT_DIR/results/$(date +%Y-%m-%d).tsv}
mkdir -p "$(dirname "$RESULTS")"

run_arm() {  # off|on, flags-or-env
  # Two statements: under `set -u` a single `local a=$1 b="$x/$a"` expands $a before it
  # is assigned ("arm: unbound variable" -- the first real run, 2026-09-05).
  local arm=$1 spec=$2
  local out="$BASE/$arm"
  mkdir -p "$out"
  local wrapper="$out/clang"
  case "$MECH" in
    mllvm)
      printf '#!/usr/bin/env bash\nexec %q %s "$@"\n' "$REAL_CLANG" "$spec" > "$wrapper"; chmod +x "$wrapper"
      export CAPSTONE_CLANG="$wrapper" ;;
    define|cflag)
      # AFTER the script's own flags, so a -D of the same macro or a -f pair is decided here.
      printf '#!/usr/bin/env bash\nexec %q "$@" %s\n' "$REAL_CLANG" "$spec" > "$wrapper"; chmod +x "$wrapper"
      export CAPSTONE_CLANG="$wrapper" ;;
    env) export "${spec?}" ;;
  esac
  # A fresh share/build tree per arm, so the image hashes below are this arm's alone.
  local stamp="$out/.stamp"; : > "$stamp"
  case "$SUITE" in
    slt)  CAPSTONE_TMP_ROOT="$out/tmp" SLT_TWIN_RESULTS="$out/slt.tsv" bash "$TWINS/run-slt-twin.sh" "$LEVEL" > "$out/summary.txt" 2>&1 ;;
    *)    CAPSTONE_TMP_ROOT="$out/tmp" bash "$TWINS/run-twin-suite.sh" "$SUITE" "$LEVEL" > "$out/driver.txt" 2>&1
          cp -f "$out/tmp/twins/$SUITE$LEVEL/summary.txt" "$out/summary.txt" 2>/dev/null ;;
  esac
  find "$out/tmp" -name '*.dom' -newer "$stamp" -exec sha256sum {} \; | sed 's|  .*/|  |' | sort > "$out/doms.sha"
  export CAPSTONE_CLANG="$REAL_CLANG"
}

echo "== $ID ($SUITE $LEVEL) OFF: $OFF"
run_arm off "$OFF"
echo "== $ID ($SUITE $LEVEL) ON:  $ON"
run_arm on "$ON"

# THE PAIR MUST DIFFER.
if [[ ! -s "$BASE/off/doms.sha" || ! -s "$BASE/on/doms.sha" ]]; then
  echo "ERROR: an arm built no image (off: $(wc -l < "$BASE/off/doms.sha"), on: $(wc -l < "$BASE/on/doms.sha"))" >&2; exit 2
fi
if cmp -s "$BASE/off/doms.sha" "$BASE/on/doms.sha"; then
  echo "ERROR: OFF and ON produced identical images -- the knob did not reach the build; no verdict" >&2
  printf 'workaround\t%s\t%s\t%s\t%s\tIDENTICAL-IMAGES\t%s\n' "$(date +%F)" "$ID" "$SUITE" "$LEVEL" "$QID" >> "$RESULTS"
  exit 2
fi
NDIFF=$(diff "$BASE/off/doms.sha" "$BASE/on/doms.sha" | grep -c '^<')

if [[ "$SUITE" == slt ]]; then
  OFFV=$(cut -f2 "$BASE/off/slt.tsv" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
  ONV=$(cut -f2 "$BASE/on/slt.tsv" | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
  printf 'workaround\t%s\t%s\t%s\t%s\t%s\toff:[%s] on:[%s]\timages differing: %s\n' "$(date +%F)" "$ID" "$SUITE" "$LEVEL" "$QID" "$OFFV" "$ONV" "$NDIFF" >> "$RESULTS"
  echo "$ID $SUITE $LEVEL: off [$OFFV] on [$ONV]; $NDIFF image(s) differ; rows in $RESULTS"
  [[ "$OFFV" == "$ONV" ]]
else
  python3 "$TWINS/compare-twins.py" --suite "$SUITE" --a "$BASE/off/summary.txt" --b "$BASE/on/summary.txt" \
    --label-a off --label-b on --meta "$ID $LEVEL $QID images-differing=$NDIFF" --tsv "$RESULTS"
fi
