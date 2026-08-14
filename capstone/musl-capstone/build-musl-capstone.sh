#!/usr/bin/env bash
# Build libc-capstone.a from whatever of musl the compiler currently accepts.
#
# Deliberately builds the PARTIAL set rather than waiting for 100 %: the useful
# question is not "does all of musl compile" but "what does a given program
# actually pull in", and only a linkable archive can answer that. Undefined
# symbols at link time are the work list; see README.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/musl-capstone-build}
OBJ_DIR="$OUT_DIR/obj"
ARCHIVE=${ARCHIVE:-$OUT_DIR/libc-capstone.a}
AR=${CAPSTONE_LLVM_AR:-$CAPSTONE_LLVM_BIN/llvm-ar}

MUSL_SRC_DIR=$(bash "$SCRIPT_DIR/prepare-musl-capstone.sh" | tail -1)

rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

# The survey owns the flags and the file set; --objects makes it keep the
# output. It exits 1 on a regression, which must not stop the build here: a
# partial archive is the point. Only a harness error (2) is fatal.
set +e
python3 "$SCRIPT_DIR/survey-musl-capstone.py" "$MUSL_SRC_DIR" \
        --objects "$OBJ_DIR" > "$OUT_DIR/survey.txt"
survey_status=$?
set -e
if [[ $survey_status -ge 2 ]]; then
  cat "$OUT_DIR/survey.txt" >&2
  echo "survey could not measure; build aborted" >&2
  exit 2
fi

objects=("$OBJ_DIR"/*.o)
if [[ ${#objects[@]} -eq 0 || ! -e "${objects[0]}" ]]; then
  echo "no objects produced in $OBJ_DIR" >&2
  exit 2
fi

rm -f "$ARCHIVE"
"$AR" rcs "$ARCHIVE" "${objects[@]}"

grep -E '^(surveyed|compiled|failed)' "$OUT_DIR/survey.txt"
printf 'archived       %d objects -> %s\n' "${#objects[@]}" "$ARCHIVE"
