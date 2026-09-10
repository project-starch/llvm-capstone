#!/usr/bin/env bash
# THE CONTROL FOR SQLITE_FEATURE_SET. Asserts that the switch actually reached the compiler.
#
# WHY THIS IS NOT AN SQL TEST. The first attempt was a .test fixture using EXPLAIN and the
# foreign-key pragma, on the assumption that SQLITE_OMIT_EXPLAIN and SQLITE_OMIT_FOREIGN_KEY remove
# them. It did not discriminate: both statements PASS under `deployed`. In an amalgamation build the
# parser tables are pre-generated, so an OMIT_* define cannot remove grammar -- it only compiles out
# C code guarded by #ifdef. That is worth knowing on its own (see the commit that added this file),
# and it means the SQL surface is the wrong place to look for the difference.
#
# The exported C API is the right place, and it separates the two readings the corpus cannot:
# identical SLT tallies for both feature sets are equally consistent with "the restored features are
# not exercised by these files" and with "the switch is inert". These six symbols decide it.
#
#   usage: check-feature-set.sh <deployed|restored> [binary]
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../../tests/capstone-test-env.sh"
WANT=${1:?usage: check-feature-set.sh <deployed|restored> [binary]}
BIN=${2:-${SLT_NATIVE_BIN:-$CAPSTONE_TMP_ROOT/slt-native/slt_native}}
[[ -x "$BIN" ]] || { echo "ERROR: $BIN missing -- run build-slt-native.sh" >&2; exit 1; }

# symbol -> the omission that removes it
declare -A GATED=(
  [sqlite3_get_table]=SQLITE_OMIT_GET_TABLE
  [sqlite3_blob_open]=SQLITE_OMIT_INCRBLOB
  [sqlite3_open16]=SQLITE_OMIT_UTF16
  [sqlite3_compileoption_get]=SQLITE_OMIT_COMPILEOPTION_DIAGS
  [sqlite3_compileoption_used]=SQLITE_OMIT_COMPILEOPTION_DIAGS
  [sqlite3_expired]=SQLITE_OMIT_DEPRECATED
)
case "$WANT" in
  deployed) expect=0 ;;
  restored) expect=1 ;;
  *) echo "ERROR: feature set must be 'deployed' or 'restored', got '$WANT'" >&2; exit 2 ;;
esac

fail=0
for sym in "${!GATED[@]}"; do
  got=$(nm "$BIN" 2>/dev/null | grep -c " $sym\$" || true)
  (( got > 1 )) && got=1
  if [[ "$got" != "$expect" ]]; then
    echo "FAIL $sym: expected $(( expect )) under '$WANT' (gated by ${GATED[$sym]}), found $got" >&2
    fail=1
  fi
done
if (( fail )); then
  echo "feature-set control FAILED for '$WANT' on $BIN" >&2
  exit 1
fi
echo "feature-set control PASSED for '$WANT': all ${#GATED[@]} gated symbols $([[ $expect == 1 ]] && echo present || echo absent)"
