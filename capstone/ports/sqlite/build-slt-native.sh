#!/usr/bin/env bash
# Build the NATIVE SQLLogicTest baseline binary.
#
# Same runner header, same SQLite amalgamation and -- critically -- THE SAME SEMANTIC
# BUILD CONFIGURATION as the domain. Produces the numbers every domain run is compared
# against; see slt/slt_native.c for why the comparison and not the absolute rate is the
# result.
#
# WHY THE DEFINES ARE HARVESTED AND NOT RETYPED. The first version of this script used a
# plain default build, and the first domain run of the negative control then reported five
# statement failures the native side did not have: `INSERT INTO t1 VALUES(3,'ccc',1.5)`
# died with `near ".": syntax error`. The cause was not capabilities -- the domain build
# carries -DSQLITE_OMIT_FLOATING_POINT=1, so its tokenizer does not accept a decimal point
# at all. A baseline built without that flag makes an ordinary configuration difference
# look exactly like a capability defect, which is the one failure this whole comparison
# exists to avoid. So the list is read from the same file the domain build reads it from.
#
# NOTE FOR ANYONE READING THE PROPOSAL DOC: the claim that "the shipped build carries no
# SQLITE_OMIT_* flags at all" was wrong. It was based on SILICON_TRIM being gated off at
# build-sqlite-silicon.sh:911, and missed that SQLITE_DEFINES in build-sqlite-capstone.sh
# is a SECOND, always-active list carrying seventeen more of them.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/slt-native}
mkdir -p "$OUT_DIR"

# The SAME amalgamation the domain gets: fetch-sqlite.sh verifies its SHA3 and prints the
# directory. Using a system libsqlite3 instead would silently baseline a different engine.
SQLITE_SRC_DIR=$(bash "$SCRIPT_DIR/fetch-sqlite.sh")
echo "== amalgamation: $SQLITE_SRC_DIR"

# The domain's define list, minus the ones that select an OS/allocator rather than SQL
# semantics. Each exclusion is here because the native build has no capability domain to
# provide the facility, NOT because it changes what SQL means:
#   SQLITE_OS_OTHER          the domain supplies its own VFS; natively we want the real one
#   SQLITE_OMIT_AUTOINIT     the domain calls sqlite3_initialize itself from its entry glue
#   SQLITE_ZERO_MALLOC       there is no system allocator in the domain; natively there is
#   SQLITE_ENABLE_MEMSYS5    the domain's 256 KiB static arena; natively irrelevant
#   SQLITE_DEFAULT_LOOKASIDE an allocator tuning knob, not a semantic one
#   SQLITE_UNTESTABLE        removes test hooks only
# Everything else -- above all SQLITE_OMIT_FLOATING_POINT, SQLITE_DQS=0, OMIT_JSON,
# OMIT_FOREIGN_KEY, OMIT_UTF16 -- DOES change what the engine accepts and is kept.
EXCLUDE='SQLITE_OS_OTHER|SQLITE_OMIT_AUTOINIT|SQLITE_ZERO_MALLOC|SQLITE_ENABLE_MEMSYS5|SQLITE_DEFAULT_LOOKASIDE|SQLITE_UNTESTABLE'
mapfile -t DEFS < <(sed -n '/^SQLITE_DEFINES=(/,/^)/p' "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                    | grep -oE '\-D[A-Za-z0-9_]+(=[^ )]*)?' \
                    | grep -vE "^-D($EXCLUDE)")
(( ${#DEFS[@]} > 10 )) || { echo "ERROR: harvested only ${#DEFS[@]} defines -- the list moved" >&2; exit 1; }
echo "== semantic defines shared with the domain: ${#DEFS[@]}"
printf '   %s\n' "${DEFS[@]}" | head -30

# ONE UPSTREAM BUG HAS TO BE PATCHED TO USE OMIT_FLOATING_POINT AT ALL, and the domain
# build already patches it -- build-sqlite-capstone.sh:66, verified at :138. In 3.53.3 the
# #else arm of sqlite3AtoF refers to `z`, which is declared only inside the #ifndef arm, so
# -DSQLITE_OMIT_FLOATING_POINT does not compile as shipped. Applying the SAME one-line fix
# here rather than reusing the domain's fully patched amalgamation keeps this build's only
# deviation from pristine identical to the domain's, with none of its capability-specific
# rewrites.
PATCHED="$OUT_DIR/sqlite3-slt-native.c"
sed -e 's/sqlite3Atoi64(z, pResult, strlen(z), SQLITE_UTF8)/sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)/' \
    "$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED"
# The gate, not a hope: a sed that silently matched nothing would leave a build that fails
# in a way that looks like a toolchain problem.
grep -q 'sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)' "$PATCHED" \
  || { echo "ERROR: the sqlite3AtoF fix did not apply -- upstream moved" >&2; exit 1; }

cc -O1 -o "$OUT_DIR/slt_native" \
  -I"$SQLITE_SRC_DIR" -I"$SCRIPT_DIR/slt" \
  "${DEFS[@]}" \
  "$SCRIPT_DIR/slt/slt_native.c" "$PATCHED" \
  -lm

echo "Built $OUT_DIR/slt_native"
