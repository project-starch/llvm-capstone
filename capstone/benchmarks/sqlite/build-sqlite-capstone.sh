#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

REPO_ROOT=$CAPSTONE_REPO_ROOT
SQLITE_SRC_DIR=${SQLITE_SRC_DIR:-$(bash "$SCRIPT_DIR/fetch-sqlite.sh")}
OUT_DIR=${OUT_DIR:-$CAPSTONE_TMP_ROOT/sqlite-build}
OBJ_DIR=${OBJ_DIR:-$OUT_DIR/obj}
PATCHED_SQLITE=${PATCHED_SQLITE:-$OUT_DIR/sqlite3-capstone.c}
OUT_DOM=${OUT_DOM:-$OUT_DIR/sqlite_memory_capstone.dom}
DOMAIN_OPT_LEVEL=${DOMAIN_OPT_LEVEL:--O0}
SQLITE_OPT_LEVEL=${SQLITE_OPT_LEVEL:--O0}

CLANG=${CLANG:-$CAPSTONE_CLANG}
LD_LLD=${LD_LLD:-$CAPSTONE_LD_LLD}
LLVM_READOBJ=${LLVM_READOBJ:-$CAPSTONE_LLVM_READOBJ}
START_SRC=${START_SRC:-$REPO_ROOT/capstone/my_first_domain/start.S}
LINKER_SCRIPT=${LINKER_SCRIPT:-$REPO_ROOT/capstone/my_first_domain/link.ld}
ADAPTED_DIR="$SCRIPT_DIR/adapted"
VFS_SKELETON_DIR="$REPO_ROOT/capstone/tests/runtime-qemu/sqlite-vfs-skeleton"
BUILTINS_DIR="$REPO_ROOT/compiler-rt/lib/builtins"
BEEBS_STRING_SRC="$REPO_ROOT/capstone/benchmarks/beebs/adapted/beebs_freestanding_string.c"

for required in "$SQLITE_SRC_DIR/sqlite3.c" "$SQLITE_SRC_DIR/sqlite3.h" \
                "$START_SRC" "$LINKER_SCRIPT"; do
  [[ -f "$required" ]] || {
    echo "missing required SQLite build input: $required" >&2
    exit 1
  }
done

mkdir -p "$OUT_DIR" "$OBJ_DIR"

# Capstone pointers are 128-bit capabilities. The current Clang constant
# evaluator asserts on SQLite's traditional function-pointer sentinel
# ((sqlite3_destructor_type)-1). Use a unique real function as the transient
# sentinel instead; SQLite only compares this value by identity and never calls
# it. Keep the official amalgamation immutable in the temporary source cache.
sed \
  -e '/^#define SQLITE_TRANSIENT[[:space:]]/c\
static void sqlite3CapstoneTransient(void *p) { (void)p; }\
#define SQLITE_TRANSIENT sqlite3CapstoneTransient\
static const char *sqlite3CapstoneViewOrTable(int isView) {\
  if( isView ) return "view";\
  return "table";\
}\
static const char *sqlite3CapstoneBeforeOrAfter(int isBefore) {\
  if( isBefore ) return "BEFORE";\
  return "AFTER";\
}\
static const char *sqlite3CapstoneDeleteOrUpdate(int isDelete) {\
  if( isDelete ) return "DELETE";\
  return "UPDATE";\
}\
static const char *sqlite3CapstoneSpaceOrEmpty(int needsSpace) {\
  if( needsSpace ) return " ";\
  return "";\
}' \
  -e 's/sqlite3Atoi64(z, pResult, strlen(z), SQLITE_UTF8)/sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)/' \
  -e 's/isView?"view":"table"/sqlite3CapstoneViewOrTable(isView)/' \
  -e 's/(IsView(pTable)? "view" : "table")/sqlite3CapstoneViewOrTable(IsView(pTable))/' \
  -e 's/(tr_tm == TK_BEFORE)?"BEFORE":"AFTER"/sqlite3CapstoneBeforeOrAfter(tr_tm == TK_BEFORE)/' \
  -e 's/op==TK_DELETE ? "DELETE" : "UPDATE"/sqlite3CapstoneDeleteOrUpdate(op==TK_DELETE)/' \
  -e '/pBest->t.z\[pBest->t.n\].*? " " : ""/c\
            sqlite3CapstoneSpaceOrEmpty(pBest->t.z[pBest->t.n]==39)' \
  -e 's/SQLITE_INT_TO_PTR(iArg)/0/g' \
  -e 's/SQLITE_INT_TO_PTR(arg)/0/g' \
  -e 's/^  static FuncDef aBuiltinFunc\[\] = {$/  FuncDef capstoneBuiltinFunc[] = {/' \
  -e '/^SQLITE_PRIVATE const sqlite3_mem_methods \*sqlite3MemGetMemsys5(void){$/,/^}$/c\
SQLITE_PRIVATE const sqlite3_mem_methods *sqlite3MemGetMemsys5(void){\
  static sqlite3_mem_methods memsys5Methods;\
  static int capstoneInitialized;\
  if( !capstoneInitialized ){\
    memsys5Methods.xMalloc = memsys5Malloc;\
    memsys5Methods.xFree = memsys5Free;\
    memsys5Methods.xRealloc = memsys5Realloc;\
    memsys5Methods.xSize = memsys5Size;\
    memsys5Methods.xRoundup = memsys5Roundup;\
    memsys5Methods.xInit = memsys5Init;\
    memsys5Methods.xShutdown = memsys5Shutdown;\
    memsys5Methods.pAppData = 0;\
    capstoneInitialized = 1;\
  }\
  return &memsys5Methods;\
}' \
  -e '/^  sqlite3WindowFunctions();$/i\
  static FuncDef aBuiltinFunc[ArraySize(capstoneBuiltinFunc)];\
  for(int capstoneI=0; capstoneI<ArraySize(capstoneBuiltinFunc); capstoneI++){\
    aBuiltinFunc[capstoneI] = capstoneBuiltinFunc[capstoneI];\
  }\
  for(int capstoneI=0; capstoneI<ArraySize(aBuiltinFunc); capstoneI++){\
    int capstoneArg = 0;\
    const char *capstoneName = aBuiltinFunc[capstoneI].zName;\
    if( strcmp(capstoneName, "ltrim")==0 ) capstoneArg = 1;\
    else if( strcmp(capstoneName, "rtrim")==0 ) capstoneArg = 2;\
    else if( strcmp(capstoneName, "trim")==0 ) capstoneArg = 3;\
    else if( strcmp(capstoneName, "max")==0 ) capstoneArg = 1;\
    else if( strcmp(capstoneName, "unistr_quote")==0 ) capstoneArg = 1;\
    else if( strcmp(capstoneName, "unlikely")==0 ) capstoneArg = 99;\
    else if( strcmp(capstoneName, "likelihood")==0 ) capstoneArg = 99;\
    else if( strcmp(capstoneName, "likely")==0 ) capstoneArg = 99;\
    else if( strcmp(capstoneName, "iif")==0 ) capstoneArg = 5;\
    else if( strcmp(capstoneName, "if")==0 ) capstoneArg = 5;\
    if( capstoneArg ) aBuiltinFunc[capstoneI].pUserData = SQLITE_INT_TO_PTR(capstoneArg);\
  }' \
  -e 's/const char \*zDbTrig = isTemp ? db->aDb\[1\].zDbSName : zDb;/const char *zDbTrig; if( isTemp ) zDbTrig = db->aDb[1].zDbSName; else zDbTrig = zDb;/' \
  -e 's/bufpt = flag_zeropad ? "null" : "NaN";/if( flag_zeropad ) bufpt = "null"; else bufpt = "NaN";/' \
  -e 's/escarg = (xtype==etESCAPE_Q ? "NULL" : "(NULL)");/if( xtype==etESCAPE_Q ) escarg = "NULL"; else escarg = "(NULL)";/' \
  -e 's/#if GCC_VERSION>=4007000 || __has_extension(c_atomic)/#if SQLITE_THREADSAFE \&\& (GCC_VERSION>=4007000 || __has_extension(c_atomic))/' \
  -e '0,/^#define YYDYNSTACK 1$/s//#define YYDYNSTACK 0/' \
  -e 's/sqlite3VdbeMemSetNull(pMem-(u<p->nField));/if( u<p->nField ) sqlite3VdbeMemSetNull(pMem-1); else sqlite3VdbeMemSetNull(pMem);/' \
  -e 's/(int)(pReadr1 - pMerger->aReadr)/(int)((__builtin_capstone_cap_get_cursor(pReadr1) - __builtin_capstone_cap_get_cursor(pMerger->aReadr)) \/ sizeof(PmaReader))/' \
  -e 's/(int)(pReadr2 - pMerger->aReadr)/(int)((__builtin_capstone_cap_get_cursor(pReadr2) - __builtin_capstone_cap_get_cursor(pMerger->aReadr)) \/ sizeof(PmaReader))/' \
  "$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED_SQLITE"

grep -q '^#define SQLITE_TRANSIENT sqlite3CapstoneTransient$' "$PATCHED_SQLITE"
grep -q 'sqlite3CapstoneViewOrTable(IsView(pTable))' "$PATCHED_SQLITE"
grep -q 'sqlite3CapstoneViewOrTable(isView)' "$PATCHED_SQLITE"
grep -q 'sqlite3CapstoneBeforeOrAfter(tr_tm == TK_BEFORE)' "$PATCHED_SQLITE"
grep -q 'sqlite3CapstoneDeleteOrUpdate(op==TK_DELETE)' "$PATCHED_SQLITE"
grep -q 'sqlite3CapstoneSpaceOrEmpty(pBest->t.z\[pBest->t.n\]' "$PATCHED_SQLITE"
grep -q 'aBuiltinFunc\[capstoneI\].pUserData = SQLITE_INT_TO_PTR(capstoneArg)' "$PATCHED_SQLITE"
grep -q 'aBuiltinFunc\[capstoneI\] = capstoneBuiltinFunc\[capstoneI\]' "$PATCHED_SQLITE"
grep -q 'memsys5Methods.xInit = memsys5Init' "$PATCHED_SQLITE"
! grep -q 'SQLITE_INT_TO_PTR(iArg)' "$PATCHED_SQLITE"
! grep -q 'SQLITE_INT_TO_PTR(arg)' "$PATCHED_SQLITE"
grep -q 'const char \*zDbTrig; if( isTemp )' "$PATCHED_SQLITE"
grep -q 'sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)' "$PATCHED_SQLITE"
grep -q 'if( flag_zeropad ) bufpt = "null"; else bufpt = "NaN";' "$PATCHED_SQLITE"
grep -q 'if( xtype==etESCAPE_Q ) escarg = "NULL"; else escarg = "(NULL)";' "$PATCHED_SQLITE"
grep -q '#if SQLITE_THREADSAFE && (GCC_VERSION>=4007000 || __has_extension(c_atomic))' "$PATCHED_SQLITE"
grep -q '^#define YYDYNSTACK 0$' "$PATCHED_SQLITE"
grep -q 'if( u<p->nField ) sqlite3VdbeMemSetNull(pMem-1); else sqlite3VdbeMemSetNull(pMem);' "$PATCHED_SQLITE"
grep -q '__builtin_capstone_cap_get_cursor(pReadr1)' "$PATCHED_SQLITE"
grep -q '__builtin_capstone_cap_get_cursor(pReadr2)' "$PATCHED_SQLITE"

SQLITE_DEFINES=(
  -DNDEBUG
  -DSQLITE_OS_OTHER=1
  -DSQLITE_THREADSAFE=0
  -DSQLITE_DEFAULT_MEMSTATUS=0
  -DSQLITE_TEMP_STORE=3
  -DSQLITE_OMIT_LOAD_EXTENSION=1
  -DSQLITE_OMIT_LOCALTIME=1
  -DSQLITE_OMIT_MMAP=1
  -DSQLITE_OMIT_WAL=1
  -DSQLITE_OMIT_SHARED_CACHE=1
  -DSQLITE_OMIT_TEMPDB=1
  -DSQLITE_OMIT_AUTOINIT=1
  -DSQLITE_OMIT_COMPILEOPTION_DIAGS=1
  -DSQLITE_OMIT_FLOATING_POINT=1
  -DSQLITE_OMIT_UTF16=1
  -DSQLITE_OMIT_INCRBLOB=1
  -DSQLITE_OMIT_GET_TABLE=1
  -DSQLITE_OMIT_DEPRECATED=1
  -DSQLITE_OMIT_EXPLAIN=1
  -DSQLITE_OMIT_FOREIGN_KEY=1
  -DSQLITE_OMIT_JSON=1
  -DSQLITE_DQS=0
  -DSQLITE_UNTESTABLE=1
  -DSQLITE_ZERO_MALLOC=1
  -DSQLITE_ENABLE_MEMSYS5=1
  -DSQLITE_DEFAULT_LOOKASIDE=0,0
  -DYYSTACKDEPTH=1000
)

COMMON_FLAGS=(
  -target capstone64-unknown-elf
  -Xclang -target-feature
  -Xclang +m
  -ffreestanding
  -fno-builtin
  -fno-jump-tables
  -fno-optimize-sibling-calls
  -ffunction-sections
  -fdata-sections
  -include "$ADAPTED_DIR/capstone_sqlite_libc.h"
  -I"$ADAPTED_DIR"
  -I"$SCRIPT_DIR"
  -I"$VFS_SKELETON_DIR"
  -I"$SQLITE_SRC_DIR"
  "${SQLITE_DEFINES[@]}"
)

"$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
  -ffreestanding -O0 -c "$START_SRC" -o "$OBJ_DIR/start.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$SQLITE_OPT_LEVEL" \
  -Wno-pointer-to-int-cast -Wno-void-pointer-to-int-cast \
  -c "$PATCHED_SQLITE" -o "$OBJ_DIR/sqlite3.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" \
  -c "$ADAPTED_DIR/capstone_sqlite_libc.c" -o "$OBJ_DIR/libc.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" \
  -c "$BEEBS_STRING_SRC" -o "$OBJ_DIR/beebs_string.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" \
  -c "$VFS_SKELETON_DIR/capstone_sqlite_vfs.c" -o "$OBJ_DIR/sqlite_vfs.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" \
  -c "$ADAPTED_DIR/capstone_sqlite_os.c" -o "$OBJ_DIR/sqlite_os.o"

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" \
  -c "$SCRIPT_DIR/sqlite_capstone_domain.c" -o "$OBJ_DIR/domain.o"

BUILTIN_OBJECTS=()
for builtin in adddf3 comparedf2 fixdfdi fixdfsi fixunsdfdi fixunsdfsi \
               floatdidf floatsidf floatunsidf fp_mode muldf3 subdf3; do
  object="$OBJ_DIR/$builtin.o"
  "$CLANG" -target capstone64-unknown-elf \
    -Xclang -target-feature -Xclang +m \
    -ffreestanding -fno-builtin -O0 \
    -I"$BUILTINS_DIR" \
    -c "$BUILTINS_DIR/$builtin.c" -o "$object"
  BUILTIN_OBJECTS+=("$object")
done

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/sqlite3.o" \
  "$OBJ_DIR/libc.o" \
  "$OBJ_DIR/beebs_string.o" \
  "$OBJ_DIR/sqlite_vfs.o" \
  "$OBJ_DIR/sqlite_os.o" \
  "$OBJ_DIR/domain.o" \
  "${BUILTIN_OBJECTS[@]}"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null
echo "Built $OUT_DOM"
