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
# The domain payload TU (the run_* driver). Default = the sqlite-memory smoke
# domain; the row3 matched-pair runner overrides it. DOMAIN_EXTRA_FLAGS lets a
# caller pass e.g. -DROW3_NO_REVOKE to build the control variant.
DOMAIN_SRC=${DOMAIN_SRC:-$SCRIPT_DIR/sqlite_capstone_domain.c}
DOMAIN_EXTRA_FLAGS=${DOMAIN_EXTRA_FLAGS:-}

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
#define SQLITE_TRANSIENT sqlite3CapstoneTransient' \
  -e 's/sqlite3Atoi64(z, pResult, strlen(z), SQLITE_UTF8)/sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)/' \
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
  -e 's/#if GCC_VERSION>=4007000 || __has_extension(c_atomic)/#if SQLITE_THREADSAFE \&\& (GCC_VERSION>=4007000 || __has_extension(c_atomic))/' \
  -e '0,/^#define YYDYNSTACK 1$/s//#define YYDYNSTACK 0/' \
  -e 's/^  char saveBuf\[PARSE_TAIL_SZ\];/  char saveBuf[PARSE_TAIL_SZ] __attribute__((aligned(16)));/' \
  -e 's/  nByte = SZ_VDBECURSOR(nField);/  nByte = (SZ_VDBECURSOR(nField)+15)\&~15;/' \
  -e 's/&pMem->z\[SZ_VDBECURSOR(nField)\]/\&pMem->z[(SZ_VDBECURSOR(nField)+15)\&~15]/' \
  "$SQLITE_SRC_DIR/sqlite3.c" > "$PATCHED_SQLITE"

grep -q '^#define SQLITE_TRANSIENT sqlite3CapstoneTransient$' "$PATCHED_SQLITE"
grep -q 'memsys5Methods.xInit = memsys5Init' "$PATCHED_SQLITE"
grep -q 'sqlite3Atoi64(zIn, pResult, strlen(zIn), SQLITE_UTF8)' "$PATCHED_SQLITE"
grep -q '#if SQLITE_THREADSAFE && (GCC_VERSION>=4007000 || __has_extension(c_atomic))' "$PATCHED_SQLITE"
grep -q '^#define YYDYNSTACK 0$' "$PATCHED_SQLITE"
# gap 6: sqlite3NestedParse saves the cap-bearing Parse tail through this buffer;
# 16-align it so memcpy's tag-preserving ldc/stc fast path applies (no byte copy).
grep -q 'char saveBuf\[PARSE_TAIL_SZ\] __attribute__((aligned(16)));' "$PATCHED_SQLITE"
# gap 8: allocateCursor embeds a cap-bearing BtCursor at SZ_VDBECURSOR(nField),
# which is only 8-aligned; 16-align its offset (and the allocation) so ldc/stc on
# the BtCursor's capability fields don't fault on unaligned cap access.
grep -q 'nByte = (SZ_VDBECURSOR(nField)+15)&~15;' "$PATCHED_SQLITE"
grep -q '&pMem->z\[(SZ_VDBECURSOR(nField)+15)&~15\]' "$PATCHED_SQLITE"

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

"$CLANG" "${COMMON_FLAGS[@]}" "$DOMAIN_OPT_LEVEL" $DOMAIN_EXTRA_FLAGS \
  -c "$DOMAIN_SRC" -o "$OBJ_DIR/domain.o"

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

# THE DOMAIN DECLARES WHAT IT NEEDS, because at 3.3 MB it no longer fits the rule
# that applies when it does not. Without .capstone_domreq the module sizes headroom
# as max(2 * code_len, 512 KiB), asks the buddy allocator for over 10 MB in one
# block, and the DOM_CREATE ioctl fails before a single instruction runs:
#
#   Loadable size = 3336736
#   SQ: obs=18446744073709551615     <- create_dom returned -1
#   create_dom failed
#
# domreq.S's own header names this exact case ("that 3x-on-the-image rule is what
# pushed an interpreter image past the buddy allocator's maximum order").
#
# SQLite's heap is the in-image sqlite_heap[] array driving memsys5, NOT dom_data,
# so the declared requirement is essentially the stack. SQLite recurses in the
# parser and the VDBE, and at -O0 with 16-byte pointers a frame is about twice its
# size on an ordinary target.
SQLITE_DOMAIN_STACK=${SQLITE_DOMAIN_STACK:-$((1024 * 1024))}
SQLITE_DOMAIN_DATA=${SQLITE_DOMAIN_DATA:-$SQLITE_DOMAIN_STACK}

_segs() { "$LLVM_READOBJ" --program-headers "$OUT_DOM" | grep -E 'Offset|VirtualAddress|FileSize|MemSize'; }

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/sqlite3.o" \
  "$OBJ_DIR/libc.o" \
  "$OBJ_DIR/beebs_string.o" \
  "$OBJ_DIR/sqlite_vfs.o" \
  "$OBJ_DIR/sqlite_os.o" \
  "$OBJ_DIR/domain.o" \
  "${BUILTIN_OBJECTS[@]}"
_before=$(_segs)

"$CLANG" -target capstone64-unknown-elf -ffreestanding \
  -DCAPSTONE_DOMREQ_DATA=$SQLITE_DOMAIN_DATA \
  -DCAPSTONE_DOMREQ_STACK=$SQLITE_DOMAIN_STACK \
  -c "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/domreq.S" -o "$OBJ_DIR/domreq.o"

"$LD_LLD" --gc-sections -T "$LINKER_SCRIPT" -o "$OUT_DOM" \
  "$OBJ_DIR/start.o" \
  "$OBJ_DIR/sqlite3.o" \
  "$OBJ_DIR/libc.o" \
  "$OBJ_DIR/beebs_string.o" \
  "$OBJ_DIR/sqlite_vfs.o" \
  "$OBJ_DIR/sqlite_os.o" \
  "$OBJ_DIR/domain.o" \
  "${BUILTIN_OBJECTS[@]}" \
  "$OBJ_DIR/domreq.o"

# The section is non-alloc, so NOTHING LOADED MAY MOVE. Verified, not asserted:
# four added instructions have flipped a passing run on this project before, and an
# image-perturbing diagnostic is how that was found.
if [[ "$(_segs)" != "$_before" ]]; then
  echo "domreq.S moved a loaded byte; the declaration must be non-alloc" >&2
  exit 2
fi
echo "declared dom_data $SQLITE_DOMAIN_DATA (stack $SQLITE_DOMAIN_STACK)"

"$LLVM_READOBJ" -h "$OUT_DOM" >/dev/null
echo "Built $OUT_DOM"
