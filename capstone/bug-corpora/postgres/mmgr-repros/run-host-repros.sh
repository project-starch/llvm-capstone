#!/usr/bin/env bash
# The PostgreSQL memory-manager defect corpus, on the host.
#
#   run-host-repros.sh [case ...]        (default: every case directory)
#
# Each case is a driver linked against PostgreSQL's REAL memory manager -- the
# seven files of src/backend/utils/mmgr from the pinned release, unmodified --
# plus whatever consumer source the defect lives in. Nothing is modelled: the
# allocator under the driver is the subject.
#
# Every case is run three ways and all three must agree before a verdict prints:
#
#   control   a plain malloc/free/use in the same binary, under ASan. ASan MUST
#             report it. If it does not, the harness cannot see this class of
#             fault at all and the run exits 75 with NO verdict, because a
#             silent subject would then say nothing.
#   plain     the subject, no sanitizer. Must print its verdict.
#   asan      the subject, under ASan. Must print the SAME verdict and ASan must
#             stay SILENT -- that silence is the finding, and it is only
#             believable because the control fired.
#
# The silence is not a bug in ASan. pfree() does not call free(): the chunk goes
# on the context's size-class free list (aset.c:1139-1143) and comes back from
# the next palloc of that class (aset.c:1000-1013). No malloc-level event ever
# happens, so there is nothing for a malloc-level tool to see.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=$(cd -- "$HERE/../../../ports/postgres" && pwd)
PG_VERSION=${PG_VERSION:-17.0}
OUT=${OUT:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-host}
SRC=$OUT/postgresql-$PG_VERSION
WORK=${WORK:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-mmgr-repros}
CC=${CC:-cc}

# The manager, and the consumer sources the cases need. bitmapset.c is here
# because two cases are Bitmapset aliasing defects; it is upstream source,
# unmodified, and it brings hashfn.c and pg_bitutils.c with it.
MMGR="aset mcxt generation slab bump alignedalloc memdebug"
CONSUMERS="src/backend/nodes/bitmapset.c src/common/hashfn.c src/port/pg_bitutils.c"
# pg_bitutils.c resolves its AVX-512 paths against these two, which need their
# own flags; PostgreSQL's own build compiles them separately for the same reason.
AVX512="src/port/pg_popcount_avx512.c"
AVX512_CHOOSE="src/port/pg_popcount_avx512_choose.c"
INC="-I$SRC/src/include -I$SRC/src/backend -I$PORT/port -I$PORT"

if [ ! -f "$SRC/src/include/pg_config.h" ]; then
    echo "run-host-repros: no prepared PostgreSQL $PG_VERSION at $SRC" >&2
    echo "  run ports/postgres/build-mmgr-host.sh first" >&2
    exit 75
fi

mkdir -p "$WORK"

# Build one case, with or without ASan. Objects are per-mode so the two never mix.
build_case() {
    local case_dir=$1
    local mode=$2
    local tag=$mode
    local san=()
    [ "$mode" = asan ] && san=(-fsanitize=address -fno-omit-frame-pointer)
    local obj="$WORK/obj-$tag"
    mkdir -p "$obj"
    local objs=()
    local f
    for f in $MMGR; do
        [ -f "$obj/$f.o" ] || $CC -c -O1 -g "${san[@]}" $INC \
            "$SRC/src/backend/utils/mmgr/$f.c" -o "$obj/$f.o" || return 1
        objs+=("$obj/$f.o")
    done
    for f in $CONSUMERS "$PORT/port/pg_stubs.c" "$PORT/port/pg_printf_host.c"; do
        local base; base=$(basename "$f" .c)
        local p=$f; [ -f "$p" ] || p="$SRC/$f"
        [ -f "$obj/$base.o" ] || $CC -c -O1 -g "${san[@]}" $INC "$p" -o "$obj/$base.o" || return 1
        objs+=("$obj/$base.o")
    done
    [ -f "$obj/avx512.o" ] || $CC -c -O1 -g "${san[@]}" -mavx512vpopcntdq -mavx512bw $INC \
        "$SRC/$AVX512" -o "$obj/avx512.o" || return 1
    [ -f "$obj/avx512choose.o" ] || $CC -c -O1 -g "${san[@]}" -mxsave $INC \
        "$SRC/$AVX512_CHOOSE" -o "$obj/avx512choose.o" || return 1
    objs+=("$obj/avx512.o" "$obj/avx512choose.o")
    $CC -O1 -g "${san[@]}" $INC "$case_dir/before.c" "${objs[@]}" \
        -o "$WORK/$(basename "$case_dir").$tag" || return 1
}

status=0
cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
    for d in "$HERE"/*/; do [ -f "$d/before.c" ] && cases+=("${d%/}"); done
fi

for case_dir in "${cases[@]}"; do
    name=$(basename "$case_dir")
    want=$(cat "$case_dir/oracle" 2>/dev/null || echo "")
    echo "=================================================================="
    echo "case: $name"
    echo "oracle: $want"
    echo

    if ! build_case "$case_dir" plain > "$WORK/$name.build-plain.log" 2>&1; then
        echo "BUILD FAILED (plain); see $WORK/$name.build-plain.log" >&2; exit 75
    fi
    if ! build_case "$case_dir" asan > "$WORK/$name.build-asan.log" 2>&1; then
        echo "BUILD FAILED (asan); see $WORK/$name.build-asan.log" >&2; exit 75
    fi

    # --- control first: ASan must see a plain malloc use-after-free ----------
    ASAN_OPTIONS=detect_leaks=0 "$WORK/$name.asan" --control \
        > "$WORK/$name.control.log" 2>&1
    if ! grep -q "heap-use-after-free" "$WORK/$name.control.log"; then
        echo "CONTROL DID NOT FIRE: ASan reported no heap-use-after-free." >&2
        echo "  The harness cannot see this class, so a silent subject proves" >&2
        echo "  nothing. No verdict. See $WORK/$name.control.log" >&2
        exit 75
    fi
    echo "control: ASan reported heap-use-after-free -- the harness can see this class"
    echo

    # --- subject, no sanitizer ----------------------------------------------
    "$WORK/$name.plain" > "$WORK/$name.plain.log" 2>&1
    plain_rc=$?
    sed 's/^/  /' "$WORK/$name.plain.log"

    # --- subject, under ASan -------------------------------------------------
    ASAN_OPTIONS=detect_leaks=0 "$WORK/$name.asan" > "$WORK/$name.asan.log" 2>&1
    asan_rc=$?

    got=$(grep -o 'VERDICT: .*' "$WORK/$name.plain.log" | sed 's/VERDICT: //')
    got_asan=$(grep -o 'VERDICT: .*' "$WORK/$name.asan.log" | sed 's/VERDICT: //')
    asan_quiet=yes
    grep -qE "ERROR: AddressSanitizer|SUMMARY: AddressSanitizer" "$WORK/$name.asan.log" && asan_quiet=no

    echo
    echo "  plain          rc=$plain_rc verdict=${got:-<none>}"
    echo "  under ASan     rc=$asan_rc verdict=${got_asan:-<none>}  asan-silent=$asan_quiet"

    if [ "$plain_rc" = 0 ] && [ "$got" = "$want" ] && [ "$got_asan" = "$want" ] \
       && [ "$asan_quiet" = yes ]; then
        echo "  RESULT: REPRODUCED, and invisible to ASan"
    else
        echo "  RESULT: NOT AS RECORDED"
        status=1
    fi
    echo
done
exit $status
