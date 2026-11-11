#!/usr/bin/env bash
# The ResourceOwner defect corpus, on the host.
#
#   build-host.sh [case ...]        (default: every case directory)
#
# Each case links against PostgreSQL's REAL resource owner -- src/backend/
# utils/resowner/resowner.c from the pin, unmodified -- over its REAL memory
# managers.  The defect is in resowner.c's order of operations, so nothing
# about it is modelled.  What a case supplies is a resource kind, which is what
# every consumer of resowner.c supplies.
#
# Each case is run two ways:
#
#   defect    the subject. Must print its own verdict lines and exit 0; the
#             CHECKs in case.c are the oracle and do not need a sanitizer.
#   control   -DRO_NEGATIVE_CONTROL: the same sequence with only the second
#             free withheld. It must complete. If the control faults, a fault
#             in the defect run is not attributable.
#
# and, when ASan is available, a third:
#
#   asan      the defect under ASan. Here ASan IS a real oracle and not a
#             restatement: the object freed twice is a libc allocation, which
#             is precisely what ASan instruments. That is the whole point of
#             this corpus -- see README.md, "Why ASan counts here".
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CC=${CC:-cc}
WORK=${WORK:-${CAPSTONE_TMP_ROOT:-/tmp}/pg-resowner-repros}
PORT=$(cd -- "$HERE/../../../ports/postgres" && pwd)

# A configured PostgreSQL 17 tree: PG_SRC is the source, PG_BUILD the directory
# holding configure's generated headers (the same directory for an in-tree
# build). The files this corpus compiles -- the seven memory managers and
# resowner.c -- are byte-identical between REL_17_0 and REL_17_5, so either pin
# supplies them; check-sources.sh asserts that rather than assuming it.
# PG_PRISTINE holds the ten .c files this corpus COMPILES, unmodified from the
# pin; check-sources.sh gates it by sha256. It defaults to PG_SRC and is a
# separate variable because a tree that has been configured for a capability
# target carries port patches -- the memory-context port's inert
# `#if SIZEOF_VOID_P > 8` guard in aset.c is one -- and the subject of a corpus
# is compiled from the pin's bytes or the corpus measures something else.
PG_SRC=${PG_SRC:-}
PG_BUILD=${PG_BUILD:-$PG_SRC}
PG_PRISTINE=${PG_PRISTINE:-$PG_SRC}
if [ -z "$PG_SRC" ] || [ ! -f "$PG_BUILD/src/include/pg_config.h" ]; then
    echo "build-host: set PG_SRC (and PG_BUILD if out-of-tree) to a configured" >&2
    echo "  PostgreSQL 17 tree; need \$PG_BUILD/src/include/pg_config.h" >&2
    exit 75
fi

MMGR="aset mcxt generation slab bump alignedalloc memdebug"
INC="-I$PG_BUILD/src/include -I$PG_SRC/src/include -I$PG_SRC/src/backend -I$HERE/shared -I$PORT/port"

if ! bash "$HERE/check-sources.sh" "$PG_PRISTINE" > "$WORK-sources.log" 2>&1; then
    cat "$WORK-sources.log" >&2
    echo "build-host: refusing to build -- see check-sources.sh above" >&2
    exit 75
fi
sed 's/^/   /' "$WORK-sources.log"

mkdir -p "$WORK"

build_case() {
    local case_dir=$1 mode=$2
    local obj="$WORK/obj-$mode" extra=() objs=()
    case $mode in
      asan)    extra=(-fsanitize=address -fno-omit-frame-pointer) ;;
      control) extra=(-DRO_NEGATIVE_CONTROL) ;;
    esac
    mkdir -p "$obj"
    local f base
    for f in $MMGR; do
        $CC -c -O1 -g "${extra[@]}" $INC \
            "$PG_PRISTINE/src/backend/utils/mmgr/$f.c" -o "$obj/$f.o" || return 1
        objs+=("$obj/$f.o")
    done
    for f in src/backend/utils/resowner/resowner.c src/common/hashfn.c \
             src/port/qsort.c; do
        base=$(basename "$f" .c)
        $CC -c -O1 -g "${extra[@]}" $INC "$PG_PRISTINE/$f" -o "$obj/$base.o" || return 1
        objs+=("$obj/$base.o")
    done
    # pg_printf_host.c forwards PostgreSQL's redirected printf family to the
    # host's, as the mmgr corpus's host runner does. Upstream src/port/snprintf.c
    # is self-contained in a backend and faults in a program this small.
    for f in "$PORT/port/pg_printf_host.c" "$HERE/shared/pg_support.c" \
             "$HERE/shared/driver.c"; do
        base=$(basename "$f" .c)
        $CC -c -O1 -g "${extra[@]}" $INC "$f" -o "$obj/$base.o" || return 1
        objs+=("$obj/$base.o")
    done
    $CC -O1 -g "${extra[@]}" $INC "$case_dir/case.c" "${objs[@]}" \
        -o "$WORK/$(basename "$case_dir").$mode" || return 1
}

cases=("$@")
if [ ${#cases[@]} -eq 0 ]; then
    for d in "$HERE"/[0-9][0-9]_*/; do [ -f "$d/case.c" ] && cases+=("${d%/}"); done
fi

status=0
for case_dir in "${cases[@]}"; do
    name=$(basename "$case_dir")
    echo "=================================================================="
    echo "case: $name"
    echo

    for mode in defect control; do
        if ! build_case "$case_dir" "$mode" > "$WORK/$name.build-$mode.log" 2>&1; then
            echo "BUILD FAILED ($mode); see $WORK/$name.build-$mode.log" >&2
            exit 75
        fi
    done

    echo "-- control (-DRO_NEGATIVE_CONTROL): must complete"
    "$WORK/$name.control" > "$WORK/$name.control.log" 2>&1
    ctl=$?
    sed 's/^/   /' "$WORK/$name.control.log"
    if [ $ctl -ne 0 ]; then
        echo "   CONTROL DID NOT COMPLETE (exit $ctl): no verdict for this case" >&2
        status=75; continue
    fi
    echo "   control completed"
    echo

    echo "-- defect: the subject, no sanitizer"
    "$WORK/$name.defect" > "$WORK/$name.defect.log" 2>&1
    rc=$?
    sed 's/^/   /' "$WORK/$name.defect.log"
    #
    # TWO shapes count as reproduced, and the difference is the allocator's,
    # not the defect's:
    #   exit 0   -- the second free went through unnoticed and case.c's own
    #               CHECKs confirmed the item was released twice;
    #   killed   -- the host allocator noticed. glibc prints "double free
    #               detected in tcache 2" and raises SIGABRT at the second
    #               free, so case.c's CHECKs are never reached. That is a
    #               DETECTION, and reading it as a failed CHECK would throw
    #               away the strongest host evidence the corpus has.
    # What must not happen is a clean exit with one release, which would mean
    # the sequence did not reproduce.
    released_twice=no
    grep -q 'pass 2 on\|releases of .*: 2' "$WORK/$name.defect.log" && released_twice=yes
    alloc_caught=no
    grep -qi 'double free\|free(): invalid\|corrupted' "$WORK/$name.defect.log" && alloc_caught=yes
    if [ $rc -eq 0 ] && [ $released_twice = yes ]; then
        echo "   reproduced: released twice; the host allocator did not notice"
    elif [ $alloc_caught = yes ]; then
        echo "   reproduced: released twice and the host allocator CAUGHT the"
        echo "   second free (exit $rc) -- native-detect is a detection here"
    else
        echo "   NOT reproduced: exit $rc, released_twice=$released_twice," >&2
        echo "   allocator_diagnostic=$alloc_caught" >&2
        status=1; continue
    fi
    echo

    if build_case "$case_dir" asan > "$WORK/$name.build-asan.log" 2>&1; then
        echo "-- defect under ASan: the double free is of a libc object, so this"
        echo "   is an oracle and not a restatement"
        ASAN_OPTIONS=detect_leaks=0 "$WORK/$name.asan" \
            > "$WORK/$name.asan.log" 2>&1
        if grep -q 'double-free\|attempting double-free' "$WORK/$name.asan.log"; then
            echo "   ASan: attempting double-free -- REPORTED"
            grep -m3 'ERROR: AddressSanitizer\|#[0-9] .*case.c\|#[0-9] .*driver.c' \
                "$WORK/$name.asan.log" | sed 's/^/     /'
        else
            echo "   ASan did NOT report a double free; log: $WORK/$name.asan.log"
            status=1
        fi
    else
        echo "-- ASan build unavailable; SKIPPED (not a pass)"
    fi
    echo
done
exit $status
