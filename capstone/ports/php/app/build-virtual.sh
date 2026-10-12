#!/usr/bin/env bash
# Build the PHP port for the VIRTUAL Capstone profile -- a capability application that runs as
# an ordinary Linux user process, not a physical domain.
#
#   capstone/container/run.sh capstone/ports/php/app/build-virtual.sh [probe|php]
#
# WHY THIS IS A NEW SCRIPT AND NOT A FLAG ON build-engine.sh. The physical build is a hand-rolled
# clang/ld.lld invocation against my_first_domain/link.ld, with -ffreestanding, -nostdlibinc and
# a 29-header stubinc farm standing in for libc. Under the virtual profile the SDK owns the
# triple, the ISA features, the headers and the link script -- ports/common/cmake/toolchains/
# capstone-application.cmake says so explicitly, and warns that restating any of them creates a
# second source of truth. Nothing of the physical command line survives, and the loader's
# .capstone_virtual ELF marker refuses an image built the other way, so there is no shared path
# to flag between them. build-engine.sh stays exactly as it is: it is the comparison column.
#
# JOBS IS DEFAULTED DOWN, DELIBERATELY. runtime/virtual/build-sdk.sh defaults to JOBS=16 for
# musl's 1361 translation units. On a host that is also running something else -- and this one
# has no swap -- 16 is how you OOM the machine and take the other work with it. 4 is the default
# here; raise it with JOBS= when the machine is idle.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh"
R=$CAPSTONE_REPO_ROOT
WHAT=${1:-probe}

OUT=${PHP_VIRTUAL_OUT:-$CAPSTONE_TMP_ROOT/virtual-runtime}
SDK="$OUT/sdk"
CC="$SDK/capstone-cc"
export MUSL_CACHE_ROOT=${MUSL_CACHE_ROOT:-$OUT/musl-source}

# ---- 1. the musl tree ------------------------------------------------------------------
# prepare-musl-capstone.sh is idempotent: a patch that already applies in reverse is skipped,
# and one that fits neither way is an error rather than a silently partial tree.
MUSL=$(bash "$R/capstone/ports/musl-capstone/prepare-musl-capstone.sh" 2>/dev/null | tail -1)
[ -d "$MUSL/arch/capstone64" ] || { echo "musl not prepared (got '$MUSL')" >&2; exit 2; }
echo "  musl: $MUSL"

# ---- 2. the SDK -----------------------------------------------------------------------
# Rebuilt only when absent or when PHP_VIRTUAL_FRESH_SDK=1. The archive is 1361 objects and
# nothing in this port changes it, so rebuilding per run is pure cost.
if [ ! -x "$CC" ] || [ "${PHP_VIRTUAL_FRESH_SDK:-0}" = 1 ]; then
  echo "  building the virtual SDK (JOBS=${JOBS:-4}) ..."
  JOBS=${JOBS:-4} bash "$R/capstone/runtime/virtual/build-sdk.sh" "$OUT" "$MUSL" \
      ${PHP_VIRTUAL_SDK_ARGS:-} || { echo "SDK build FAILED" >&2; exit 1; }
else
  echo "  SDK: reusing $SDK (PHP_VIRTUAL_FRESH_SDK=1 to rebuild)"
fi
[ -x "$CC" ] || { echo "no capstone-cc at $CC" >&2; exit 1; }

# The SDK records what it is. An image built against the physical profile is rejected by the
# launcher with ENOEXEC rather than misbehaving, but the error arrives after a boot -- cheaper
# to assert it here.
# sdk.json carries no "profile" key, so key on the two things that actually differ:
# -capstone-image-gp in codegen (mandatory for virtual, absent for physical) and a libc
# archive named ...-virtual.a. Checked here because the alternative is a boot and ENOEXEC.
if ! grep -q 'capstone-image-gp' "$SDK/sdk.json" || ! grep -q 'libc-capstone-virtual' "$SDK/sdk.json"; then
  echo "SDK at $SDK is NOT the virtual profile:" >&2
  grep -E '"codegen"|"libc"' "$SDK/sdk.json" >&2
  echo "  (delete $SDK and re-run, or check CAPSTONE_APPLICATION_PROFILE)" >&2
  exit 1
fi

# ---- 3. what to build -----------------------------------------------------------------
case "$WHAT" in
  probe)
    # The seam probe: can a lent block be split so that retiring the payload leaves PHP's
    # header live? Everything in the allocator plan depends on the answer.
    SRC="$R/capstone/ports/php/zend-alloc/sublet/probe-sublet-split.c"
    IMG="$OUT/probe-sublet-split.dom"
    echo "  compiling the seam probe ..."
    # -I $R/capstone so <sublet/sublet.h> resolves to capstone/sublet/sublet.h, which is the
    # header the virtual heap itself uses (runtime/virtual/vm.h includes ../../sublet/sublet.h).
    "$CC" -O1 -I"$R/capstone" "$SRC" -o "$IMG" || { echo "probe build FAILED" >&2; exit 1; }
    echo "  built $IMG"
    "$CAPSTONE_LLVM_BIN/llvm-readobj" --sections "$IMG" 2>/dev/null \
      | awk '/Name: .capstone_virtual/{f=1} f&&/Size:/{print "    .capstone_virtual size: " $2; exit}'
    "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$IMG" 2>/dev/null \
      | awk '$2==".text"||$2==".data"||$2==".bss"{printf "    %-10s %d\n",$2,strtonum("0x" $3)}'
    ;;
  hier-probe)
    # Does the HIERARCHICAL wrapper still need zend_cap_widen here? Phase A reuses
    # libc/php_capstone_zend_hier.c unchanged, and that file's widen reads the PHYSICAL
    # arena's slot table, which does not exist on this profile. See the probe's own header.
    IMG="$OUT/probe-hier-widen.dom"
    echo "  compiling the hier-widen probe ..."
    # php_virtual_lower.c is linked in, not re-implemented: stages 3 and 4 test the PORT's own
    # zend_cap_widen, so a probe with its own copy would pass while the engine still broke.
    "$CC" -O1 "$HERE/probe-hier-widen.c" "$HERE/php_virtual_lower.c" -o "$IMG" \
      || { echo "hier probe build FAILED" >&2; exit 1; }
    echo "  built $IMG"
    ;;
  php)
    # ---- the engine itself, as a hosted process ----------------------------------------
    #
    # WHAT CHANGES RELATIVE TO engine/build-engine.sh, and nothing else does: no -target, no
    # +m, no -ffreestanding, no -nostdlibinc, no -isystem stubinc, no link script, no
    # domreq.S, no compiler-rt list, no start.S, no setjmp.S. capstone-cc supplies every one
    # of those (it passes -ffreestanding -fno-builtin -fno-jump-tables -nostdinc itself and
    # points -isystem at the Capstone musl), and ports/common/cmake/toolchains/
    # capstone-application.cmake is explicit that restating any of them is a second source of
    # truth. The 29-header stubinc farm and five of the six libc/ files are not ported -- musl
    # replaces them -- and engine/ keeps them, because it is still the comparison column.
    P=${PHP_SRC:-/corpus/build/php-5.0.0}
    [ -d "$P/Zend" ] || { echo "PHP source not at $P -- corpus not mounted?" >&2; exit 2; }
    E="$R/capstone/ports/php/engine"
    # Per-arm object and log directories. One shared obj/ would have the second build
    # overwrite the first's objects, and a pair assembled from whichever came last is not a
    # pair -- the two images would be byte-identical and the suite's own gate would refuse them.
    case " ${EXTRA_CF:-} " in
      *" -DZEND_CAP_BOUNDS_REAL_SIZE "*) _arm=control ;;
      *) _arm=fault ;;
    esac
    OBJ="$OUT/php-virtual/$_arm/obj"; LOG="$OUT/php-virtual/$_arm/log"
    mkdir -p "$OBJ" "$LOG"

    MODE=${PHP_CAP_ALLOC_MODE:-hier}
    if [ "$MODE" != hier ]; then
      # seam mode is the physical port's arena allocator serving emalloc directly, and its
      # arena, slot table and revocation are physical-profile code. Replacing it is the
      # allocator-seam work the plan treats as its own design question -- not Phase A.
      echo "  PHP_CAP_ALLOC_MODE=$MODE is not available on the virtual profile yet (hier only)." >&2
      echo "  The temporal axis arrives with the Sublet seam; see the plan's allocator section." >&2
      exit 2
    fi

    CF=(-O0 -std=gnu89 -fno-common -gline-tables-only
        -I"$P" -I"$P/Zend" -I"$P/main" -I"$P/TSRM" -I"$P/ext" -I"$P/ext/standard")

    # SHRINK-STACK: off by default here, and that is about ISOLATION, not about fit. The
    # physical arm forces it off because narrowing costs ~33% of .text and the image then
    # exceeds the 4 MiB domain ceiling. That ceiling is gone on this profile -- which is one
    # of the reasons for the retarget, since stack-object bounds are what make CRASH-100
    # (stack-use-after-return) catchable at all. But Phase A's whole job is that any
    # difference from the physical arm is attributable to the PROFILE, so the flag stays where
    # the physical arm has it and becomes its own variable afterwards.
    if [ "${PHP_VIRTUAL_SHRINK_STACK:-0}" = 1 ]; then
      echo "  shrink-stack: ON (stack objects bounded; NOT the Phase A comparison build)"
    else
      CF+=(-mllvm -capstone-shrink-stack=false)
    fi
    # shellcheck disable=SC2206
    [ -n "${EXTRA_CF:-}" ] && CF+=(${EXTRA_CF})

    # THE ZEND TU LIST IS READ OUT OF build-engine.sh, not copied. Two lists that must agree
    # and sit in two files is how a port quietly stops being the same program as its
    # comparison column; this way the physical build is the single source of truth and a TU
    # added there appears here automatically.
    TUS=$(sed -n '/^TUS="/,/"$/p' "$E/build-engine.sh" | sed 's/^TUS="//; s/"$//')
    for required in zend_compile zend_execute zend_hash zend_operators; do
      case " $TUS " in *" $required "*) ;; *)
        echo "  the TU list read from build-engine.sh has no '$required' -- its format changed" >&2
        exit 2 ;;
      esac
    done
    # zend_alloc is appended by the MODE case below, there as here, so finding it already in
    # the list means build-engine.sh moved it and this script would compile it twice.
    case " $TUS " in *" zend_alloc "*)
      echo "  build-engine.sh now carries zend_alloc in TUS itself -- drop the append below" >&2
      exit 2 ;;
    esac
    TU_COUNT=$(printf '%s\n' $TUS | wc -l)
    [ "$TU_COUNT" -ge 30 ] || { echo "  only $TU_COUNT Zend TUs read from build-engine.sh" >&2; exit 2; }
    # hier mode keeps zend_alloc.c and renames its allocating entry points aside, exactly as
    # the physical build does, so libc/php_capstone_zend_hier.c can wrap them. The rename is
    # scoped to that ONE TU: applied globally it would bypass the wrapper everywhere.
    TUS="$TUS zend_alloc"
    ZEND_ALLOC_RENAME=(-D_emalloc=php_raw_emalloc -D_ecalloc=php_raw_ecalloc
                       -D_erealloc=php_raw_erealloc -D_safe_emalloc=php_raw_safe_emalloc
                       -D_estrdup=php_raw_estrdup -D_estrndup=php_raw_estrndup
                       -D_efree=php_raw_efree)

    # ext/standard, byte-identical from the corpus tree. Checked against build-engine.sh's own
    # list rather than trusted: these two must select the same TUs or the two columns are not
    # the same program.
    EXT="$P/ext/standard/url.c"
    case "${PHP_CAP_EXT:-}" in
      pure) for e in string array var html strnatcmp var_unserializer formatted_print; do
              grep -q "ext/standard/$e.c" "$E/build-engine.sh" \
                || { echo "  build-engine.sh's pure set no longer names $e.c" >&2; exit 2; }
              EXT="$EXT $P/ext/standard/$e.c"
            done
            CF+=(-DPHP_CAP_EXT_PURE) ;;
      "")   : ;;
      *)    for e in ${PHP_CAP_EXT//,/ }; do EXT="$EXT $P/ext/standard/$e.c"; done ;;
    esac

    # The support layer, down from eight files to three plus two of our own. php_capstone_
    # {libc,stdio,os,malloc,depth}.c are NOT ported: musl supplies real string, stdio, signals,
    # dlopen and a growing heap, and runtime/virtual/heap.c's mallocng already gives
    # per-object bounds and revoke-before-reuse underneath. What stays is PHP-level rather
    # than OS-level, and therefore has no musl counterpart:
    #   php_capstone_zend_hier.c  the bounds wrapper -- this IS the experiment, reused unchanged
    #   php_capstone_php_stubs.c  main/'s globals (PG/SG/BG, sapi_module) and php_printf et al,
    #                             which exist because main/ is not linked until Phase B
    #   php_capstone_ext_stubs.c  the three externs ext/standard/url.c needs from streams
    # and ours:
    #   php_virtual_lower.c       zend_cap_widen, the identity here -- see that file
    #   php_virtual_main.c        a real main(), the trigger on stdin
    SUP=("$E/libc/php_capstone_zend_hier.c" "$E/libc/php_capstone_php_stubs.c"
         "$E/libc/php_capstone_ext_stubs.c" "$HERE/php_virtual_lower.c"
         "$HERE/php_virtual_main.c")

    OBJS=()
    fails=0
    echo "  compiling $TU_COUNT+1 Zend TUs, $(printf '%s\n' $EXT | wc -l) ext/standard, ${#SUP[@]} support ..."
    for t in $TUS; do
      _rn=(); [ "$t" = "zend_alloc" ] && _rn=("${ZEND_ALLOC_RENAME[@]}")
      "$CC" "${CF[@]}" ${_rn[@]+"${_rn[@]}"} -c "$P/Zend/$t.c" -o "$OBJ/$t.o" 2>"$LOG/$t.log" \
        || { echo "    FAILED Zend/$t"; fails=$((fails + 1)); continue; }
      OBJS+=("$OBJ/$t.o")
    done
    for s in $EXT; do
      b=$(basename "$s" .c)
      "$CC" "${CF[@]}" -c "$s" -o "$OBJ/ext_$b.o" 2>"$LOG/ext_$b.log" \
        || { echo "    FAILED ext/$b"; fails=$((fails + 1)); continue; }
      OBJS+=("$OBJ/ext_$b.o")
    done
    for s in "${SUP[@]}"; do
      b=$(basename "$s" .c)
      "$CC" "${CF[@]}" -c "$s" -o "$OBJ/$b.o" 2>"$LOG/$b.log" \
        || { echo "    FAILED $b"; fails=$((fails + 1)); continue; }
      OBJS+=("$OBJ/$b.o")
    done
    echo "  compiled ${#OBJS[@]} objects, $fails failed"
    if [ "$fails" -gt 0 ]; then
      echo "  first errors:" >&2
      grep -l "error:" "$LOG"/*.log 2>/dev/null | head -8 | while read -r l; do
        echo "    == $(basename "$l" .log)" >&2
        grep "error:" "$l" | head -4 | sed 's/^/       /' >&2
      done
      exit 1
    fi

    # THE IMAGE IS NAMED AFTER ITS ARM, derived from the flag rather than passed in, so the two
    # arms of a pair cannot be mislabelled by a typo in a wrapper. The spatial control is the
    # one that reproduces stock PHP's REAL_SIZE rounding and must COMPLETE; the fault arm bounds
    # every capability to the caller's request.
    case " ${EXTRA_CF:-} " in
      *" -DZEND_CAP_BOUNDS_REAL_SIZE "*) ARM=control ;;
      *) ARM=fault ;;
    esac
    IMG="$OUT/php-virtual/php-$ARM.dom"
    echo "  arm: $ARM"
    echo "  linking ..."
    "$CC" "${CF[@]}" "${OBJS[@]}" -lm -o "$IMG" 2>"$LOG/link.log" \
      || { echo "  LINK FAILED"; tail -30 "$LOG/link.log" >&2; exit 1; }
    echo "  built $IMG"
    "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$IMG" 2>/dev/null \
      | awk '$2==".text"||$2==".rodata"||$2==".data"||$2==".bss"{printf "    %-10s %d\n",$2,strtonum("0x" $3)}'
    # C-56: run-domain-smoke.py refuses any image referencing an UNDEFINED WEAK symbol, and
    # that refusal happens after a boot. Cheaper to see it here.
    undef_weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" --undefined-only "$IMG" 2>/dev/null | awk '$1=="w"{print $2}')
    if [ -n "$undef_weak" ]; then
      echo "  UNDEFINED WEAK symbols (C-56 will refuse this image):" >&2
      printf '    %s\n' $undef_weak >&2
      exit 1
    fi
    sha256sum "$IMG"
    ;;
  *) echo "usage: build-virtual.sh [probe|hier-probe|php]" >&2; exit 2 ;;
esac
