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
  inert)
    # PATCH INERTNESS (the plan's Verification 4). The patched zend_alloc.c, compiled WITHOUT
    # -DZEND_CAP_SUBLET, must produce the same object as the pristine source. If it does, then
    # the physical column's results and Phase A's hierarchical results are untouched by the
    # file having been patched at all, and the three ZEND_CAP_* macros really are inert.
    #
    # Compiled at -g0 for this comparison ONLY. Debug info embeds the source PATH, and the two
    # files necessarily live at different paths (/corpus is read-only, so the patched copy is
    # under $OUT) -- that difference is not codegen, and comparing with it in would make the
    # check fail for a reason that has nothing to do with the patch.
    P=${PHP_SRC:-/corpus/build/php-5.0.0}
    [ -d "$P/Zend" ] || { echo "PHP source not at $P -- corpus not mounted?" >&2; exit 2; }
    SEAM="$R/capstone/ports/php/zend-alloc/sublet"
    PATCHES="$R/capstone/ports/php/zend-alloc/patches"
    W="$OUT/php-virtual/inert"
    rm -rf "$W"; mkdir -p "$W/patched/Zend"
    cp "$P/Zend/zend_alloc.c" "$W/patched/Zend/zend_alloc.c"
    patch -s -p1 -d "$W/patched" < "$PATCHES/php-5.0.0-0001-zend-alloc-sublet-hooks.patch" \
      || { echo "  the patch does not apply" >&2; exit 1; }
    ICF=(-O0 -std=gnu89 -fno-common -g0 -mllvm -capstone-shrink-stack=false
         -I"$P" -I"$P/Zend" -I"$P/main" -I"$P/TSRM" -I"$P/ext" -I"$P/ext/standard")
    "$CC" "${ICF[@]}" -c "$P/Zend/zend_alloc.c" -o "$W/pristine.o" 2>"$W/pristine.log" \
      || { echo "  pristine zend_alloc.c did not compile" >&2; tail -5 "$W/pristine.log" >&2; exit 1; }
    "$CC" "${ICF[@]}" -c "$W/patched/Zend/zend_alloc.c" -o "$W/patched.o" 2>"$W/patched.log" \
      || { echo "  patched zend_alloc.c did not compile with the define absent" >&2
           tail -5 "$W/patched.log" >&2; exit 1; }
    if cmp -s "$W/pristine.o" "$W/patched.o"; then
      echo "  PATCH IS INERT: patched == pristine byte for byte with ZEND_CAP_SUBLET undefined"
      "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$W/pristine.o" 2>/dev/null \
        | awk '$2==".text"{printf "    .text %d bytes, identical\n", strtonum("0x" $3)}'
    else
      echo "  PATCH IS NOT INERT -- the guarded hunks changed the object:" >&2
      cmp -l "$W/pristine.o" "$W/patched.o" 2>/dev/null | head -5 >&2
      "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$W/pristine.o" > "$W/h-pristine.txt" 2>/dev/null
      "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$W/patched.o"  > "$W/h-patched.txt"  2>/dev/null
      diff "$W/h-pristine.txt" "$W/h-patched.txt" >&2 || true
      exit 1
    fi
    # And the positive control: WITH the define it must differ, or the check above is vacuous.
    "$CC" "${ICF[@]}" -DZEND_CAP_SUBLET -I"$SEAM" -I"$R/capstone" \
      -c "$W/patched/Zend/zend_alloc.c" -o "$W/sublet.o" 2>"$W/sublet.log" \
      || { echo "  patched zend_alloc.c did not compile WITH the define" >&2
           tail -5 "$W/sublet.log" >&2; exit 1; }
    if cmp -s "$W/pristine.o" "$W/sublet.o"; then
      echo "  CONTROL FAILED: -DZEND_CAP_SUBLET produced the same object, so the hunks do nothing" >&2
      exit 1
    fi
    echo "  control: with -DZEND_CAP_SUBLET the object DOES differ, so the hunks are live"
    ;;
  seam-probe)
    # The seam allocator, driven directly through the hooks the patched zend_alloc.c calls,
    # before any of PHP is pointed at it. Links the REAL zend-cap-*.c: a probe with its own
    # copy of the logic would pass while the engine broke.
    SEAM="$R/capstone/ports/php/zend-alloc/sublet"
    IMG="$OUT/probe-seam.dom"
    echo "  compiling the seam probe ..."
    "$CC" -O1 -I"$R/capstone" -I"$SEAM" \
      "$SEAM/probe-seam.c" "$SEAM/zend-cap-blocks.c" "$SEAM/zend-cap-backing.c" \
      -o "$IMG" || { echo "seam probe build FAILED" >&2; exit 1; }
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
    MODE=${PHP_CAP_ALLOC_MODE:-hier}
    case "$MODE" in
      hier|seam) ;;
      *) echo "  PHP_CAP_ALLOC_MODE=$MODE? (hier|seam)" >&2; exit 2 ;;
    esac
    SEAM="$R/capstone/ports/php/zend-alloc/sublet"
    PATCHES="$R/capstone/ports/php/zend-alloc/patches"

    # Per-arm object and log directories. One shared obj/ would have the second build
    # overwrite the first's objects, and a pair assembled from whichever came last is not a
    # pair -- the two images would be byte-identical and the suite's own gate would refuse them.
    #
    # SEAM MODE IS THE EXCEPTION, and deliberately so: there the arm is a RUNTIME mode
    # (PHP_CAP_MODE, read by zend_cap_mode()), so ONE image serves both runs and the two arms
    # cannot differ in anything but the mode word. The byte-identical check that guards the
    # hierarchical pair would be exactly wrong here -- see the plan's Verification 5, which
    # replaces it with: the image must contain the revoke encoding, the two runs must report
    # different mode lines, and the fault run must report strictly more revokes.
    case " ${EXTRA_CF:-} " in
      *" -DZEND_CAP_ESTRNDUP_AUDIT "*)   _arm=audit ;;
      *" -DZEND_CAP_BOUNDS_REAL_SIZE "*) _arm=control ;;
      *) _arm=fault ;;
    esac
    [ "$MODE" = seam ] && _arm=seam
    OBJ="$OUT/php-virtual/$_arm/obj"; LOG="$OUT/php-virtual/$_arm/log"
    mkdir -p "$OBJ" "$LOG"


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

    # ---- the allocator seam ----------------------------------------------------------
    # zend_alloc.c is PATCHED rather than renamed aside: the six derivation sites have to go
    # through a lookup, which no -D can do. The corpus is mounted read-only, so the patch is
    # applied to a copy and the pristine tree is never touched.
    ZEND_ALLOC_SRC="$P/Zend/zend_alloc.c"
    if [ "$MODE" = seam ]; then
      PATCHED="$OUT/php-virtual/$_arm/patched"
      rm -rf "$PATCHED"
      mkdir -p "$PATCHED/Zend"
      cp "$P/Zend/zend_alloc.c" "$PATCHED/Zend/zend_alloc.c"
      patch -s -p1 -d "$PATCHED" < "$PATCHES/php-5.0.0-0001-zend-alloc-sublet-hooks.patch" \
        || { echo "  the sublet patch does not apply to this tree" >&2; exit 1; }
      ZEND_ALLOC_SRC="$PATCHED/Zend/zend_alloc.c"
      # -I the seam for zend-cap-port.h, and -I capstone for <sublet/sublet.h>, which is the
      # same header the virtual heap itself uses.
      # NOT -DPHP_CAP_ALLOC_SEAM: that is the PHYSICAL port's flag for "zend_alloc.c is
      # dropped", under which libc/php_capstone_php_stubs.c supplies alloc_globals and
      # start_memory_manager itself. This seam KEEPS zend_alloc.c -- that is the whole point --
      # so those definitions must come from PHP, and PHP's AG(cache) stays live. ZEND_CAP_SUBLET
      # is the only flag the seam needs.
      CF+=(-DZEND_CAP_SUBLET -I"$SEAM" -I"$R/capstone")
      ZEND_ALLOC_RENAME=()          # the seam calls PHP's own entry points; nothing is renamed
      echo "  allocator: SEAM -- zend_alloc.c patched (8 guarded hunks), per-emalloc lifetimes"
    else
      echo "  allocator: HIERARCHICAL -- zend_alloc.c renamed aside and re-bounded"
    fi

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
    SUP=("$E/libc/php_capstone_php_stubs.c" "$E/libc/php_capstone_ext_stubs.c"
         "$HERE/php_virtual_lower.c" "$HERE/php_virtual_main.c")
    if [ "$MODE" = seam ]; then
      # The seam serves PHP's allocator from below instead of wrapping it from above, so the
      # re-bounding wrapper is not linked at all: there is nothing to re-bound, because
      # ZEND_CAP_PAYLOAD already hands out a capability bounded to the caller's object.
      SUP+=("$SEAM/zend-cap-blocks.c" "$SEAM/zend-cap-backing.c")
    else
      SUP+=("$E/libc/php_capstone_zend_hier.c")
    fi

    OBJS=()
    fails=0
    echo "  compiling $TU_COUNT+1 Zend TUs, $(printf '%s\n' $EXT | wc -l) ext/standard, ${#SUP[@]} support ..."
    for t in $TUS; do
      _rn=(); _src="$P/Zend/$t.c"
      if [ "$t" = "zend_alloc" ]; then
        _src="$ZEND_ALLOC_SRC"
        [ ${#ZEND_ALLOC_RENAME[@]} -gt 0 ] && _rn=("${ZEND_ALLOC_RENAME[@]}")
      fi
      "$CC" "${CF[@]}" ${_rn[@]+"${_rn[@]}"} -c "$_src" -o "$OBJ/$t.o" 2>"$LOG/$t.log" \
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
    # The audit arm is diagnostic and is never scored: it halts at the allocator boundary with
    # the requested length and what the source capability authorises, which is how a fault
    # performed BY a shared copy primitive gets attributed to a case at all.
    case " ${EXTRA_CF:-} " in
      *" -DZEND_CAP_ESTRNDUP_AUDIT "*)   ARM=audit ;;
      *" -DZEND_CAP_BOUNDS_REAL_SIZE "*) ARM=control ;;
      *) ARM=fault ;;
    esac
    [ "$MODE" = seam ] && ARM=seam
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
  realphp)
    # ---- PHASE B: REAL PHP -------------------------------------------------------------
    #
    # Everything the Makefile builds: PHP_GLOBAL_OBJS + PHP_CLI_OBJS, 145 translation units,
    # including main/ (PHP's runtime host), main/streams/, sapi/cli, TSRM/, regex/ and ALL 65
    # of ext/standard. The `php` case above links 34 Zend TUs, 8 of ext/standard and a ~360-line
    # SAPI of ours standing in for main/'s 12,828; this case links PHP.
    #
    # NOTHING IS DROPPED AND NOTHING IS STUBBED. The plan expected to exclude ext/standard's
    # mail.c and proc_open.c as the only fork() users and to pass -UHAVE_MMAP; neither is
    # needed, because this profile is a Linux process and the SDK's musl has fork, vfork,
    # posix_spawn, popen, system, mmap, munmap AND fopencookie. Those are asserted below rather
    # than assumed -- an SDK that loses one should fail here, not at link.
    P=${PHP_SRC:-/corpus/build/php-5.0.0}
    [ -d "$P/Zend" ] || { echo "PHP source not at $P -- corpus not mounted?" >&2; exit 2; }
    [ -f "$P/Makefile" ] || { echo "no $P/Makefile -- the TU list has no source of truth" >&2; exit 2; }
    SEAM="$R/capstone/ports/php/zend-alloc/sublet"
    PATCHES="$R/capstone/ports/php/zend-alloc/patches"
    MODE=${PHP_REALPHP_ALLOC:-seam}
    case "$MODE" in
      seam|stock) ;;
      *) echo "  PHP_REALPHP_ALLOC=$MODE? (seam|stock)" >&2; exit 2 ;;
    esac
    B="$OUT/realphp/$MODE"
    OBJ="$B/obj"; LOG="$B/log"; T="$B/tree"
    rm -rf "$B"; mkdir -p "$OBJ" "$LOG" "$T"

    # ---- the SDK really does have what real PHP needs --------------------------------
    # Checked against the archives, not against a memory of what musl carries. Each of these
    # is a configure answer php_config.h asserts, and a missing one changes the TU list.
    : > "$B/libc-syms.txt"
    find "$OUT" -name '*.a' -print 2>/dev/null | while read -r _a; do
      "$CAPSTONE_LLVM_BIN/llvm-nm" --defined-only "$_a" 2>/dev/null | awk '$1!="U"{print $NF}'
    done | sort -u > "$B/libc-syms.txt"
    for _s in fork vfork posix_spawn popen system mmap munmap fopencookie dn_expand dn_skipname; do
      grep -qx "$_s" "$B/libc-syms.txt" \
        || { echo "  the SDK has no '$_s' -- php_config.h claims it; re-check the TU list" >&2; exit 2; }
    done

    # ---- the source overlay ----------------------------------------------------------
    # php_config.h is configure OUTPUT, produced against the host's GLIBC. It defines every
    # HAVE_ unconditionally and carries not one #ifndef guard, so -U/-D on the command line is
    # OVERRIDDEN by the header and does nothing at all -- which is why the plan's -UHAVE_MMAP
    # would have been silently ineffective. Nor can the include path intercept it:
    # Zend/zend_config.h:1 and TSRM/tsrm_config.h:1 include it as "../main/php_config.h", a
    # path relative to the including FILE, which bypasses every -I. The only way PHP sees a
    # corrected answer is for the file at that path to be ours.
    #
    # So: a full MIRROR of the tree -- real directories, symlinked leaves. Mirroring only the
    # top level does NOT work and fails quietly: with $T/Zend a symlink to the corpus
    # directory, zend_config.h's own "../main/php_config.h" resolves against the link TARGET
    # and walks back out to the pristine config, so the correction is ignored and only the
    # link tells you. With real directories every relative include stays inside $T.
    #
    # Because the leaves are symlinks, "byte-identical to the corpus" is not a claim to audit.
    # It is the filesystem, and the assertion below holds it that way.
    while IFS= read -r _d; do mkdir -p "$T/${_d#$P/}"; done < <(find "$P" -mindepth 1 -type d)
    while IFS= read -r _f; do ln -s "$_f" "$T/${_f#$P/}"; done < <(find "$P" -mindepth 1 -type f)

    # THE CROSS-CONFIGURE DELTA, and it is two answers. musl 1.2.5 has res_init, res_mkquery,
    # res_send, res_query, res_search, dn_expand and dn_comp, but NOT the reentrant res_n*
    # variants glibc carries. ext/standard/dns.h:26 derives HAVE_DNS_FUNCS from exactly these
    # two, so correcting them drops dns_get_record() and PHP_MINIT_FUNCTION(dns) the way PHP
    # already does on any platform without a reentrant resolver -- a configuration PHP
    # supports, not a stub of ours. Each is defined FOUR times in the generated config.
    rm "$T/main/php_config.h"
    sed -e 's|^#define HAVE_RES_NMKQUERY 1$|#define HAVE_RES_NMKQUERY 0  /* musl 1.2.5: no reentrant resolver */|' \
        -e 's|^#define HAVE_RES_NSEND 1$|#define HAVE_RES_NSEND 0  /* musl 1.2.5: no reentrant resolver */|' \
        "$P/main/php_config.h" > "$T/main/php_config.h"
    _delta=$(diff "$P/main/php_config.h" "$T/main/php_config.h" | grep -c '^<' || true)
    [ "$_delta" -eq 8 ] || { echo "  php_config.h: corrected $_delta answers, expected 8 (4 duplicate defines each) -- the config's shape changed" >&2; exit 2; }
    echo "  php_config.h: 2 configure answers corrected (8 defines), everything else symlinked"

    # ---- the allocator seam ----------------------------------------------------------
    # The seam patches Zend/zend_alloc.c, which in the mirror means replacing one symlink with
    # the patched copy. ONE IMAGE SERVES EVERY ARM: zend_cap_mode() reads PHP_CAP_MODE from the
    # environment, so the spatial pair (bound rounded vs exact) and the three temporal columns
    # (nothing revoked / per block / per object) are all runtime modes over these same bytes.
    # Phase A needed a different image per spatial arm; this does not, so an arm cannot differ
    # in anything but the mode word.
    SUP=()
    if [ "$MODE" = seam ]; then
      rm "$T/Zend/zend_alloc.c"
      cp "$P/Zend/zend_alloc.c" "$T/Zend/zend_alloc.c"
      patch -s -p1 -d "$T" < "$PATCHES/php-5.0.0-0001-zend-alloc-sublet-hooks.patch" \
        || { echo "  the sublet patch does not apply to this tree" >&2; exit 1; }
      SUP=("$SEAM/zend-cap-blocks.c" "$SEAM/zend-cap-backing.c" "$HERE/php_virtual_lower.c")
      echo "  allocator: SEAM -- zend_alloc.c patched (8 guarded hunks), per-emalloc lifetimes"
    else
      echo "  allocator: STOCK -- pristine zend_alloc.c over mallocng, the honest-gap column"
    fi
    # Only the files we deliberately replaced may be real; everything else must still BE the
    # corpus file. Without this the overlay could silently become a fork of PHP.
    _reals=$(find "$T" -type f ! -name php_config.h ! -name zend_alloc.c | wc -l)
    [ "$_reals" -eq 0 ] || { echo "  overlay has $_reals real files beyond the two replaced:" >&2
                             find "$T" -type f ! -name php_config.h ! -name zend_alloc.c >&2; exit 2; }

    # ---- the TU list, from the Makefile ----------------------------------------------
    # PHP_GLOBAL_OBJS and PHP_CLI_OBJS are what `make` builds. Read, not retyped: a list of
    # real PHP that lives in our repo would drift from real PHP and the drift would be invisible.
    { sed -n '/^PHP_GLOBAL_OBJS *=/p' "$P/Makefile"; sed -n '/^PHP_CLI_OBJS *=/p' "$P/Makefile"; } \
      | grep -o '[A-Za-z_0-9/]*\.lo' | sed 's/\.lo$/.c/' | sort -u > "$B/tus.txt"
    TU_N=$(wc -l < "$B/tus.txt")
    [ "$TU_N" -ge 140 ] || { echo "  only $TU_N TUs read from the Makefile -- PHP_GLOBAL_OBJS moved" >&2; exit 2; }
    # The point of this case is that PHP's own host is linked. If these are absent we are
    # building the Phase A engine again under a new name.
    for _r in sapi/cli/php_cli.c main/main.c main/php_ini.c main/streams/streams.c \
              ext/standard/basic_functions.c ext/standard/datetime.c TSRM/TSRM.c; do
      grep -qx "$_r" "$B/tus.txt" \
        || { echo "  the Makefile's list has no $_r -- this is not real PHP" >&2; exit 2; }
    done
    while IFS= read -r _f; do
      [ -f "$T/$_f" ] || { echo "  no source for $_f" >&2; exit 2; }
    done < "$B/tus.txt"
    echo "  TUs: $TU_N (the Makefile's own list, nothing dropped)"

    # ---- flags -----------------------------------------------------------------------
    # -D__off64_t=off_t is the ONLY PHP-side compiler change, and it is type-correct rather
    # than a workaround. php_config.h sets COOKIE_SEEKER_USES_OFF64_T, whose branch in
    # main/streams/cast.c declares the seeker as (void *, __off64_t *, int) -- a POINTER, which
    # is exactly musl's cookie_seek_function_t; musl simply has no glibc-INTERNAL __off64_t
    # typedef, and it defines off64_t as off_t. So fopencookie is used, not disabled.
    #
    # -fno-common, NOT PHP's own -fcommon, and this is forced. PHP 5.0.0's Makefile compiles
    # with -fcommon, but -fcommon CRASHES this backend: CapstoneAsmPrinter::emitGlobalVariable
    # routes every global through SelectSectionForGlobal, and there is no ELF section kind for
    # a common symbol -- "Unknown section kind", UNREACHABLE at
    # TargetLoweringObjectFileImpl.cpp:631, on 8 TUs. That is a real limitation and not a stale
    # toolchain: there is no commit touching llvm/lib/Target/Capstone since this clang was
    # built. It also makes sense -- a common symbol has no definite size or section until link,
    # and a capability bound needs both.
    #
    # -Wno-implicit-int -Wno-implicit-function-declaration -Wno-format-security come from the
    # Makefile's own CFLAGS_CLEAN. They are warnings, so they change no codegen; they are here
    # so PHP's pre-C99 source is not read as our build being broken.
    CF=(-O0 -std=gnu89 -fno-common -gline-tables-only
        -D__off64_t=off_t
        -Wno-implicit-int -Wno-implicit-function-declaration -Wno-format-security
        -I"$T" -I"$T/Zend" -I"$T/main" -I"$T/TSRM" -I"$T/ext" -I"$T/ext/standard"
        -I"$T/regex" -I"$T/sapi/cli")
    if [ "$MODE" = seam ]; then CF+=(-DZEND_CAP_SUBLET -I"$SEAM" -I"$R/capstone"); fi
    # SHRINK-STACK: its own variable, as the plan requires. The 4 MiB domain ceiling that
    # forced it off physically is gone, and it is what makes a stack-object defect (CRASH-100,
    # stack-use-after-return) catchable at all -- but it is not free and must be measured
    # against the known cases before anything is claimed about the stack class.
    if [ "${PHP_VIRTUAL_SHRINK_STACK:-0}" = 1 ]; then
      echo "  shrink-stack: ON (stack objects bounded)"
    else
      CF+=(-mllvm -capstone-shrink-stack=false)
      echo "  shrink-stack: off (the comparison build)"
    fi
    # shellcheck disable=SC2206
    [ -n "${EXTRA_CF:-}" ] && CF+=(${EXTRA_CF})

    # ---- compile ---------------------------------------------------------------------
    OBJS=(); fails=0
    echo "  compiling $TU_N PHP TUs${SUP[0]+ plus ${#SUP[@]} support} ..."
    while IFS= read -r _f; do
      _o="$OBJ/$(echo "$_f" | tr '/' '_' | sed 's/\.c$/.o/')"
      if "$CC" "${CF[@]}" -c "$T/$_f" -o "$_o" 2>"$LOG/$(echo "$_f" | tr '/' '_').log"
      then OBJS+=("$_o")
      else echo "    FAILED $_f"; fails=$((fails + 1)); fi
    done < "$B/tus.txt"
    for _s in ${SUP[0]+"${SUP[@]}"}; do
      _b=$(basename "$_s" .c)
      if "$CC" "${CF[@]}" -c "$_s" -o "$OBJ/sup_$_b.o" 2>"$LOG/sup_$_b.log"
      then OBJS+=("$OBJ/sup_$_b.o")
      else echo "    FAILED $_b"; fails=$((fails + 1)); fi
    done
    echo "  compiled ${#OBJS[@]} objects, $fails failed"
    if [ "$fails" -gt 0 ]; then
      grep -l "error:" "$LOG"/*.log 2>/dev/null | head -8 | while read -r _l; do
        echo "    == $(basename "$_l" .log)" >&2
        grep "error:" "$_l" | head -4 | sed 's/^/       /' >&2
      done
      exit 1
    fi

    # ---- the one duplicate symbol ----------------------------------------------------
    # flex emits yytext as a plain global in BOTH generated scanners. In each file line 16 maps
    # yytext to the prefixed name, line 349 #undefs it (which also wipes any -Dyytext= from the
    # command line), line 478 then emits a bare `char *yytext;`, and line 501 redefines yytext
    # as SCNG(yy_text) -- so every real USE goes to the scanner globals and the bare word is
    # dead. PHP's own build merges the two via -fcommon, which this backend cannot do.
    #
    # It is localized in the INI scanner, so the file left untouched is PHP's lexer -- the one
    # whose codegen the corpus cases actually exercise. NOT --allow-multiple-definition, which
    # would also hide the next duplicate, and that one might be real. The assertion first: if
    # any object ever references yytext across a TU boundary, localizing it would break the
    # link, and we want to hear that here rather than debug it in the guest.
    _refs=$("$CAPSTONE_LLVM_BIN/llvm-nm" --undefined-only "${OBJS[@]}" 2>/dev/null | awk '$2=="yytext"' | wc -l)
    [ "$_refs" -eq 0 ] || { echo "  yytext is referenced across a TU boundary now ($_refs refs) -- do not localize it" >&2; exit 2; }
    "$CAPSTONE_LLVM_BIN/llvm-objcopy" --localize-symbol=yytext "$OBJ/Zend_zend_ini_scanner.o" \
      || { echo "  could not localize the ini scanner's dead yytext" >&2; exit 1; }

    # ---- link ------------------------------------------------------------------------
    IMG="$OUT/realphp/php-real-$MODE.dom"
    echo "  linking ..."
    "$CC" "${CF[@]}" "${OBJS[@]}" -lm -o "$IMG" 2>"$LOG/link.log" \
      || { echo "  LINK FAILED"; grep -o "undefined symbol: [A-Za-z_0-9]*" "$LOG/link.log" \
             | sed 's/undefined symbol: //' | sort -u | sed 's/^/    U /' >&2
           grep -i error "$LOG/link.log" | grep -v "undefined symbol" | sort -u | head >&2; exit 1; }
    echo "  built $IMG"
    "$CAPSTONE_LLVM_BIN/llvm-objdump" -h "$IMG" 2>/dev/null \
      | awk '$2==".text"||$2==".rodata"||$2==".data"||$2==".bss"{printf "    %-10s %d\n",$2,strtonum("0x" $3)}'
    ls -l "$IMG" | awk '{print "    image bytes: " $5}'
    # C-56: run-domain-smoke.py refuses an image referencing an undefined weak symbol, and it
    # refuses it AFTER a boot. Cheaper here.
    undef_weak=$("$CAPSTONE_LLVM_BIN/llvm-nm" --undefined-only "$IMG" 2>/dev/null | awk '$1=="w"{print $2}')
    if [ -n "$undef_weak" ]; then
      echo "  UNDEFINED WEAK symbols (C-56 will refuse this image):" >&2
      printf '    %s\n' $undef_weak >&2; exit 1
    fi
    # The seam is a runtime mode, so the image must at least CONTAIN the revoke encoding --
    # otherwise every temporal column would read the same and all three would look like
    # "nothing revoked". The same gate run-cases.py applies before it believes a pair.
    if [ "$MODE" = seam ]; then
      _rev=$("$CAPSTONE_LLVM_BIN/llvm-objdump" -d "$IMG" 2>/dev/null | grep -c '\brevoke\b')
      [ "$_rev" -gt 0 ] || { echo "  the seam image contains NO revoke instruction" >&2; exit 1; }
      echo "    revoke instructions: $_rev"
    fi
    sha256sum "$IMG"
    ;;
  *) echo "usage: build-virtual.sh [probe|hier-probe|seam-probe|inert|php|realphp]" >&2; exit 2 ;;
esac
