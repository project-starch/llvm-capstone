#!/usr/bin/env bash
# run-arm.sh OUT: the nine virtual-runtime landing gates for one arm, with EVERY image the gates
# run built from that arm -- the contract programs and the applications alike.
#
#   env TREE=<the arm's llvm-capstone tree> PLATFORM=<dir: qemu-system-riscv64, images/, adapter/> \
#       HOOKS=<the arm's CV_TEST_HOOKS module .ko> LLVM=<compiler build dir: bin/, lib/clang> \
#       CROSS_GCC=<riscv64 Linux gcc, for the native references> \
#       [SDK=<an application SDK built from TREE>] [PHASE=all|build|gates] [GATES="hosted runtime ..."] \
#       [NATIVE_EXEC=1 (hosted/runtime only)] [MODULE_ARGS=<runtime module parameters>] \
#       [CPU_EXTRA=<QEMU -cpu extras; default the 32-slot gate set>] [QEMU_LOCK=<lock file>] \
#       [APP_SQLITE=... APP_MRUBY=... APP_PERL=... APP_MEMCACHED=...] [JOBS=8] \
#     bash run-arm.sh OUT
#
# WHY THE APPLICATIONS ARE BUILT HERE. The runtime gate's sqlite/mruby/perl rows and the memcached
# gate used to take images prebuilt in a kit. Once the image ABI moved on, the launcher refused all
# of them (exit 126, "capstone-vexec: image: Exec format error") in the control and in the arm
# alike, so those rows "matched the control" without ever running, and a runtime or SDK change was
# never exercised by an application. Here each application is built from the arm's own tree, with
# the arm's SDK where its recipe takes one (sqlite, mruby) and with the SDK its recipe builds from
# the same tree where it builds its own (perl, memcached). images.sha256 records every image and
# every libc archive, so the SDK identity of each image can be read back.
#
# The APP_* overrides take a prebuilt image (APP_MEMCACHED: a build directory) instead. They exist for the positive
# control of compare.py (a deliberately stale image must make the verdict INCOMPLETE), not for a
# landing run. An application that fails to build is staged as an empty image: its row then does
# not execute, and compare.py reports it NOT RUN instead of letting the gate pass or match.
#
# MUSL_ARCHIVE=<musl-1.2.5.tar.gz> is passed through to every recipe's fetch-musl.sh, which checks its
# pinned SHA-256: one local archive instead of five downloads per arm.
#
# PHASE=build stops after the images; PHASE=gates runs the gates on an OUT whose build finished.
# Compare two arms with compare.py CONTROL_OUT ARM_OUT. Gate arms share host ports (memcached's
# native reference): run them one after another, never two at once.
set -uo pipefail
OUT=${1:?usage: run-arm.sh OUT (see the header)}
: "${TREE:?TREE=<the arm tree>}" "${PLATFORM:?PLATFORM=<platform dir>}" "${HOOKS:?HOOKS=<hooks .ko>}"
: "${LLVM:?LLVM=<compiler build dir>}" "${CROSS_GCC:?CROSS_GCC=<riscv64 linux gcc>}"
PHASE=${PHASE:-all}
if [[ ${NATIVE_EXEC:-0} == 1 ]]; then
  GATES=${GATES:-hosted runtime}
  for gate in $GATES; do
    [[ $gate == hosted || $gate == runtime ]] || {
      echo "NATIVE_EXEC supports only hosted and runtime; unsupported gate: $gate" >&2; exit 2;
    }
  done
else
  GATES=${GATES:-hosted runtime nodes malloc pin-ahead fp pthread step-hooks memcached}
fi
JOBS=${JOBS:-8}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TREE=$(realpath "$TREE"); PLATFORM=$(realpath "$PLATFORM"); LLVM=$(realpath "$LLVM")
OUT=$(realpath -m "$OUT")
V=$TREE/capstone/runtime/virtual
P=$PLATFORM
B=$OUT/build
G=$OUT/gates
# The recipes read their inputs from the environment first: capstone-test-env.sh takes
# CAPSTONE_REPO_ROOT, CAPSTONE_CLANG and the other tool paths from it when set, capstone-cc prefers
# CAPSTONE_SDK over its own directory, perl prefers RUNTIME_REPO, and each port has its own knobs.
# An inherited value would build from another checkout, toolchain or SDK, so every such variable
# is cleared and the arm's own are set.
for v in $(compgen -e); do
  case $v in
    CAPSTONE_*|MRBD_*|PERLD_*|MC_*|SQLITE_*|PHP_*|MUSL_CACHE_ROOT|MUSL_SRC_DIR|MUSL_VERSION|RUNTIME_REPO) unset "$v" ;;
  esac
done
export CAPSTONE_REPO_ROOT=$TREE CAPSTONE_LLVM_BUILD_DIR=$LLVM CAPSTONE_LLVM_BIN=$LLVM/bin
export CAPSTONE_TMP_ROOT=$OUT/tmp JOBS

# A gate platform is real files: a symlink into another platform turns a later copy into a write
# to that platform, and lets two arms run one binary.
links=$(find "$P" "$HOOKS" -type l | head -3)
[[ -z $links ]] || { echo "platform has symlinks: $links" >&2; exit 2; }

build_phase() {
  [[ ! -e $B ]] || { echo "$B exists: use a fresh OUT" >&2; exit 2; }
  mkdir -p "$B/img" "$CAPSTONE_TMP_ROOT"
  local S=$B/BUILD.txt; : > "$S"
  {
    echo "tree $TREE revision $(cat "$TREE/REVISION" 2>/dev/null || git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "clang $(sha256sum < "$(readlink -f "$LLVM/bin/clang")" | cut -c1-64)"
    echo "build_start $(date -u +%FT%TZ)"
  } >> "$S"
  # ---- the arm's SDK ----
  if [[ -z ${SDK:-} ]]; then
    if ( export MUSL_CACHE_ROOT=$B/musl-src; mkdir -p "$MUSL_CACHE_ROOT"
         musl=$(bash "$TREE/capstone/ports/musl-capstone/prepare-musl-capstone.sh" | tail -1) &&
         bash "$V/build-sdk.sh" "$B/sdk-build" "$musl" ) > "$B/sdk.log" 2>&1; then
      SDK=$B/sdk-build/sdk
    else
      echo "SDK BUILD FAILED (sdk.log)" >> "$S"; echo "build_end $(date -u +%FT%TZ)" >> "$S"; return 1
    fi
  fi
  SDK=$(realpath "$SDK"); echo "sdk $SDK" >> "$S"; printf '%s\n' "$SDK" > "$B/SDK"
  # capstone-cc takes its configuration from CAPSTONE_SDK when that is set; perl's and memcached's
  # recipes set it to the SDK they build.
  export CAPSTONE_SDK=$SDK
  local cc=$SDK/capstone-cc inc=$TREE/capstone/runtime/include
  # ---- the contract images ----
  img() { "$@" >> "$B/images.log" 2>&1 || echo "IMAGE BUILD FAILED: $*" >> "$S"; }
  img "$cc" -O2 "$V/hosted-contract.c" -o "$B/img/hosted.dom"
  img "$cc" -O2 "$V/hosted-noreturn.c" -o "$B/img/noreturn.dom"
  img "$cc" -O1 -I"$inc" "$V/contract.c" -o "$B/img/contract.dom"
  img "$cc" -O1 -I"$inc" "$V/node-contract.c" -o "$B/img/node.dom"
  img "$cc" -I"$V" "$V/malloc-contract.c" -o "$B/img/malloc.dom"
  img "$CROSS_GCC" -static -O1 -o "$B/img/native-hosted" "$V/hosted-contract.c"
  img "$CROSS_GCC" -static -O1 -o "$B/img/native-malloc" "$V/malloc-contract.c"
  img "$cc" -O1 -I"$inc" "$V/pin-ahead-contract.c" -o "$B/img/pin-ahead.dom" -lpthread
  img "$cc" -O1 -I"$inc" "$V/fp-contract.c" -o "$B/img/fp.dom" -lpthread
  img "$cc" -O1 -I"$inc" "$V/pthread-contract.c" -o "$B/img/pthread.dom" -lpthread
  # ---- the applications, in parallel, each from the arm's tree ----
  local ports=$TREE/capstone/ports pids=() app
  app_build() {   # app_build NAME CMD...: one log per application, its status in apps/NAME.status
    local name=$1; shift; local t0; t0=$(date +%s)
    ( "$@" ) > "$B/apps/$name.log" 2>&1
    printf '%s %s\n' $? $(( $(date +%s) - t0 )) > "$B/apps/$name.status"
  }
  mkdir -p "$B/apps"
  if [[ -z ${APP_SQLITE:-} ]]; then
    app_build sqlite bash -c 'set -e; src=$(bash "$1/sqlite/app/prepare-sources.sh" "$2/sqlite-source" | tail -1)
      SQLITE_HEAP=memsys5 bash "$1/sqlite/app/build-domain.sh" "$3" "$src" "$2/sqlite"' _ "$ports" "$B" "$SDK" & pids+=($!)
  fi
  if [[ -z ${APP_MRUBY:-} ]]; then
    app_build mruby env MRBD_ROOT="$B/mruby" MRBD_SDK="$SDK" MRBD_SUBLET=0 \
      bash "$ports/mruby/app/build-mruby-domain.sh" & pids+=($!)
  fi
  if [[ -z ${APP_PERL:-} ]]; then
    app_build perl env PERLD_ROOT="$B/perl" PERLD_PROFILE=virtual PERLD_SUBLET=0 \
      bash "$ports/perl/musl/build-perl-domain.sh" & pids+=($!)
  fi
  if [[ -z ${APP_MEMCACHED:-} ]]; then
    app_build memcached env -u MC_WORK bash "$ports/memcached/app/host/build-virtual.sh" "$B/memcached" & pids+=($!)
  fi
  for pid in "${pids[@]}"; do wait "$pid"; done
  # Resolve each application's image: built, given, or (failed build) an empty stand-in.
  resolve() {   # resolve NAME BUILT_PATH OVERRIDE
    local name=$1 built=$2 given=$3
    if [[ -n $given ]]; then
      printf '%s\n' "$(realpath "$given")" > "$B/apps/$name.image"; echo "app $name GIVEN $given" >> "$S"
    elif [[ $(cut -d' ' -f1 "$B/apps/$name.status") == 0 && -s $built ]]; then
      printf '%s\n' "$built" > "$B/apps/$name.image"
      echo "app $name BUILT in $(cut -d' ' -f2 "$B/apps/$name.status") s" >> "$S"
    else
      : > "$B/apps/$name.missing.dom"; printf '%s\n' "$B/apps/$name.missing.dom" > "$B/apps/$name.image"
      echo "APP BUILD FAILED: $name (rc $(cut -d' ' -f1 "$B/apps/$name.status" 2>/dev/null), apps/$name.log); staged as an empty image" >> "$S"
    fi
  }
  resolve sqlite "$B/sqlite/sqlite3.dom" "${APP_SQLITE:-}"
  resolve mruby "$B/mruby/src/mruby/build/capstone/bin/mruby" "${APP_MRUBY:-}"
  resolve perl "$B/perl/src/perl-5.36.3/perl" "${APP_PERL:-}"
  if [[ -n ${APP_MEMCACHED:-} ]]; then
    printf '%s\n' "$(realpath "$APP_MEMCACHED")" > "$B/apps/memcached.image"; echo "app memcached GIVEN $APP_MEMCACHED" >> "$S"
  elif [[ $(cut -d' ' -f1 "$B/apps/memcached.status") == 0 && -s $B/memcached/domain/memcached.dom ]]; then
    printf '%s\n' "$B/memcached" > "$B/apps/memcached.image"
    echo "app memcached BUILT in $(cut -d' ' -f2 "$B/apps/memcached.status") s" >> "$S"
  else
    echo "APP BUILD FAILED: memcached (rc $(cut -d' ' -f1 "$B/apps/memcached.status" 2>/dev/null), apps/memcached.log); the gate cannot stage" >> "$S"
  fi
  # ---- identities: every image and every libc archive the images were linked from ----
  {
    sha256sum "$B"/img/*
    for name in sqlite mruby perl; do sha256sum "$(cat "$B/apps/$name.image")" | sed "s|  |  [$name] |"; done
    if [[ -f $B/apps/memcached.image ]]; then
      m=$(cat "$B/apps/memcached.image")
      sha256sum "$m"/domain/memcached.dom "$m"/marker/memcached.dom "$m"/safety/memcached-safety-virtual.dom "$m"/pthread.dom 2>&1
    fi
    find "$(dirname "$SDK")/" "$B" -name 'libc-capstone*.a' -print0 2>/dev/null | sort -zu | xargs -0 -r realpath -z | sort -zu | xargs -0 -r sha256sum
  } > "$B/images.sha256"
  echo "build_end $(date -u +%FT%TZ)" >> "$S"
  touch "$B/BUILD_DONE"
}

gates_phase() {
  [[ -f $B/BUILD_DONE ]] || { echo "no finished build in $B" >&2; exit 2; }
  [[ ! -e $G ]] || { echo "$G exists: use a fresh OUT" >&2; exit 2; }
  mkdir -p "$G/tmp"
  export CAPSTONE_QEMU_LOCK=${QEMU_LOCK:-$OUT/qemu.lock} CAPSTONE_TMP_ROOT=$G/tmp
  export CAPSTONE_QEMU_CPU_EXTRA=${CPU_EXTRA:-x-capstone-test-csrs=true,x-capstone-supervisor-slots=32}
  sha256sum "$P/qemu-system-riscv64" "$P"/images/* "$P/adapter/capstone-vexec" "$P/adapter/capstone-job" \
    "$P/adapter/module/capstone_vm.ko" "$HOOKS" > "$OUT/platform.sha256"
  local plat=(--qemu "$P/qemu-system-riscv64" --images "$P/images") I=$B/img S=$G/SUMMARY.txt
  : > "$S"
  gate() {
    local name=$1; shift
    [[ " $GATES " == *" $name "* ]] || return 0
    local s; s=$(date +%s); "$@" > "$G/$name.log" 2>&1
    printf '%-10s rc=%s %6ss\n' "$name" $? $(( $(date +%s) - s )) >> "$S"
  }
  local native_exec=()
  [[ ${NATIVE_EXEC:-0} != 1 ]] || native_exec+=(--native-exec)
  gate hosted python3 "$V/run-hosted.py" --adapter "$P/adapter" --application "$I/hosted.dom" \
    "${native_exec[@]}" --noreturn "$I/noreturn.dom" --native "$I/native-hosted" "${plat[@]}" --work "$G/hosted"
  gate runtime python3 "$V/run.py" --adapter "$P/adapter" --application "$I/contract.dom" \
    "${native_exec[@]}" --sqlite "$(cat "$B/apps/sqlite.image")" --mruby "$(cat "$B/apps/mruby.image")" \
    --perl "$(cat "$B/apps/perl.image")" --record "$G/runtime.json" --timeout 900 \
    --module-args "${MODULE_ARGS:-}" "${plat[@]}"
  gate nodes python3 "$V/run-nodes.py" --adapter "$P/adapter" --application "$I/node.dom" "${plat[@]}" --work "$G/nodes"
  gate malloc python3 "$V/run-malloc.py" --adapter "$P/adapter" --application "$I/malloc.dom" \
    --native "$I/native-malloc" "${plat[@]}" --work "$G/malloc"
  gate pin-ahead python3 "$V/run-pin-ahead.py" --adapter "$P/adapter" --application "$I/pin-ahead.dom" "${plat[@]}" --work "$G/pin-ahead"
  gate fp python3 "$V/run-fp.py" --adapter "$P/adapter" --application "$I/fp.dom" "${plat[@]}" --work "$G/fp"
  gate pthread python3 "$V/run-pthreads.py" --adapter "$P/adapter" --application "$I/pthread.dom" "${plat[@]}" --work "$G/pthread"
  # The README's form: the pin-ahead image (contract.dom dies at VM_CONTRACT_FAIL:argv here).
  gate step-hooks python3 "$V/run-step-hooks.py" --adapter "$P/adapter" --module "$HOOKS" \
    --application "$I/pin-ahead.dom" "${plat[@]}" --work "$G/step-hooks"
  if [[ -f $B/apps/memcached.image ]]; then
    local m; m=$(cat "$B/apps/memcached.image")
    gate memcached python3 "$TREE/capstone/ports/memcached/app/host/run-virtual.py" --build "$m" \
      --adapter "$P/adapter" "${plat[@]}" --cross-cc "$CROSS_GCC" --pthread "$m/pthread.dom" --work "$G/memcached"
  elif [[ " $GATES " == *" memcached "* ]]; then
    printf '%-10s NOT RUN (no memcached build)\n' memcached >> "$S"
  fi
  echo "ARM_DONE $(date -u +%FT%TZ)" >> "$S"
  cat "$S"
}

case "$PHASE" in
  all) build_phase && gates_phase ;;
  build) build_phase ;;
  gates) gates_phase ;;
  *) echo "PHASE=$PHASE? (all, build, gates)" >&2; exit 2 ;;
esac
