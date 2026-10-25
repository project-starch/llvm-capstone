#!/usr/bin/env bash
# THE ONE ENTRY for building an application on the virtual profile.
#
# Fresh source builds: changing only the final link cannot migrate physical objects.
#
#   build-virtual.sh [--dry-run] APP NEW_OUTPUT
#
#   APP   cpython | postgres | ffmpeg | tshark | memcached | perl | mruby | sqlite
#         -- the eight ports with a nested allocator.
#
# WHY ONE ENTRY. Each port keeps its own recipe, because each upstream build is
# its own problem: perl needs perl-cross, tshark needs seven dependencies,
# PostgreSQL needs a host/target split. What was NOT each port's own problem
# was the way in: four of the eight were reachable through this script and four
# were not, and the four that were not each spelled the same choice
# differently -- CPY_SUBLET=1, PGSU_NESTED=sublet, FFAPP_SUBLET=1,
# TSAPP_SUBLET=1, PERLD_SUBLET=1, MRBD_SUBLET=1, SQLITE_HEAP=memsys5-sublet.
# A caller had to know which of seven names its app used, and two ports needed
# an SDK handed to them that nothing built: the mruby images of 2026-10-11 were
# built against "the PostgreSQL lane's pgv5-none SDK, copied"
# (bug-corpora/mruby/release-differential/results/2026-10-11-virtual). An SDK
# that arrives by copy is an SDK nothing identifies.
#
# ONE SELECTOR, for every app:
#
#   CAPSTONE_VIRTUAL_LIFETIMES=0|1   0 (default): the nested allocator as
#                                    released, its blocks from virtual mallocng.
#                                    1: its objects get managed lifetimes --
#                                    CDERIVE on handout, CREVOKE on release.
#
# A port's own variable still works and still means what it meant; setting one
# that contradicts the selector is an error rather than a silent winner.
#
# ONE SDK. Ports that accept an SDK (mruby, sqlite) are given one built here,
# so every image in one output directory shares one SDK identity. Ports that
# build their own (cpython, postgres, ffmpeg, tshark, memcached, perl) still
# do; pipeline.json records which, because that is the remaining difference and
# the next thing to close.
#
# --dry-run resolves the pipeline, writes pipeline.json and builds nothing.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../tests/capstone-test-env.sh" >/dev/null
DRY=0
[[ ${1:-} != --dry-run ]] || { DRY=1; shift; }
[[ $# == 2 ]] || { echo 'usage: build-virtual.sh [--dry-run] {cpython|postgres|ffmpeg|tshark|memcached|perl|mruby|sqlite} NEW_OUTPUT' >&2; exit 2; }
APP=$1
OUT=$(realpath -m "$2")
[[ ! -e "$OUT" ]] || { echo "use a fresh output directory: $OUT" >&2; exit 2; }
case "$APP" in cpython|postgres|ffmpeg|tshark|memcached|perl|mruby|sqlite) ;;
  *) echo "unsupported application: $APP" >&2; exit 2;; esac

# ---- the selector, and the port variable it stands for ----------------------
LIFETIMES=${CAPSTONE_VIRTUAL_LIFETIMES:-0}
case "$LIFETIMES" in 0|1) ;; *) echo "CAPSTONE_VIRTUAL_LIFETIMES=$LIFETIMES? (0, 1)" >&2; exit 2;; esac
# Each entry: the port's own variable, its value for 0, its value for 1.
case "$APP" in
  cpython)   PORT_VAR=CPY_SUBLET;   PORT_OFF=0; PORT_ON=1 ;;
  postgres)  PORT_VAR=PGSU_NESTED;  PORT_OFF=none; PORT_ON=sublet ;;
  ffmpeg)    PORT_VAR=FFAPP_SUBLET; PORT_OFF=0; PORT_ON=1 ;;
  tshark)    PORT_VAR=TSAPP_SUBLET; PORT_OFF=0; PORT_ON=1 ;;
  perl)      PORT_VAR=PERLD_SUBLET; PORT_OFF=0; PORT_ON=1 ;;
  mruby)     PORT_VAR=MRBD_SUBLET;  PORT_OFF=0; PORT_ON=1 ;;
  sqlite)    PORT_VAR=SQLITE_HEAP;  PORT_OFF=memsys5; PORT_ON=memsys5-sublet ;;
  # memcached's recipe builds both arms into one output directory -- stock,
  # safety fixtures and managed lifetimes are separate images, not separate
  # builds -- so there is nothing for the selector to choose.
  memcached) PORT_VAR=; PORT_OFF=; PORT_ON= ;;
esac
[[ $LIFETIMES == 1 ]] && PORT_VALUE=$PORT_ON || PORT_VALUE=$PORT_OFF
if [[ -n $PORT_VAR ]]; then
  if [[ -n ${!PORT_VAR:-} && ${!PORT_VAR} != "$PORT_VALUE" ]]; then
    echo "$PORT_VAR=${!PORT_VAR} contradicts CAPSTONE_VIRTUAL_LIFETIMES=$LIFETIMES" \
         "(which means $PORT_VAR=$PORT_VALUE); set one or the other" >&2
    exit 2
  fi
  export "$PORT_VAR=$PORT_VALUE"
elif [[ $LIFETIMES == 1 ]]; then
  echo "memcached builds every arm in one output; leave CAPSTONE_VIRTUAL_LIFETIMES at 0" >&2
  exit 2
fi

# Ports that take an SDK rather than building one.
case "$APP" in mruby|sqlite) SHARED_SDK=1 ;; *) SHARED_SDK=0 ;; esac
export CAPSTONE_APPLICATION_PROFILE=virtual
PORTS=$CAPSTONE_REPO_ROOT/capstone/ports
NESTED=none
LIBC_FLAGS=()
STACK=1048576
RELINK=1   # cleared by the ports whose recipe links the final image itself

report() {
  mkdir -p "$OUT"
  [[ -n $PORT_VAR ]] && SELECTED="$PORT_VAR=$PORT_VALUE" || SELECTED=
  CAPSTONE_PIPELINE_APP=$APP CAPSTONE_PIPELINE_LIFETIMES=$LIFETIMES \
  CAPSTONE_PIPELINE_SELECTED=$SELECTED CAPSTONE_PIPELINE_SHARED_SDK=$SHARED_SDK \
  CAPSTONE_PIPELINE_RELINK=$RELINK CAPSTONE_PIPELINE_DRY=$DRY \
  CAPSTONE_PIPELINE_OUT=$OUT \
  python3 "$HERE/pipeline.py" > "$OUT/pipeline.json"
  cat "$OUT/pipeline.json"
}
if [[ $DRY == 1 ]]; then
  case "$APP" in memcached|perl|mruby|sqlite) RELINK=0 ;; esac
  report
  exit 0
fi

mkdir -p "$OUT/source"

# ---- one SDK for the ports that take one ------------------------------------
if [[ $SHARED_SDK == 1 ]]; then
  export MUSL_CACHE_ROOT=$OUT/musl-src
  mkdir -p "$MUSL_CACHE_ROOT"
  MUSL=$(bash "$PORTS/musl-capstone/prepare-musl-capstone.sh" | tail -1)
  bash "$CAPSTONE_REPO_ROOT/capstone/runtime/virtual/build-sdk.sh" "$OUT/sdk-build" "$MUSL"
  SDK=$OUT/sdk-build/sdk
  ln -sfn "$SDK" "$OUT/sdk"
fi

case "$APP" in
  cpython)
    export CPY_ROOT=$OUT/source
    bash "$PORTS/cpython/app/prepare-cpython-capstone.sh"
    source "$CPY_ROOT/build/capstone-env.sh"
    python3 "$PORTS/cpython/app/survey-cpython-capstone.py" "$CPY_ROOT/build" --jobs "${JOBS:-12}"
    python3 "$PORTS/cpython/app/link-cpython-capstone.py" "$CPY_ROOT/build" \
      --native "$CPY_ROOT/build-python" --out "$CPY_ROOT/link"
    [[ $LIFETIMES != 1 ]] || NESTED=cpython
    # CPython 3.13 sizes its C recursion limit (Py_C_RECURSION_LIMIT 10000) for the
    # 8 MiB main stack Linux gives a process. With 1 MiB, deep recursion overflows
    # the stack before the interpreter's own guard raises RecursionError.
    STACK=8388608
    ;;
  postgres)
    export PG_SU_ROOT=$OUT/source
    bash "$PORTS/postgres/app/build-domain.sh"
    [[ $LIFETIMES != 1 ]] || NESTED=postgres
    ;;
  ffmpeg)
    export FFAPP_WORK=$OUT/source
    bash "$PORTS/ffmpeg/app/host/build-domain.sh"
    [[ $LIFETIMES != 1 ]] || NESTED=ffmpeg
    ;;
  tshark)
    export TS_WORK=$OUT/source
    for lib in zlib pcre2 c-ares libgpg-error libgcrypt libxml2 glib; do
      bash "$PORTS/wireshark/app/deps/build-$lib.sh"
    done
    bash "$PORTS/wireshark/app/host/cross-build.sh"
    bash "$PORTS/wireshark/app/host/build-domain.sh"
    [[ $LIFETIMES != 1 ]] || NESTED=wireshark
    source "$PORTS/wireshark/app/deps/env.sh"
    LIBC_FLAGS=(--musl "$TS_MUSL" --libc "$TS_LIBC_ARCHIVE")
    ;;
  memcached)
    # The port's own entry: libevent, the domain, the native reference, the
    # marker, the safety fixtures, the managed-lifetime arm and the pthread
    # contract, all linked by the SDK its build-domain.sh builds.
    bash "$PORTS/memcached/app/host/build-virtual.sh" "$OUT/image"
    RELINK=0
    ;;
  perl)
    # build-perl-domain.sh builds its own virtual SDK and links the interpreter,
    # so there is nothing left to relink. PERLD_PROFILE is this script's profile.
    export PERLD_ROOT=$OUT/source PERLD_PROFILE=virtual
    bash "$PORTS/perl/musl/build-perl-domain.sh"
    ln -sfn "$OUT/source" "$OUT/image"
    RELINK=0
    ;;
  mruby)
    # 4.0.0-rc2 is the pin whose patches carry 0008 (the GC slots); the head pin
    # has no managed-lifetime patch.
    export MRBD_ROOT=$OUT/source MRBD_SDK=$SDK
    [[ $LIFETIMES != 1 ]] || export MRBD_PIN=${MRBD_PIN:-4.0.0-rc2}
    bash "$PORTS/mruby/app/build-mruby-domain.sh"
    ln -sfn "$OUT/source" "$OUT/image"
    RELINK=0
    ;;
  sqlite)
    # 3.22.0: the shell and speedtest1, built by the SDK's capstone-cc against
    # musl's unix VFS. Not the freestanding ../build-sqlite-capstone.sh image.
    SQ_SRC=$(bash "$PORTS/sqlite/app/prepare-sources.sh" "$OUT/source" | tail -1)
    bash "$PORTS/sqlite/app/build-domain.sh" "$SDK" "$SQ_SRC" "$OUT/image"
    RELINK=0
    ;;
esac

if [[ $RELINK == 1 ]]; then
  python3 "$HERE/build.py" --app "$APP" --root "$OUT/source" \
    --profile virtual --nested "$NESTED" --toolchain "$CAPSTONE_LLVM_BUILD_DIR" \
    --input-revision HEAD --stack-bytes "$STACK" "${LIBC_FLAGS[@]}" --out "$OUT/image"
fi
report
