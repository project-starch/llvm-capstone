#!/usr/bin/env bash
# build-plain.sh: the Plain SPEC images (label plain-rv64) for the train runs. A fresh copy of spec/pristine/SPEC17 as
# spec/tree-plain, the manifest's src.alt packages installed as tools/build.sh installs them (sha256 checked), the
# config plain-rv64.cfg (mkcfg.py from the kit's capstone-virtual.cfg), runcpu build, then runcpu --fake --action=run
# --size=train for the stages. Log spec/plain/build-plain.log, marker spec/plain/.build-plain-done.
main() {
  set -u
  local W=${CAPSTONE_WORK:-/mnt/hdd/$USER/capstone}; local S=$W/spec P=$W/spec/plain T=$W/spec/tree-plain
  local K=$W/wt-spec-layout/capstone/ports/spec-cpu2017 L=$P/build-plain.log
  local pin="taskset -c 16-31,48-63 nice -n 10 ionice -c3"
  local B="500.perlbench_r 505.mcf_r 519.lbm_r 525.x264_r 538.imagick_r 544.nab_r 557.xz_r"
  : > "$L"
  [ -e "$T" ] && { echo "tree-plain exists; refusing" >> "$L"; touch "$P/.build-plain-done"; return 1; }
  mkdir -p "$T" && $pin cp -a "$S/pristine/SPEC17" "$T/" || { echo "copy FAILED" >> "$L"; touch "$P/.build-plain-done"; return 1; }
  cd "$T/SPEC17" || return 1
  set +u; source ./shrc > /dev/null; set -u
  while read -r b p want; do
    [ "$(sha256sum "$S/$p" | cut -c1-64)" = "$want" ] || { echo "srcalt $p hash mismatch" >> "$L"; touch "$P/.build-plain-done"; return 1; }
    specxz -dc "$S/$p" | spectar -xf - && echo "srcalt $b $want" >> "$L"
  done < <(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[print(b,p,d['srcalt_sha256'][b]) for b,p in d['srcalt'].items()]" "$K/manifest.lock.json")
  cp "$P/plain-rv64.cfg" config/plain-rv64.cfg
  echo "config $(sha256sum config/plain-rv64.cfg | cut -c1-64)" >> "$L"
  echo "clang $(sha256sum "$(readlink -f "$W/cc-prmerged/bin/clang")" | cut -c1-64)" >> "$L"
  echo "musl libc.a $(sha256sum "$W/perf/T0b/musl-rv64gc/lib/libc.a" | cut -c1-64) libgcc.a $(sha256sum "$W/perf/T0b/tools/libgcc.a" | cut -c1-64)" >> "$L"
  $pin runcpu --config=plain-rv64 --action=build --tune=base --define plain_cc="$P/plain-cc" $B > "$P/runcpu-build.log" 2>&1
  echo "runcpu build rc=$?" >> "$L"
  for b in $B; do for e in benchspec/CPU/$b/exe/*_base.plain-rv64; do
    [ -f "$e" ] && echo "exe $(sha256sum "$e" | cut -c1-64) $e" >> "$L" || echo "exe MISSING $b" >> "$L"; done; done
  $pin runcpu --config=plain-rv64 --tune=base --define plain_cc="$P/plain-cc" --fake --action=run --size=train \
    --noreportable --iterations=1 $B > "$P/fake-train.log" 2>&1
  echo "fake train rc=$?" >> "$L"
  touch "$P/.build-plain-done"
}
main "$@"; exit $?
