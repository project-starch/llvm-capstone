#!/bin/bash
# Build every benchmark of manifest.lock.json for one or more arms in a pristine SPEC tree with runcpu.
# usage: build.sh KIT SPEC_ROOT SDKS RECORD_DIR [LABEL...]
#   KIT         this kit (CAPSTONE_SPEC_PACKAGES: the private directory with the src.alt packages, if not here)
#   SPEC_ROOT   a tree from tools/spec-extract.py ... ALL, never built in before
#   SDKS        recipes/sdks.sh's directory (one SDK per arm), or a single SDK directory (sdk.json), which
#               then serves the default arm only
#   RECORD_DIR  new directory: build-record.txt, arms.json, the effective config, runcpu's logs, ldecod-host
#   LABEL...    arms to build (manifest `arms`); default: every arm the SDKS directory can serve
#   CAPSTONE_SPEC_SOURCE_SELECTION optionally names a source-only overlay; the historical lock stays intact.
# Installs the manifest's src.alt packages once, builds the host input generator of 525.x264_r, writes
# the config runcpu uses, builds each arm under its own label, records the hashes of everything that went
# in and came out, and then checks that the arms' builds differ only by their SDK: every object file of
# every benchmark must be byte-identical across labels (the SDK enters at the link only).
set -euo pipefail
KIT=$(cd "$1" && pwd) SPEC_ROOT=$(cd "$2" && pwd) SDKS=$(cd "$3" && pwd) REC=$4
shift 4
[ -e "$REC" ] && { echo "build: $REC exists" >&2; exit 2; }
mkdir -p "$REC"; REC=$(cd "$REC" && pwd)
M=$KIT/manifest.lock.json
CFG=capstone-virtual
R=$REC/build-record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
fail() { echo "build: $*" >&2; note "FAILED $*"; exit 1; }
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$M" "$1"; }
mapfile -t BENCH < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['workloads']['benchmarks']))" "$M")
# BENCHMARKS="502.gcc_r 538.imagick_r" restricts a build to a subset (a quick look); the record says so.
if [ -n "${BENCHMARKS:-}" ]; then read -r -a BENCH <<< "$BENCHMARKS"; fi
selection_args=()
if [ -n "${CAPSTONE_SPEC_SOURCE_SELECTION:-}" ]; then
  selection_args=(--overlay "$CAPSTONE_SPEC_SOURCE_SELECTION")
fi
python3 "$KIT/tools/build-record.py" select --manifest "$M" "${selection_args[@]}" \
  --out "$REC/source-selection.json" --plan "$REC/srcalt.tsv" --benchmarks "${BENCH[@]}"
PERL_SRCALT=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["source_alternatives"].get("500.perlbench_r",{}).get("name","capstone"))' "$REC/source-selection.json")
# Arms and their SDKs.
declare -A SDK
if [ -f "$SDKS/sdk.json" ]; then
  LABELS=("${@:-capstone-virtual}"); [ ${#LABELS[@]} = 1 ] || fail "a single SDK serves one arm"
  SDK[${LABELS[0]}]=$SDKS
else
  if [ $# -gt 0 ]; then LABELS=("$@"); else mapfile -t LABELS < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['arms']))" "$M"); fi
  for label in "${LABELS[@]}"; do
    dir=$(m "arms.$label.sdk" 2>/dev/null) || fail "$label is not an arm of the manifest"
    [ -f "$SDKS/$dir/sdk.json" ] || fail "$SDKS/$dir has no sdk.json (run recipes/sdks.sh)"
    SDK[$label]=$SDKS/$dir
  done
fi
python3 - "$M" "${BENCH[*]}" "${LABELS[@]}" <<'PYTHON'
import json, re, sys
allowed = json.load(open(sys.argv[1]))['workloads']['benchmarks']
benchmarks = sys.argv[2].split()
if not benchmarks or len(set(benchmarks)) != len(benchmarks) or any(b not in allowed for b in benchmarks):
    raise SystemExit('build: invalid or duplicate benchmark selection')
if any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', label) for label in sys.argv[3:]):
    raise SystemExit('build: invalid arm label')
PYTHON
cd "$SPEC_ROOT"
[ -e "config/$CFG.cfg" ] && fail "$SPEC_ROOT already has config/$CFG.cfg: use a fresh tree"
ls benchspec/CPU/*/src/src.alt/capstone > /dev/null 2>&1 && fail "$SPEC_ROOT already has a capstone src.alt: use a fresh tree"
while read -r b p want name; do
  [ ! -e "benchspec/CPU/$b/src/src.alt/$name" ] || fail "source alternative $b/$name already exists"
done < "$REC/srcalt.tsv"
for label in "${LABELS[@]}"; do
  for b in "${BENCH[@]}"; do
    if compgen -G "benchspec/CPU/$b/build/build_base_${label}.*" > /dev/null; then
      fail "$b/$label already has a build directory"
    fi
  done
done
set +u; source ./shrc > /dev/null; set -u
note "spec_root $SPEC_ROOT version $(cat version.txt)"
note "benchmarks ${BENCH[*]}$([ -n "${BENCHMARKS:-}" ] && echo ' (SUBSET via BENCHMARKS)')"
note "kit $(git -C "$KIT" rev-parse HEAD 2>/dev/null || echo unknown) $(git -C "$KIT" status --porcelain 2>/dev/null | grep -q . && echo dirty || echo clean)"
note "source-selection $(sha256sum "$REC/source-selection.json" | cut -c1-64) perl_srcalt=$PERL_SRCALT"
# src.alt packages, exactly the manifest's. They are SPEC-derived and live outside this public
# directory: CAPSTONE_SPEC_PACKAGES names the private kit that holds packages/ (default: this kit).
PKG=${CAPSTONE_SPEC_PACKAGES:-$KIT}
while read -r b p want name; do
  [ -f "$PKG/$p" ] || fail "package $p is missing under $PKG (set CAPSTONE_SPEC_PACKAGES to the private kit)"
  got=$(sha256sum "$PKG/$p" | cut -c1-64)
  [ "$got" = "$want" ] || fail "$p is $got, the manifest says $want"
  specxz -dc "$PKG/$p" | spectar -xf -
  [ -d "benchspec/CPU/$b/src/src.alt/$name" ] || fail "$p did not install benchspec/CPU/$b/src/src.alt/$name"
  note "srcalt $b $got $(basename "$p") name=$name"
done < "$REC/srcalt.tsv"
# SDKs and the compiler (one compiler for all arms).
for label in "${LABELS[@]}"; do
  sdk=${SDK[$label]}
  note "sdk $label $sdk profile $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('profile'),'heap_lifetimes',d.get('heap_lifetimes',True),'heap_scrub_free',d.get('heap_scrub_free',True))" "$sdk/sdk.json")"
  # An SDK serving an arm of the manifest has that arm's heap switches; an SDK from before the
  # switches has neither field and was built with both ON.
  if m "arms.$label.heap_scrub_free" > /dev/null 2>&1; then
    for field in heap_lifetimes heap_scrub_free; do
      got=$(python3 -c "import json,sys;print(str(json.load(open(sys.argv[1])).get(sys.argv[2],True)).lower())" "$sdk/sdk.json" "$field")
      [ "$got" = "$(m "arms.$label.$field" | tr 'A-Z' 'a-z')" ] || fail "$label: sdk.json $field is $got, the manifest's arm says $(m "arms.$label.$field")"
    done
  fi
  ( cd "$sdk" && find -L . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$REC/sdk-inventory-$label.sha256"
  note "sdk-inventory $label $(sha256sum "$REC/sdk-inventory-$label.sha256" | cut -c1-64) ($(wc -l < "$REC/sdk-inventory-$label.sha256") files)"
  python3 "$KIT/tools/build-record.py" sdk --sdk "$sdk" --out "$REC/sdk-inputs-$label.json"
done
clang=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['cc'])" "${SDK[${LABELS[0]}]}/sdk.json")
for label in "${LABELS[@]}"; do
  [ "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['cc'])" "${SDK[$label]}/sdk.json")" = "$clang" ] || fail "$label's SDK names another compiler"
done
note "clang $(sha256sum "$(readlink -f "$clang")" | cut -c1-64) $("$clang" --version | head -1)"
# 525.x264_r's input generator for the host.
CC=${HOST_CC:-cc} bash "$KIT/tools/build-host-ldecod.sh" "$SPEC_ROOT" "$REC/ldecod-host"
note "ldecod-host $(sha256sum "$REC/ldecod-host" | cut -c1-64) cc $(${HOST_CC:-cc} --version | head -1)"
# The config runcpu uses is the kit's, byte for byte; the arm enters as --define arm_label and sdk.
cp "$KIT/config/$CFG.cfg" "config/$CFG.cfg"
cp "config/$CFG.cfg" "$REC/$CFG.cfg"
note "config $(sha256sum "config/$CFG.cfg" | cut -c1-64) config/$CFG.cfg"
arms_json="{"
for label in "${LABELS[@]}"; do
  rc=0
  command=(env CAPSTONE_SDK="${SDK[$label]}" runcpu --config="$CFG" --action=build --tune=base --define sdk="${SDK[$label]}" --define kit="$KIT"
           --define arm_label="$label" --define perl_srcalt="$PERL_SRCALT" "${BENCH[@]}")
  python3 - "$REC/build-command-$label.json" "${command[@]}" <<'PYTHON'
import json, sys
with open(sys.argv[1], 'x') as out:
    json.dump(sys.argv[2:], out, indent=2)
    out.write('\n')
PYTHON
  "${command[@]}" > "$REC/runcpu-build-$label.log" 2>&1 || rc=$?
  note "runcpu build $label rc=$rc"
  missing=0; exes="{"
  for b in "${BENCH[@]}"; do
    found=0
    for e in benchspec/CPU/$b/exe/*_base."$label"; do
      [ -f "$e" ] || continue
      found=1; h=$(sha256sum "$e" | cut -c1-64); note "exe $label $h $e"; exes="$exes\"$(basename "$e")\": \"$h\", "
    done
    [ $found = 1 ] || { note "exe $label MISSING $b"; missing=1; }
  done
  exes="${exes%, }}"
  [ $rc = 0 ] && [ $missing = 0 ] || fail "runcpu build $label rc=$rc missing=$missing (see $REC/runcpu-build-$label.log)"
  python3 "$KIT/tools/build-record.py" freeze --spec "$SPEC_ROOT" --label "$label" \
    --benchmarks "${BENCH[@]}" --selection "$REC/source-selection.json" --sdk "${SDK[$label]}" \
    --sdk-before "$REC/sdk-inputs-$label.json" --out "$REC/frozen-$label.json"
  note "frozen $label $(sha256sum "$REC/frozen-$label.json" | cut -c1-64)"
  arms_json="$arms_json\"$label\": {\"sdk\": \"${SDK[$label]}\", \"exe\": $exes}, "
done
echo "${arms_json%, }}" | python3 -m json.tool > "$REC/arms.json"
# Integrity across arms: the objects of every benchmark's build directory, name and sha256, must be
# the same for every label; only the link (the SDK's runtime archive) may differ.
if [ ${#LABELS[@]} -gt 1 ]; then
  for label in "${LABELS[@]}"; do
    for b in "${BENCH[@]}"; do
      d=$({ ls -d "benchspec/CPU/$b/build/build_base_${label}."[0-9][0-9][0-9][0-9] 2>/dev/null || true; } | tail -1)
      [ -n "$d" ] || fail "no build directory for $b $label"
      ( cd "$d" && find . -name '*.o' -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sed "s|^|$b |" )
    done > "$REC/objects-$label.txt"
    note "objects $label $(wc -l < "$REC/objects-$label.txt") files $(sha256sum "$REC/objects-$label.txt" | cut -c1-64)"
    [ -s "$REC/objects-$label.txt" ] || fail "no objects recorded for $label"
  done
  for label in "${LABELS[@]:1}"; do
    if ! cmp -s "$REC/objects-${LABELS[0]}.txt" "$REC/objects-$label.txt"; then
      { diff "$REC/objects-${LABELS[0]}.txt" "$REC/objects-$label.txt" || true; } | head -20 >&2
      fail "objects differ between ${LABELS[0]} and $label (above): the arms do not differ by their SDK alone"
    fi
  done
  note "integrity: every benchmark object identical across ${LABELS[*]}"
fi
# The installed sources and inputs are still SPEC's.
python3 "$KIT/tools/spec-extract.py" --verify "$SPEC_ROOT/.." ALL > "$REC/verify-after-build.txt" 2>&1 \
  || fail "the SPEC tree no longer matches its MANIFEST (see $REC/verify-after-build.txt)"
note "verify-after-build $(tail -1 "$REC/verify-after-build.txt")"
note BUILD_DONE
