#!/bin/bash
# Build the measurement SDKs of manifest.lock.json `arms` from the runtime tree: the toolchain's source
# tree plus `runtime.patch` (PR #279's heap switches), against the toolchain's pinned compiler, prepared
# musl source and libc archive. The compiler is not rebuilt: `runtime.unchanged_subtrees` (checked by
# recipes/integration.sh in git) say the patch leaves llvm, clang, lld, compiler-rt and the musl port
# as the toolchain commit has them. Checks the patch and every patched file against the manifest, then
# proves that the SDKs differ in heap.c.obj only.
# usage: sdks.sh KIT TOOLCHAIN WORKDIR
#   TOOLCHAIN  recipes/toolchain.sh's directory (build/, musl/, musl-src/, sdk/, src/, toolchain-record.txt)
#   WORKDIR    empty or absent; gets src/, one SDK directory per arm, tmp/ and sdks-record.txt
set -euo pipefail
KIT=$(cd "$1" && pwd) TC=$(cd "$2" && pwd) W=$3
[ -e "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ] && { echo "sdks: $W is not empty" >&2; exit 2; }
mkdir -p "$W"; W=$(cd "$W" && pwd)
M=$KIT/manifest.lock.json
m() { python3 -c "import json,sys;d=json.load(open(sys.argv[1]));[d:=d[k] for k in sys.argv[2].split('.')];print(d)" "$M" "$1"; }
R=$W/sdks-record.txt; : > "$R"
note() { echo "$*" | tee -a "$R"; }
fail() { echo "sdks: $*" >&2; note "FAILED $*"; exit 1; }
check() { local got; got=$(sha256sum "$1" | cut -c1-64); [ "$got" = "$2" ] || fail "$1 is $got, manifest says $2"; }
grep -q "^llvm-capstone commit $(m toolchain.commit) " "$TC/toolchain-record.txt" || fail "$TC is not the manifest's toolchain"
# The runtime tree is the toolchain's own when the manifest names no patch (the switches are in the
# toolchain commit); otherwise the toolchain source plus `runtime.patch`.
PATCH=$(python3 -c "import json,sys;r=json.load(open(sys.argv[1]))['runtime'];print(r.get('patch',{}).get('file',''))" "$M")
[ -z "$PATCH" ] || { PATCH=$KIT/$PATCH; check "$PATCH" "$(m runtime.patch.sha256)"; }
SRC=$TC/src/llvm-capstone; [ -d "$SRC/capstone/runtime/virtual" ] || fail "no toolchain source at $SRC"
note "toolchain $TC commit $(m toolchain.commit)"
note "runtime commit $(m runtime.commit) tree $(m runtime.tree) patch ${PATCH:+$(m runtime.patch.sha256)}${PATCH:-none}"
# The runtime tree: the subtrees the SDK build reads, copied from the toolchain source, plus the patch.
T=$W/src/llvm-capstone; mkdir -p "$T/llvm" "$T/capstone/docs/plans"
(cd "$SRC" && tar cf - capstone/runtime capstone/ports/common capstone/ports/musl-capstone capstone/sublet \
  capstone/my_first_domain capstone/tests/capstone-test-env.sh capstone/tests/toolchain-fresh.py compiler-rt/lib/builtins) | tar -C "$T" -xf -
if [ -n "$PATCH" ]; then
  patch -d "$T" -p1 -N -s -f < "$PATCH" || fail "the runtime patch does not apply to the toolchain source"
fi
while read -r path want; do
  [ -f "$T/$path" ] || fail "runtime file $path is missing"
  check "$T/$path" "$want"
done < <(python3 -c "import json,sys;[print(p,h) for p,h in json.load(open(sys.argv[1]))['runtime'].get('files',{}).items()]" "$M")
note "runtime files $(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['runtime'].get('files',{})))" "$M") checked"
# Same compiler, prepared musl source and libc archive as the toolchain's SDK.
MUSL=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['musl'])" "$TC/sdk/sdk.json")
LIBC=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['libc'])" "$TC/sdk/sdk.json")
[ -d "$MUSL" ] && [ -f "$LIBC" ] || fail "the toolchain's musl source or libc archive is missing"
note "libc $(sha256sum "$LIBC" | cut -c1-64)"
export CAPSTONE_REPO_ROOT=$T CAPSTONE_LLVM_BUILD_DIR=$TC/build CAPSTONE_TMP_ROOT=$W/tmp CAPSTONE_APPLICATION_PROFILE=virtual
mkdir -p "$W/tmp"
mapfile -t LABELS < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['arms']))" "$M")
for label in "${LABELS[@]}"; do
  dir=$(m "arms.$label.sdk")
  mapfile -t opts < <(python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['options']))" "$M" "$label")
  ( cd "$T" && source capstone/tests/capstone-test-env.sh > /dev/null 2>&1 &&
    bash capstone/ports/common/application/build-sdk.sh "$W/$dir" "$MUSL" "$LIBC" "${opts[@]}" ) > "$W/$dir.log" 2>&1 \
    || fail "SDK $label did not build (see $W/$dir.log)"
  for field in heap_lifetimes heap_scrub_free; do
    got=$(python3 -c "import json,sys;print(str(json.load(open(sys.argv[1]))[sys.argv[2]]).lower())" "$W/$dir/sdk.json" "$field")
    [ "$got" = "$(m "arms.$label.$field" | tr 'A-Z' 'a-z')" ] || fail "$label: sdk.json $field is $got"
  done
  ( cd "$W/$dir" && find -L . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > "$W/$dir-inventory.sha256"
  note "sdk $label $dir heap_lifetimes=$(m "arms.$label.heap_lifetimes") heap_scrub_free=$(m "arms.$label.heap_scrub_free") options=[${opts[*]:-}]"
  note "sdk-inventory $label $(sha256sum "$W/$dir-inventory.sha256" | cut -c1-64) ($(wc -l < "$W/$dir-inventory.sha256") files)"
  note "sdk-linked $label $(cd "$W/$dir" && sha256sum libapplication-runtime.a libcapstone-application-builtins.a capstone-cc | awk '{printf "%s=%s ", $2, substr($1,1,16)}')"
done
# Proof: against the first arm, the builtins archive and capstone-cc are identical and the runtime
# archive differs in heap.c.obj only; the same against the toolchain's own SDK.
AR=$TC/build/bin/llvm-ar
members() { rm -rf "$W/ar-$2"; mkdir -p "$W/ar-$2"; (cd "$W/ar-$2" && "$AR" x "$1/libapplication-runtime.a" && sha256sum ./*.obj | sort -k2 > "$W/members-$2.txt"); }
first=$(m "arms.${LABELS[0]}.sdk")
members "$W/$first" "$first"; members "$TC/sdk" toolchain
for label in "${LABELS[@]}"; do
  dir=$(m "arms.$label.sdk"); members "$W/$dir" "$dir"
  for other in "$first" toolchain; do
    [ "$dir" = "$other" ] && continue
    # (diff exits 1 when the archives differ and grep exits 1 when they do not; under pipefail neither may end the recipe)
    differ=$({ diff "$W/members-$other.txt" "$W/members-$dir.txt" || true; } | { grep -E '^[<>]' || true; } | awk '{print $3}' | sort -u | tr '\n' ' ')
    note "members $dir vs $other: ${differ:-none}"
    [ "$differ" = "./heap.c.obj " ] || [ -z "$differ" ] || fail "$dir differs from $other in more than heap.c.obj: $differ"
  done
  if [ "$dir" != "$first" ]; then
    for f in libcapstone-application-builtins.a capstone-cc; do
      [ "$(sha256sum "$W/$dir/$f" | cut -c1-64)" = "$(sha256sum "$W/$first/$f" | cut -c1-64)" ] || fail "$dir: $f differs from $first's"
    done
  fi
done
note "proof: builtins archive and capstone-cc identical across arms; runtime archives differ in heap.c.obj only"
note SDKS_DONE
