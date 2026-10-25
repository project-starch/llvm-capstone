#!/usr/bin/env bash
# The native matched pair for a C-API driver: the pin, and the pin with only the upstream fix's
# hunks to the case's consumer applied, built natively with the port's own native configuration
# and running the same driver. The pair differs by the fix alone, so a driver that crashes on
# the first and completes on the second reaches the defect that fix removes.
#
#   native-control.sh <pin tree: $MRBD_ROOT/src/mruby> <case dir> <fix diff> <OUT>
#
# <fix diff> is `git show <upstream_fix> -- <consumer>` from an mruby clone. Hunks for code the
# pin does not have are dropped; native-control.sh prints which applied. The trees are copies
# under OUT; the pin tree is not touched. NC_ASAN=1 builds both with AddressSanitizer
# (-O1 -g -fsanitize=address), whose report on the pin names the access.
set -euo pipefail
PIN=${1:?pin tree}; CASE=${2:?case dir}; FIX=${3:?fix diff}; OUT=${4:?OUT}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$HERE/../../../../ports/mruby/app/build_config.rb
trig=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["trigger"])' "$CASE/case.json")
[[ $trig == *.c ]] || { echo "native-control: $CASE's trigger is not a C driver" >&2; exit 2; }
mkdir -p "$OUT"
# The port's native build alone: its first MRuby::Build block, without the cross build that
# needs the Capstone SDK.
python3 - "$CONFIG" "$OUT/native_config.rb" <<'EOF'
import sys
text = open(sys.argv[1]).read()
open(sys.argv[2], "w").write(text[:text.index("MRuby::CrossBuild.new")])
EOF
san=()
if [[ ${NC_ASAN:-0} == 1 ]]; then
  export MRBD_OPT="-O1 -g -fsanitize=address -fno-omit-frame-pointer"
  san=(-g -fsanitize=address -fno-omit-frame-pointer)
  printf 'MRuby.each_target { |t| t.linker.flags << "-fsanitize=address" }\n' >> "$OUT/native_config.rb"
fi
for arm in pin fix; do
  T=$OUT/$arm
  rm -rf "$T"; mkdir -p "$T"
  (cd "$PIN" && tar --exclude=./build -cf - .) | tar -xf - -C "$T"
  if [[ $arm == fix ]]; then
    # Hunk by hunk: a hunk whose context the pin lacks fails alone and is reported.
    (cd "$T" && patch -p1 --forward --no-backup-if-mismatch -r - < "$FIX" || true) | sed 's/^/native-control: /'
    consumer=$(sed -n 's|^+++ b/||p' "$FIX" | head -1)
    ! cmp -s "$PIN/$consumer" "$T/$consumer" || { echo "native-control: no hunk of $FIX applied" >&2; exit 2; }
  fi
  (cd "$T" && MRUBY_CONFIG="$OUT/native_config.rb" rake -j"${JOBS:-8}" all) > "$OUT/$arm-rake.log" 2>&1 \
    || { tail -5 "$OUT/$arm-rake.log" >&2; exit 2; }
  B=$T/build/native
  cc -O1 "${san[@]}" -std=gnu99 -DPOOL_ALIGNMENT=16 -DMRB_NO_DIRECT_THREADING -DMRB_NO_BOXING \
    -I"$T/include" -I"$B/include" -I"$T/mrbgems/mruby-task/include" \
    "$CASE/$trig" "$B/lib/libmruby.a" -lm -o "$OUT/capi-$arm"
  set +e
  "$OUT/capi-$arm" > "$OUT/capi-$arm.out" 2>&1
  rc=$?
  set -e
  echo "native-control: $arm exit $rc: $(tr '\n' '|' < "$OUT/capi-$arm.out")"
done
