#!/bin/bash
# Reproduce CANDIDATES.md: build the pin three ways plus a control, extract each
# candidate's own upstream test from the fix commit that added it, and run the
# differential. Native x86-64 only -- no toolchain, no QEMU, no board.
#
#   MRUBY_SRC=<a clone of mruby with full history>  ./survey.sh [workdir]
#
# The clone must reach both pins and the fix commits, so it cannot be shallow:
# `git clone https://github.com/mruby/mruby.git` or an existing clone with
# `git fetch --unshallow`.
set -u

PIN=9d523e2f74f2e63ca02840937523de61398a617d   # 4.0.0-rc2, the port's pin
HEAD_PIN=ad98f216eb472202c8e5deece5ea13655d9f7969  # the port's 'head' pin

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-${TMPDIR:-/tmp}/mruby-slot-survey}
SRC=${MRUBY_SRC:-}

[[ -n $SRC && -d $SRC/.git ]] || { echo "set MRUBY_SRC to an mruby clone" >&2; exit 2; }
[[ $(git -C "$SRC" rev-parse --is-shallow-repository) == false ]] \
  || { echo "$SRC is shallow; git -C $SRC fetch --unshallow" >&2; exit 2; }
for c in $PIN $HEAD_PIN; do
  git -C "$SRC" cat-file -e "$c" 2>/dev/null || { echo "$SRC does not have $c" >&2; exit 2; }
done
for t in gcc ruby rake bison; do
  command -v $t >/dev/null || { echo "$t is required" >&2; exit 2; }
done

# case name : the fix commit that added the test : the file it added it to
CASES=(
  "string-strip-bang-uaf:1c57532b2:mrbgems/mruby-string-ext/test/string.rb"
  "hash-matched-vacated:a54353ecf:test/t/hash.rb"
  "hash-scans-vacated:08a0432d1:test/t/hash.rb"
  "hash-read-back:eb7693857:test/t/hash.rb"
  "hash-delete-in-eql:4663fef45:test/t/hash.rb"
  "hashext-walk-carries:fb4974528:mrbgems/mruby-hash-ext/test/hash.rb"
  "hash-pair-into-set:606d9a6b2:test/t/hash.rb"
  "openter-block:39aecc143:test/t/gc.rb"
  "iv-walk-freed-block:0cf969a2b:test/t/kernel.rb"
  "hash-iter-deleted-ahead:5e8a65457:test/t/hash.rb"
  "ary-splice-self-aset:17d124b00:test/t/array.rb"
  "string-prepend-self:af6f23ddb:mrbgems/mruby-string-ext/test/string.rb"
)
# The fixes the control carries. The three left out each depend on an
# intermediate commit and do not apply onto rc2 alone; CANDIDATES.md says so.
CONTROL_FIXES=(5e8a65457 4663fef45 a54353ecf 08a0432d1 fb4974528 859288c19)
# ctl2 carries this one alone: it is three lines and needs no predecessor.
CONTROL2_FIXES=(1c57532b2)

mkdir -p "$WORK/cases"

echo "== checking out the pin"
for t in pin ctl ctl2; do
  [[ -d $WORK/$t ]] || git -C "$SRC" worktree add -q --detach "$WORK/$t" $PIN || exit 1
  cp "$HERE/probe_config.rb" "$WORK/$t/probe_config.rb"
done

echo "== applying the control's fixes"
apply_into() {  # <tree> <commit>...
  local tree=$1; shift
  for c in "$@"; do
    printf '   %-8s -> %-5s ' "$c" "$tree"
    if git -C "$SRC" show "$c" -- src/ mrbgems/mruby-hash-ext/src/ mrbgems/mruby-method/src/ \
         mrbgems/mruby-string-ext/src/ include/ \
       | git -C "$WORK/$tree" apply --whitespace=nowarn 2>/dev/null; then echo applied; else echo "DID NOT APPLY"; fi
  done
}
apply_into ctl  "${CONTROL_FIXES[@]}"
apply_into ctl2 "${CONTROL2_FIXES[@]}"

for t in pin ctl ctl2; do
  echo "== building $t"
  ( cd "$WORK/$t" && MRUBY_CONFIG="$WORK/$t/probe_config.rb" rake -j"$(nproc)" ) >"$WORK/$t.build.log" 2>&1 \
    || { echo "build failed, see $WORK/$t.build.log" >&2; exit 1; }
done

echo "== extracting each candidate's own upstream test"
for spec in "${CASES[@]}"; do
  name=${spec%%:*}; rest=${spec#*:}; commit=${rest%%:*}; file=${rest#*:}
  out=$WORK/cases/$name.rb
  cat "$HERE/harness.rb" > "$out"
  git -C "$SRC" show "$commit" -- "$file" | grep -E '^\+' | grep -v '^+++' | sed 's/^+//' >> "$out"
  cat "$HERE/footer.rb" >> "$out"
done

verdict() {  # <binary> <case file>
  local out rc
  out=$(timeout 300 "$1" "$2" 2>&1); rc=$?
  if   [[ $rc -ge 128 ]]; then echo "CRASH($rc)"
  elif echo "$out" | grep -q '\["PASS"\]'; then echo "PASS"
  else echo "FAIL($(echo "$out" | grep -c '^\['))"; fi
}

asan_of() { timeout 300 "$1" "$2" 2>&1 | grep -oE 'AddressSanitizer: [a-z-]+' | head -1; }

printf '\n%-26s %-11s %-11s %-11s %-11s %-18s %s\n' \
  case pin ctl ctl2 stress asan asan-page1
printf '%-26s %-11s %-11s %-11s %-11s %-18s %s\n' \
  ------ --- --- ---- ------ ---- ----------
for spec in "${CASES[@]}"; do
  name=${spec%%:*}; f=$WORK/cases/$name.rb
  a=$(asan_of  "$WORK/pin/build/asan/bin/mruby" "$f")
  a1=$(asan_of "$WORK/pin/build/asan-page1/bin/mruby" "$f")
  printf '%-26s %-11s %-11s %-11s %-11s %-18s %s\n' "$name" \
    "$(verdict "$WORK/pin/build/host/bin/mruby" "$f")" \
    "$(verdict "$WORK/ctl/build/host/bin/mruby" "$f")" \
    "$(verdict "$WORK/ctl2/build/host/bin/mruby" "$f")" \
    "$(verdict "$WORK/pin/build/stress/bin/mruby" "$f")" \
    "${a:-<silent>}" "${a1:-<silent>}"
done

cat <<'EOF'

A row that fails at the pin and passes in the control is a defect live at the pin,
attributed to the fix the control carries. A row whose ASan column is <silent> is
one the malloc layer cannot see. Silent in asan and loud in asan-page1 means a GC slot was
reused inside a standing page; silent in both means the reuse never reaches the collector.
EOF
