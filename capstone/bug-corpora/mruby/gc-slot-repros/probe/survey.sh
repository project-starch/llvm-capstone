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
  "hash-matched-vacated:a54353ecf:test/t/hash.rb"
  "hash-scans-vacated:08a0432d1:test/t/hash.rb"
  "hash-read-back:eb7693857:test/t/hash.rb"
  "hash-delete-in-eql:4663fef45:test/t/hash.rb"
  "hashext-walk-carries:fb4974528:mrbgems/mruby-hash-ext/test/hash.rb"
  "hash-pair-into-set:606d9a6b2:test/t/hash.rb"
  "openter-block:39aecc143:test/t/gc.rb"
  "hash-iter-deleted-ahead:5e8a65457:test/t/hash.rb"
)
# The fixes the control carries. The three left out each depend on an
# intermediate commit and do not apply onto rc2 alone; CANDIDATES.md says so.
CONTROL_FIXES=(5e8a65457 4663fef45 a54353ecf 08a0432d1 fb4974528 859288c19)

mkdir -p "$WORK/cases"

echo "== checking out the pin"
for t in pin ctl; do
  [[ -d $WORK/$t ]] || git -C "$SRC" worktree add -q --detach "$WORK/$t" $PIN || exit 1
  cp "$HERE/probe_config.rb" "$WORK/$t/probe_config.rb"
done

echo "== applying the control's fixes"
for c in "${CONTROL_FIXES[@]}"; do
  printf '   %-12s ' "$c"
  if git -C "$SRC" show "$c" -- src/ mrbgems/mruby-hash-ext/src/ mrbgems/mruby-method/src/ include/ \
     | git -C "$WORK/ctl" apply --whitespace=nowarn 2>/dev/null; then echo applied; else echo "DID NOT APPLY"; fi
done

for t in pin ctl; do
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

printf '\n%-26s %-14s %-14s %-14s %s\n' case pin control pin-stress pin-asan
printf '%-26s %-14s %-14s %-14s %s\n' ------ --- ------- ---------- --------
for spec in "${CASES[@]}"; do
  name=${spec%%:*}; f=$WORK/cases/$name.rb
  asan_out=$(timeout 300 "$WORK/pin/build/asan/bin/mruby" "$f" 2>&1)
  asan=$(echo "$asan_out" | grep -oE 'AddressSanitizer: [a-z-]+' | head -1)
  printf '%-26s %-14s %-14s %-14s %s\n' "$name" \
    "$(verdict "$WORK/pin/build/host/bin/mruby" "$f")" \
    "$(verdict "$WORK/ctl/build/host/bin/mruby" "$f")" \
    "$(verdict "$WORK/pin/build/stress/bin/mruby" "$f")" \
    "${asan:-<silent>}"
done

cat <<'EOF'

A row that fails at the pin and passes in the control is a defect live at the pin,
attributed to the fix the control carries. A row whose ASan column is <silent> is
one the malloc layer cannot see: the slot was vacated and reused, never released.
EOF
