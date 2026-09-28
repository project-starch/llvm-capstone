#!/bin/bash
# How much of the survey depends on the pin. Builds each version named below and runs
# the cases survey.sh extracted against it, so "would another version give us these?"
# is answered by running rather than by reading the history.
#
#   MRUBY_SRC=<a full clone>  ./versions.sh <survey.sh's workdir> [tag...]
#
# survey.sh must have run first: this reuses its $WORK/cases. Ancestry of a fix commit
# is NOT used, deliberately -- it disagreed with the runs once (row 5 at 4.1.0-rc2),
# because an equivalent change can reach a tag by another route.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:?usage: versions.sh <survey workdir> [tag...]}
shift
TAGS=("$@")
[[ ${#TAGS[@]} -gt 0 ]] || TAGS=(3.4.0 4.0.0-rc2 4.0.0 4.1.0-rc2)
SRC=${MRUBY_SRC:-}

[[ -n $SRC && -d $SRC/.git ]] || { echo "set MRUBY_SRC to an mruby clone" >&2; exit 2; }
[[ -d $WORK/cases ]] || { echo "$WORK/cases is missing -- run survey.sh first" >&2; exit 2; }

# Only host and asan: the version question needs a verdict and a sanitizer reading,
# not the stress or page-size arms, which answer a different question.
cat > "$WORK/vercfg.rb" <<'CFG'
gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gem :core => 'mruby-bin-mruby'
end
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG)
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end
MRuby::Build.new('asan') do |conf|
  conf.toolchain :gcc
  conf.cc.flags += %w(-fsanitize=address -fno-omit-frame-pointer -g -O1)
  conf.linker.flags += %w(-fsanitize=address)
  gems.call(conf)
end
CFG

for t in "${TAGS[@]}"; do
  d=$WORK/v-$t
  if [[ ! -x $d/build/host/bin/mruby ]]; then
    echo "== building $t"
    [[ -d $d ]] || git -C "$SRC" worktree add -q --detach "$d" "$t" || exit 1
    cp "$WORK/vercfg.rb" "$d/probe_config.rb"
    # the 4.1 line compiles through Prism, which is a submodule
    git -C "$d" submodule update -q --init mrbgems/mruby-compiler/lib/prism 2>/dev/null
    ( cd "$d" && MRUBY_CONFIG="$d/probe_config.rb" rake -j"$(nproc)" ) >"$WORK/v-$t.log" 2>&1 \
      || { echo "   $t DID NOT BUILD, see $WORK/v-$t.log"; continue; }
  fi
done

# A case whose failures include a NoMethodError/NameError is reporting a missing
# method, not a defect: on an older version that is a feature gap and must not be
# counted as a reproduction. It is called out rather than folded into the count.
verdict() {
  local o r; o=$(timeout 300 "$1" "$2" 2>&1); r=$?
  if   [[ $r -ge 128 ]]; then echo "CRASH($r)"
  elif echo "$o" | grep -q '\["PASS"\]'; then echo "PASS"
  elif echo "$o" | grep -qE 'NoMethodError|NameError'; then echo "FAIL+gap"
  else echo "FAIL($(echo "$o" | grep -c '^\['))"; fi
}
asan_of() { timeout 300 "$1" "$2" 2>&1 | grep -oE 'AddressSanitizer: [a-z-]+' | head -1 | sed 's/AddressSanitizer: //'; }

printf '\n%-24s' "case"
for t in "${TAGS[@]}"; do printf '%-26s' "$t"; done; echo
printf '%-24s' "----"
for t in "${TAGS[@]}"; do printf '%-26s' "-------------------------"; done; echo

for f in "$WORK"/cases/*.rb; do
  n=$(basename "$f" .rb)
  printf '%-24s' "$n"
  for t in "${TAGS[@]}"; do
    h=$WORK/v-$t/build/host/bin/mruby
    if [[ -x $h ]]; then
      a=$(asan_of "$WORK/v-$t/build/asan/bin/mruby" "$f")
      printf '%-26s' "$(verdict "$h" "$f")/${a:-silent}"
    else
      printf '%-26s' "-"
    fi
  done; echo
done

cat <<'EOF'

PASS in a column means the defect is not live in that version -- either fixed by then
or not yet written. The two are told apart by asking whether the code exists at all:
git grep <the function the fix touches> <tag>. FAIL+gap means the case hit a missing
method on that version and says nothing about the defect.
EOF
