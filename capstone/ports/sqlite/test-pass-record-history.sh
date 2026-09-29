#!/usr/bin/env bash
# Test the pass-record writer's new prepend behaviour WITHOUT spending QEMU runs: the block is
# extracted verbatim from run-speedtest1-measure.sh and driven with synthetic values.
set -euo pipefail
T=$(mktemp -d); QEMU_PASS_DIR=$T; fails=0
say() { printf '  %-58s %s\n' "$1" "$2"; }
chk() { if [ "$2" = "$3" ]; then say "$1" "PASS"; else say "$1" "FAIL (got '$2', wanted '$3')"; fails=$((fails+1)); fi; }

write_rec() { # write_rec <sha> <args> <log>
  local _dom_sha=$1 SPEEDTEST1_ARGS=$2 OUT_DIR=$3 SPEEDTEST1_CMA_MB=none SPEEDTEST1_LOG_FILE=$3/x.log
  local _rec=$QEMU_PASS_DIR/$_dom_sha _prev=""
  if [ -f "$_rec" ]; then _prev=$(tail -n +2 -- "$_rec"); fi
  {
    printf '%s  sqlite_silicon.dom\n' "$_dom_sha"
    printf 'args=%s\ncma_mb=%s\nlog=%s\nwhen=%s\n' "$SPEEDTEST1_ARGS" \
      "${SPEEDTEST1_CMA_MB:-none}" "${SPEEDTEST1_LOG_FILE:-$OUT_DIR/sqlite-speedtest1.log}" "$(date -u +%FT%TZ)"
    if [ -n "$_prev" ]; then printf '\n%s\n' "$_prev"; fi
  } > "$_rec.tmp$$" && mv -f "$_rec.tmp$$" "$_rec"
}

SHA=$(printf 'x' | sha256sum | cut -d' ' -f1)
R=$QEMU_PASS_DIR/$SHA

echo "== run 1 (fresh) =="
write_rec "$SHA" "--testset main --size 1 --verify" /canon
chk "file exists (what 6 drivers test)" "$([ -f "$R" ] && echo yes)" "yes"
chk "C16 first args= is the run's" "$(grep -m1 '^args=' "$R")" "args=--testset main --size 1 --verify"
chk "exactly one hash line" "$(grep -c "^$SHA" "$R")" "1"
chk "one args= block" "$(grep -c '^args=' "$R")" "1"

echo "== a curated note= is added by hand =="
printf 'note=curated: log points into the staged folder\n' >> "$R"
chk "note present" "$(grep -c '^note=' "$R")" "1"

echo "== run 2 (an exploratory --stats run over the SAME image) =="
write_rec "$SHA" "--testset main --size 1 --verify --stats" /scratch
chk "C16 first args= is the NEWEST run" "$(grep -m1 '^args=' "$R")" "args=--testset main --size 1 --verify --stats"
chk "still exactly one hash line" "$(grep -c "^$SHA" "$R")" "1"
chk "both runs present" "$(grep -c '^args=' "$R")" "2"
chk "the curated note SURVIVED" "$(grep -c '^note=' "$R")" "1"
chk "the old canonical log= survived" "$(grep -c 'log=/canon/' "$R")" "1"

echo "== run 3, idempotence =="
write_rec "$SHA" "--testset main --size 20 --verify" /third
chk "three runs present" "$(grep -c '^args=' "$R")" "3"
chk "still one hash line" "$(grep -c "^$SHA" "$R")" "1"
chk "newest still first" "$(grep -m1 '^args=' "$R")" "args=--testset main --size 20 --verify"
chk "note still there" "$(grep -c '^note=' "$R")" "1"

echo "== NEGATIVE CONTROL: the OLD truncating writer loses all of it =="
printf '%s  sqlite_silicon.dom\nargs=%s\n' "$SHA" "clobbered" > "$R"
chk "old writer: note gone" "$(grep -c '^note=' "$R")" "0"
chk "old writer: only one run left" "$(grep -c '^args=' "$R")" "1"

echo
rm -rf "$T"
[ "$fails" -eq 0 ] && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
