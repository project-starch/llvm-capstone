#!/usr/bin/env bash
# BLOCKED case for lua-hiredis. This is not a pass/fail bug reproduction: it runs
# the strongest attempt at the described CDP shape under ASan and shows it is
# SAFE (reply is a Lua copy; context free is null-guarded), then exits 3=BLOCKED
# with the reason. `./build.sh && ./run.sh` is intentionally non-zero: there is
# no filed hard-tier CDP UAF in lua-hiredis to reproduce.
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); LC=$(cd "$HERE/.."; pwd); W="$HERE/.work"
LUADIR="$LC/_toolchain/.work/lua54"; ASAN=$(cc -print-file-name=libasan.so)
[ -f "$W/hiredis.so" ] || { echo "run ./build.sh first" >&2; exit 2; }
command -v redis-server >/dev/null || { echo "need redis-server" >&2; exit 2; }

SOCK="$W/redis.sock"; LOG="$W/redis.log"; OUT="$W/trigger.out"; rm -f "$SOCK"
redis-server --port 0 --unixsocket "$SOCK" --save '' --daemonize no --loglevel warning >"$LOG" 2>&1 &
RPID=$!
cleanup(){ redis-cli -s "$SOCK" shutdown nosave >/dev/null 2>&1; kill "$RPID" >/dev/null 2>&1; wait "$RPID" 2>/dev/null; rm -f "$SOCK"; }
trap cleanup EXIT
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.1; done
[ -S "$SOCK" ] || { echo "redis did not start (see $LOG)" >&2; exit 2; }

SOCK="$SOCK" LD_PRELOAD="$ASAN" LD_LIBRARY_PATH="$LUADIR" LUA_CPATH="$W/?.so;;" \
  ASAN_OPTIONS="detect_leaks=0:abort_on_error=0" "$LUADIR/lua-shared" "$HERE/trigger.lua" >"$OUT" 2>&1
echo "--- trigger output ---"; cat "$OUT"

reply_copy=0; grep -qa 'OK-reply-is-a-lua-copy'      "$OUT" && reply_copy=1
ctx_safe=0;   grep -qa 'OK-context-not-double-freed'  "$OUT" && ctx_safe=1
no_uaf=1;     grep -qa -E 'heap-use-after-free|AddressSanitizer' "$OUT" && no_uaf=0
echo "--- safe-by-construction: reply_copy=$reply_copy ctx_safe=$ctx_safe no_asan_report=$no_uaf ---"

if [ "$reply_copy" = 1 ] && [ "$ctx_safe" = 1 ] && [ "$no_uaf" = 1 ]; then
  echo "BLOCKED: lua-hiredis has no filed hard-tier Lua<->C CDP UAF."
  echo "  The redisReply* is deep-copied into Lua values + freeReplyObject'd in-call"
  echo "  (never crosses the boundary as a handle); the redisContext* userdata has a"
  echo "  null-guarded __gc==close. Nothing to reproduce -- reassign the slot. See CASE.md."
  exit 3
fi
echo "UNEXPECTED: safety demo did not hold as documented (see $OUT); do not treat as PASS." >&2
exit 1
