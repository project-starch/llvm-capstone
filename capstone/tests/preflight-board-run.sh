#!/usr/bin/env bash
# Board-run preflight gate. exit 0 = GO, exit 1 = BLOCKED.
#
# A SCRIPT, not a subagent, and deliberately so: like precommit-scan.sh, a gate that decides
# whether to spend a scarce shared resource must be deterministic. Never delegate it, and never
# weaken a check to make a run start.
#
# Every check here corresponds to a failure that actually cost board time:
#   C1 construct-in-artifact  -- CAPSTONE_SQLITE_STAGE was never compiled in, so every dom:NNN
#                                selector was ignored and the domain ran the full workload.
#                                Cost ~8 runs on 2026-08-06 and 3 earlier sessions.
#   C2 distinct images        -- three "different" images were byte-identical; the flags did not
#                                reach the amalgamation.
#   C3 oracle present         -- a rung with no oracle cannot be scored.
#   C4 control has a record   -- matmult_int was used as "the known-good control"; it is a
#                                DOCUMENTED silicon miscompile. That single error produced the
#                                retracted bitstream-regression claim.
#   C5 DTS matches bitstream  -- a firmware built for the other bitstream hands Linux shadow-tag
#                                memory as RAM and dies in early init, looking like a dead board.
#   C6 firmware freshness     -- relinking without re-staging ships the previous initramfs.
#   C7 slot budget            -- split_out_cap's middle exact-fit case spins at ~the 5th
#                                create_dom, and whatever occupies that slot gets blamed.
set -uo pipefail
FAIL=0
say() { printf "  %-6s %s\n" "$1" "$2"; }
bad() { say "BLOCK" "$1"; FAIL=1; }
ok()  { say "ok" "$1"; }

FW=${FPGA_FW:-}
BITSTREAM=${FPGA_BITSTREAM:-}
RUNGS=${BAKED_RUNGS:-}
DOMS=${SQLITE_STAGE_DOMS:-}
OVERLAY=${PREFLIGHT_OVERLAY:-capstone/caplifive-system/sw/buildroot/overlay/test-domains}
ORACLES=${PREFLIGHT_ORACLES:-/tmp/capstone/ladder-fpga}
CONTROLS=capstone/agent-handoff/ref/known-good-controls.md

echo "=== preflight-board-run ==="

# C5/C6 -------------------------------------------------------------------------------------
[[ -n "$FW" && -f "$FW" ]] || bad "FPGA_FW unset or missing -- an implicit firmware boots whatever was built last"
[[ -n "$BITSTREAM" ]] || bad "FPGA_BITSTREAM unset -- the resident-silicon guard cannot fire"
if [[ -f "$FW" ]]; then
  have_1021=$(python3 -c "print(open('$FW','rb').read().count(bytes.fromhex('000000003c3c0000')))")
  have_64k=$(python3 -c "print(open('$FW','rb').read().count(bytes.fromhex('000000003c2d2000')))")
  case "$BITSTREAM" in
    *fixed_forward*) [[ "$have_1021" -ge 1 && "$have_64k" -eq 0 ]] \
        && ok "DTS matches $BITSTREAM (0x3c3c0000)" \
        || bad "DTS/bitstream MISMATCH: fixed_forward needs 0x3c3c0000 (found 1021=$have_1021 64k=$have_64k)" ;;
    *65536*)         [[ "$have_64k" -ge 1 && "$have_1021" -eq 0 ]] \
        && ok "DTS matches $BITSTREAM (0x3c2d2000)" \
        || bad "DTS/bitstream MISMATCH: 65536_nodes needs 0x3c2d2000 (found 1021=$have_1021 64k=$have_64k)" ;;
    *) say "warn" "unknown bitstream $BITSTREAM -- DTS pairing unchecked" ;;
  esac
  python3 -c "
import sys
d=open('$FW','rb').read()
sys.exit(0 if d.count(bytes.fromhex('d00dfeed'))==1 else 1)" \
    && ok "firmware embeds exactly one FDT" || bad "firmware does not embed exactly one FDT"

  # C8 -- the console REJECTS an image over 32 MB, and it does so only after the board has
  # been locked and power-cycled, so an oversize image costs a full boot slot and returns
  # no verdict at all:
  #     ActionError: images/upload -> HTTP 413: 'File exceeds 32 MB limit'
  # That is exactly what happened on 2026-08-06 at 34.2 MB. The cause is silent accumulation:
  # each staged SQLite domain is ~1.5 MB, six of them had piled up in overlay/test-domains,
  # and nothing in the build warns. Prune stale domains by EXPLICIT NAME -- never a glob; a
  # prefix glob once deleted the package-installed sbi.dom.
  FW_BYTES=$(stat -c%s "$FW")
  FW_LIMIT=$((32 * 1024 * 1024))
  if (( FW_BYTES > FW_LIMIT )); then
    bad "firmware is $((FW_BYTES / 1024 / 1024)) MB; the console rejects anything over 32 MB (HTTP 413) AFTER locking and power-cycling the board, so this run would burn a slot and return nothing. Prune overlay/test-domains by explicit name and rebuild."
  else
    ok "firmware $((FW_BYTES / 1024 / 1024)) MB, under the console's 32 MB upload limit"
  fi
fi

# C7 ----------------------------------------------------------------------------------------
n=0
[[ -n "$RUNGS" ]] && n=$(wc -w <<<"$RUNGS")
[[ -n "$DOMS"  ]] && n=$(awk -F, '{print NF}' <<<"$DOMS")
if [[ "$n" -gt 4 ]]; then
  bad "$n domains requested; the monitor's middle exact-fit case spins at ~the 5th create_dom, so slots 5+ carry NO verdict (set PREFLIGHT_ALLOW_SLOTS=1 to override deliberately)"
  [[ "${PREFLIGHT_ALLOW_SLOTS:-0}" == "1" ]] && { FAIL=0; say "warn" "slot budget overridden on purpose"; }
elif [[ "$n" -gt 0 ]]; then ok "$n domains -- within the 4-slot budget"; fi

# C2/C3/C4 ----------------------------------------------------------------------------------
if [[ -n "$RUNGS" ]]; then
  first=$(awk '{print $1}' <<<"$RUNGS")
  # Match ONLY the known-good table, which ends at the "## NOT controls" heading. The first
  # version of this grepped the whole file and therefore matched matmult_int in the DO-NOT-USE
  # table, returning GO for the exact mistake this gate exists to catch. A gate that cannot
  # fail its own negative test is worse than no gate: it grants false confidence.
  if [[ -f "$CONTROLS" ]] && sed '/^## NOT controls/,$d' "$CONTROLS" | grep -qE "^\| *\`?$first\`? "; then
    ok "control '$first' has a published passing record"
  else
    bad "first rung '$first' is not listed in $CONTROLS as a known-good control -- a boot whose control is invalid carries no verdict (matmult_int is a documented MISCOMPILE, not a control)"
  fi
  shas=""
  for r in $RUNGS; do
    [[ -f "$OVERLAY/$r.dom" ]] || { bad "$r.dom not staged in $OVERLAY"; continue; }
    [[ -f "$ORACLES/$r.oracle" ]] || bad "$r has no oracle in $ORACLES"
    shas+="$(sha256sum "$OVERLAY/$r.dom" | cut -d' ' -f1)\n"
  done
  u=$(printf "$shas" | sort -u | grep -c .); t=$(printf "$shas" | grep -c .)
  [[ "$u" -eq "$t" ]] && ok "$t staged images, all distinct" \
                      || bad "only $u distinct images among $t -- identical images cannot give different results"
fi

# C1 ----------------------------------------------------------------------------------------
if [[ -n "${PREFLIGHT_CONSTRUCT:-}" && -n "${PREFLIGHT_CONSTRUCT_DOM:-}" ]]; then
  if grep -qa "$PREFLIGHT_CONSTRUCT" "$PREFLIGHT_CONSTRUCT_DOM" 2>/dev/null; then
    ok "construct '$PREFLIGHT_CONSTRUCT' present in $(basename "$PREFLIGHT_CONSTRUCT_DOM")"
  else
    bad "construct '$PREFLIGHT_CONSTRUCT' NOT in $(basename "$PREFLIGHT_CONSTRUCT_DOM") -- gate the ARTIFACT, never the flags"
  fi
else
  say "warn" "PREFLIGHT_CONSTRUCT unset -- artifact not checked for the construct under test"
fi

echo "=========================================="
[[ "$FAIL" -eq 0 ]] && { echo "preflight: GO"; exit 0; } || { echo "preflight: BLOCKED"; exit 1; }
