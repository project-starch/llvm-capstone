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

# Domain count, hoisted: C11 and C12 both reference it and run before C7 computes it.
n=0
[[ -n "$RUNGS" ]] && n=$(wc -w <<<"$RUNGS")
[[ -n "$DOMS"  ]] && n=$(awk -F, '{print NF}' <<<"$DOMS")

# C9 -- OVERLAY CLEANLINESS. -----------------------------------------------------------------
# The overlay grew to 111 files / 35 MB over one session because every probe was staged and
# none retired. Measured cost: 30.6 minutes of pure JTAG upload across 39 boots, at a rock-solid
# 128.2 KiB/s, plus one boot lost outright to HTTP 413 -- and that rejection came from being
# 689,160 bytes over, i.e. less than ONE stale SQLite domain. Retiring 99 files cut the firmware
# by exactly 10 MiB = 80 s off every boot.
OVERLAY_KEEP_ALWAYS="sbi.dom sbi.smode smode.dom smode.smode thread.dom fib.dom lpc sqlite_host.user k800.dom q31.dom"
if [[ -d "$OVERLAY" ]]; then
  _wanted=" $OVERLAY_KEEP_ALWAYS "
  # BAKED_RUNGS names rungs WITHOUT the .dom suffix ("lf0"), while SQLITE_STAGE_DOMS gives full
  # paths ("/test-domains/q31.dom", optionally "host args|path:selector"). Accept both spellings
  # or the gate flags the very domains the run needs -- which it did on the first dry run.
  for _d in ${RUNGS} $(tr ',' ' ' <<<"$DOMS"); do
    _b="$(basename "${_d##*|}" | cut -d: -f1)"
    _wanted+="$_b ${_b%.dom}.dom "
  done
  # Cost is proportional to SIZE, so the gate is too: at 128 KiB/s a stray 13 KB rung costs a
  # tenth of a second and a stray 1.5 MB SQLite domain costs twelve. BLOCK only on the big ones;
  # mention the small ones without stopping the run. The 30.6 minutes lost this session were
  # essentially all in 21 multi-megabyte domains.
  _stale=(); _small=0
  while IFS= read -r _f; do
    [[ -n "$_f" ]] || continue
    [[ "$_wanted" == *" $_f "* ]] && continue
    if (( $(stat -c%s "$OVERLAY/$_f" 2>/dev/null || echo 0) > 262144 )); then
      _stale+=("$_f")
    else
      _small=$(( _small + 1 ))
    fi
  done < <(ls -1 "$OVERLAY" 2>/dev/null)
  (( _small > 0 )) && say "warn" "$_small small unused file(s) in the overlay (<256 KB each -- negligible upload, not blocking)"
  if (( ${#_stale[@]} > 0 )); then
    bad "overlay has ${#_stale[@]} file(s) this run does not use -- every one is uploaded over JTAG at 128 KiB/s on EVERY boot. Retire them by EXPLICIT NAME (never a glob; a prefix glob once deleted the package-installed sbi.dom):
    mkdir -p /tmp/capstone/overlay-attic && cd $OVERLAY && mv -t /tmp/capstone/overlay-attic ${_stale[*]}
  then re-run BOTH: make build LINUX_PAYLOAD=1 A=linux-rebuild ... && ... A=opensbi-rebuild ...
  (the linux-rebuild is NOT optional -- omitting it leaves the firmware byte-identical and the
   prune silently does nothing)"
  else
    ok "overlay clean (${#_wanted} expected files, no strays)"
  fi
fi

# C10 -- FIRMWARE BUDGET, not just the hard limit. --------------------------------------------
# C8 blocks at the console's 33,554,432 rejection threshold. That is too late to be useful: a
# clean overlay with four 20 KB rungs is ~17.5 MB, so anything past 20 MB means stale domains
# are riding along. There is a SECOND, quieter ceiling too -- run_ladder_perf_fpga.py gives
# `monitor load_image` a fixed 300 s, and at 128.2 KiB/s a 32 MiB image needs 256 s of that.
# One boot already spent 245 s of the 300 with no warning.
if [[ -n "$FW" && -f "$FW" ]]; then
  _b=$(stat -c%s "$FW"); _upl=$(( _b / 131293 ))
  if (( _b > 20000000 )); then
    bad "firmware $(( _b / 1048576 )) MB exceeds the 20 MB working budget -- the overlay is dirty. A clean one is ~17.5 MB; only a retained multi-megabyte domain gets it past 20."
  else
    ok "firmware $(( _b / 1048576 )) MB within the 20 MB working budget (~${_upl}s upload)"
  fi
  if (( _upl > 240 )); then
    bad "estimated JTAG upload ${_upl}s against load_image's fixed 300s budget -- under 60s of margin, and a timeout there costs the whole boot"
  fi
fi

# C11 -- ENTRY-STALL STREAK. ------------------------------------------------------------------
# R-16 is PER-IMAGE and deterministic per binary, so retrying a stalling image is futile; the
# skill says REDRAW. Seven consecutive boots (32-38) each spent a full firmware build and boot
# on ONE probe that never entered -- SHA5 last, no SHA6 -- producing zero information about any
# code under test. The very next boot staged three redraws of one probe in a single image and
# got 3 of 3 returns. This gate refuses the eighth single-probe load.
_recent_stalls=0
for _t in $(ls -t /tmp/capstone/boot*-raw.txt /tmp/capstone/boot*.txt 2>/dev/null | head -3); do
  python3 - "$_t" <<'PYEOF' || _recent_stalls=$(( _recent_stalls + 1 ))
import sys, re
d = open(sys.argv[1], 'rb').read()
sys.exit(0 if (b'SHA6:' in d or b'ENT1' in d) else 1)
PYEOF
done
if (( _recent_stalls >= 3 )); then
  n_probes=$(( n > 0 ? n - 1 : 0 ))
  if (( n_probes < 3 )); then
    bad "the last 3 runs all ended without SHA6/ENT1 (entry stalls, no verdict) and this load carries only $n_probes probe(s). R-16 is per-image: RETRYING IS FUTILE. Stage >=3 REDRAWS of the same probe (vary QR_DRAW / FDREG_DRAW, sha256sum them, abort if any two match) in ONE image instead."
  else
    ok "recent entry stalls, but this load carries $n_probes probes -- redraw shape"
  fi
fi

# C12 -- UNUSED SLOTS. ------------------------------------------------------------------------
# The gate below caps the load at 4. Fifteen loads this session requested only 3, and an empty
# slot is free information: a 13 KB rung adds ~0 upload and ~4 s of run time.
if (( n > 0 && n < 4 )); then
  say "warn" "$n domains requested of a 4-slot budget -- $(( 4 - n )) slot(s) left empty. A returning rung there is nearly free (set PREFLIGHT_ALLOW_SHORT=1 to silence)."
fi

# C7 ----------------------------------------------------------------------------------------
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
