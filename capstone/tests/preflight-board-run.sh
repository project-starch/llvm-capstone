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
# A run names its rungs EITHER in BAKED_RUNGS or in the ladder-host form inside
# SQLITE_STAGE_DOMS ("/test-domains/lpc|k800:/test-domains/k800.dom"). Only the first was read,
# so for every boot that used the second, RUNGS was empty and the per-rung gates below --
# control-has-a-record (C4), oracle present (C3), qemu-pass hash (C13), initramfs membership
# (C14) -- were skipped in silence while preflight still printed GO. That is the failure mode
# CLAUDE.md names outright: "a run parameterised by the wrong variable name, printing a pass
# having checked nothing." Derive the names from both spellings.
if [[ -z "$RUNGS" && "$DOMS" == *"|"* ]]; then
  _derived=""
  for _e in $(tr ',' ' ' <<<"$DOMS"); do
    [[ "$_e" == *"|"* ]] || continue          # "path" / "path:selector" are not ladder rungs
    _r=${_e#*|}; _r=${_r%%:*}                  # "host|RUNG:path" -> RUNG
    [[ -n "$_r" && "$_r" != /* ]] && _derived+="$_r "
  done
  [[ -n "$_derived" ]] && RUNGS=$_derived
fi

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
    # caplifive_r20.bit is branch r20-fix, which is based on the 65536-node lineage: its
    # capstone_rev_node.anvil uses a 16-bit head with REVNODE_SENTINEL = 65535, so it takes
    # the SAME DTS as 65536_nodes. Listed explicitly because the name matches neither
    # pattern, so this check fell through to the "unknown bitstream" warn and silently
    # verified nothing for every run on the resident silicon.
    *65536*|*_r20*)  [[ "$have_64k" -ge 1 && "$have_1021" -eq 0 ]] \
        && ok "DTS matches $BITSTREAM (0x3c2d2000)" \
        || bad "DTS/bitstream MISMATCH: $BITSTREAM needs 0x3c2d2000 (found 1021=$have_1021 64k=$have_64k)" ;;
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
# MEASURED AGAINST THE NEXT SESSION (2026-08-07), this gate as originally written would have
# blocked on NOTHING. That session accumulated 26 files / 1.91 MB: 25 were under the 256 KB
# threshold below and so only warned, and the 26th was q31.dom at 1.5 MB, which was on this
# exemption list. The prune that actually happened was manual. Two consequences:
#   * q31.dom and sqlite_host.user come OFF the permanent exemption -- they are experiment
#     artifacts, not infrastructure, and a run that needs them names them in SQLITE_STAGE_DOMS,
#     which is already folded into _wanted below;
#   * small strays now have a CUMULATIVE budget (C9b) instead of an unbounded warning, because
#     the ladder-rung workflow generates 13 KB files by the dozen and 25 of them is 2.7 s of
#     JTAG on every boot forever.
# Retire with: bash capstone/tests/stage-board-domains.sh --apply <rungs...>
OVERLAY_KEEP_ALWAYS="sbi.dom sbi.smode smode.dom smode.smode thread.dom fib.dom lpc k800.dom"
if [[ -d "$OVERLAY" ]]; then
  _wanted=" $OVERLAY_KEEP_ALWAYS "
  # BAKED_RUNGS names rungs WITHOUT the .dom suffix ("lf0"), while SQLITE_STAGE_DOMS gives full
  # paths ("/test-domains/q31.dom", optionally "host args|path:selector"). Accept both spellings
  # or the gate flags the very domains the run needs -- which it did on the first dry run.
  _req=()
  for _d in ${RUNGS} $(tr ',' ' ' <<<"$DOMS"); do
    _b="$(basename "${_d##*|}" | cut -d: -f1)"
    _wanted+="$_b ${_b%.dom}.dom "
    [[ "$_b" == *.dom ]] && _req+=("$_b")
  done
  # C15 REQUESTED-BUT-ABSENT. The gate above catches files that are present and unwanted; this
  # catches the opposite, which is the one that costs a boot. A run naming a .dom that was never
  # staged boots, reaches the arm, and dies on "ladder-perf: open .dom failed" -- and everything
  # after it in the sequence is lost with it.
  #
  # It happens in a specific, quiet way: verify-and-stage-rung.sh deletes its artifact FIRST,
  # then builds, then compares against the native oracle, and only stages if that matches. A rung
  # whose oracle cannot match -- one reporting a capability TYPE, which native code cannot model
  # -- still PRINTS its QEMU retval, so the measurement looks taken while the .dom was never
  # placed. The only signal is an absent "staged:" line. On 2026-08-25 that read as success and
  # cost a boot.
  #
  # BLOCKING, not a warning: there is no version of "the run needs a file the image lacks" that
  # is worth spending a JTAG upload and a boot to discover.
  _missing=()
  for _b in "${_req[@]}"; do [[ -f "$OVERLAY/$_b" ]] || _missing+=("$_b"); done
  if (( ${#_missing[@]} )); then
    bad "requested but NOT staged in the overlay: ${_missing[*]} -- the boot would reach the arm and die on 'open .dom failed', losing every test after it. Re-run verify-and-stage-rung.sh and confirm it prints a 'staged:' line (an oracle mismatch aborts BEFORE staging while still printing a retval)."
  else
    ok "all ${#_req[@]} requested domain(s) present in the overlay"
  fi
  # Cost is proportional to SIZE, so the gate is too: at 128 KiB/s a stray 13 KB rung costs a
  # tenth of a second and a stray 1.5 MB SQLite domain costs twelve. BLOCK only on the big ones;
  # mention the small ones without stopping the run. The 30.6 minutes lost this session were
  # essentially all in 21 multi-megabyte domains.
  _stale=(); _small=0; _smallover=0
  while IFS= read -r _f; do
    [[ -n "$_f" ]] || continue
    [[ "$_wanted" == *" $_f "* ]] && continue
    if (( $(stat -c%s "$OVERLAY/$_f" 2>/dev/null || echo 0) > 262144 )); then
      _stale+=("$_f")
    else
      _small=$(( _small + 1 ))
    fi
  done < <(ls -1 "$OVERLAY" 2>/dev/null)
  # C9b -- CUMULATIVE budget for the small ones. Individually negligible, collectively not: the
  # ladder workflow emits 13 KB rungs by the dozen, and 25 of them measured 360 KB = 2.7 s of
  # JTAG on EVERY boot, silently, forever. Per-file thresholds cannot see that; a total can.
  if (( _small > 0 )); then
    _smallb=0
    while IFS= read -r _f; do
      [[ -n "$_f" ]] || continue
      [[ "$_wanted" == *" $_f "* ]] && continue
      _sz=$(stat -c%s "$OVERLAY/$_f" 2>/dev/null || echo 0)
      (( _sz <= 262144 )) && _smallb=$(( _smallb + _sz ))
    done < <(ls -1 "$OVERLAY" 2>/dev/null)
    if (( _smallb > 524288 )); then
      _smallover=1
      bad "$_small small unused file(s) total $_smallb B in the overlay -- past the 512 KB cumulative budget, ~$(( _smallb / 131072 ))s of JTAG on every boot. Retire: bash capstone/tests/stage-board-domains.sh --apply \$BAKED_RUNGS"
    else
      say "warn" "$_small small unused file(s) in the overlay ($_smallb B, under the 512 KB cumulative budget)"
    fi
  fi
  if (( ${#_stale[@]} > 0 )); then
    bad "overlay has ${#_stale[@]} file(s) this run does not use -- every one is uploaded over JTAG at 128 KiB/s on EVERY boot. Retire them by EXPLICIT NAME (never a glob; a prefix glob once deleted the package-installed sbi.dom):
    mkdir -p /tmp/capstone/overlay-attic && cd $OVERLAY && mv -t /tmp/capstone/overlay-attic ${_stale[*]}
  then re-run BOTH: make build LINUX_PAYLOAD=1 A=linux-rebuild ... && ... A=opensbi-rebuild ...
  (the linux-rebuild is NOT optional -- omitting it leaves the firmware byte-identical and the
   prune silently does nothing)"
  elif (( _smallover == 0 )); then
    ok "overlay clean (${#_wanted} expected files, no strays)"
  fi
fi

# C9c -- THE TARGET DIR, which is what actually gets packed. -----------------------------------
# Until 2026-08-07 this gate looked ONLY at overlay/. But buildroot packs
# build/target/test-domains into the cpio; overlay/ is merely copied into it. So a file retired
# from the overlay but left in the target still ships on every boot while the gate reports
# "overlay clean" -- a silent pass in exactly the situation the gate exists to catch. Checked
# separately because the target legitimately holds package-installed files the overlay never has.
TARGET_DIR=${STAGE_TARGET:-capstone/caplifive-system/sw/buildroot/build/target/test-domains}
TARGET_KEEP="sbi.dom sbi.smode smode.dom smode.smode thread.dom fib.dom lpc"
if [[ -d "$TARGET_DIR" && -n "$RUNGS$DOMS" ]]; then
  _tw=" $TARGET_KEEP $OVERLAY_KEEP_ALWAYS "
  for _d in ${RUNGS} $(tr ',' ' ' <<<"$DOMS"); do
    _b="$(basename "${_d##*|}" | cut -d: -f1)"; _tw+="$_b ${_b%.dom}.dom "
  done
  _tstale=(); _tb=0
  while IFS= read -r _f; do
    [[ -n "$_f" ]] || continue
    [[ "$_tw" == *" $_f "* ]] && continue
    _tstale+=("$_f"); _tb=$(( _tb + $(stat -c%s "$TARGET_DIR/$_f" 2>/dev/null || echo 0) ))
  done < <(ls -1 "$TARGET_DIR" 2>/dev/null)
  if (( _tb > 524288 )); then
    bad "buildroot TARGET dir carries ${#_tstale[@]} unused file(s) totalling $_tb B -- these ship even if the overlay is clean, because the cpio is packed from the target. Retire: bash capstone/tests/stage-board-domains.sh --apply \$BAKED_RUNGS"
  elif (( ${#_tstale[@]} > 0 )); then
    say "warn" "target dir has ${#_tstale[@]} unused file(s) ($_tb B, under budget): ${_tstale[*]}"
  else
    ok "buildroot target dir clean (packs exactly what this run needs)"
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
#
# 2026-08-07: this gate was INERT and had been for its whole life. The test was `b'SHA6:' in d`
# over the WHOLE transcript, but the same skill mandates a known-good control as rung 1 of every
# boot, and that control always emits SHA6. So every file scored "clean" no matter how many of its
# probes stalled, `_recent_stalls` never left 0, and the block below was unreachable in practice --
# including across the five stage-34 stalls it was written to catch. Same class of defect as a
# `grep -c "A\|B"` that cannot say which matched: the check was real, its scope was wrong.
#
# Scope it to the LAST rung of each transcript instead. That is the position an unknown occupies
# under the ordering rule, so it is the one whose stall carries information.
_recent_stalls=0
for _t in $(ls -t /tmp/capstone/boot*-raw.txt /tmp/capstone/boot*.txt /tmp/capstone/*-run.txt /tmp/capstone/baked-rungs.txt 2>/dev/null | head -3); do
  python3 - "$_t" <<'PYEOF' || _recent_stalls=$(( _recent_stalls + 1 ))
import sys, re
d = open(sys.argv[1], 'rb').read()
# Split on the per-rung banner the drivers emit ("===== <rung> (pos N) ====="), and examine
# only the final block. Split on that banner ALONE: the transcripts also carry
# "### RUNG n/N <rung> ###" lines, and including those over-splits so the last block is the
# trailing "END rc=" fragment, which never contains SHA6 -- that scored a fully successful
# 4-rung boot as a stall. Verified both ways against gvfix-run.txt (all four returned -> clean)
# and prov/prov2/prov3-run.txt (probe stalled -> caught).
blocks = re.split(rb'^={3,}\s*\S+.*$', d, flags=re.M)
tail = blocks[-1] if len(blocks) > 1 else d
sys.exit(0 if (b'SHA6:' in tail or b'ENT1' in tail) else 1)
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
  # C14 INITRAMFS-MEMBERSHIP GATE. Being in overlay/ is NOT being in the image.
  #
  # buildroot does not track overlay/ -> cpio, so a .dom copied in AFTER the last
  # `A=linux-rebuild` is simply absent from the initramfs the board boots. What makes this
  # worth a hard gate rather than a note is how the runner then reports it: a missing file
  # makes `lpc` exit non-zero, `DN_$rc` still matches DN_\d, so wedged=False and got=None,
  # and run_baked_rungs_fpga.py:136-141 prints "NO SHA6 -- ENTRY STALL (R-16) ... REDRAW".
  # The operator is told to redraw an image that was never on the board. Five stalls on
  # 2026-08-07 were investigated as R-16 with this possibility never checked, and at the
  # time this gate was written gvf0.dom and gvf6.dom were both in overlay/ and both absent
  # from a rootfs.cpio built eight minutes earlier.
  #
  # The cpio is uncompressed, so a plain byte-substring test is exact and needs no tooling.
  CPIO=${CPIO:-$(dirname "$OVERLAY")/../build/images/rootfs.cpio}
  if [[ -f "$CPIO" ]]; then
    for r in $RUNGS; do
      [[ -f "$OVERLAY/$r.dom" ]] || continue
      if ! python3 - "$CPIO" "$OVERLAY/$r.dom" <<'PY'
import sys
cpio=open(sys.argv[1],'rb').read()
sys.exit(0 if open(sys.argv[2],'rb').read() in cpio else 1)
PY
      then
        bad "$r.dom is staged in $OVERLAY but its bytes are NOT in $(basename "$CPIO") -- it will not exist on the board, and the runner will misreport that as an R-16 entry stall. Rebuild: make build LINUX_PAYLOAD=1 A=linux-rebuild, THEN A=opensbi-rebuild."
      fi
    done
  else
    say "warn" "rootfs.cpio not found at $CPIO -- cannot verify the staged .dom files are actually in the initramfs"
  fi
  # C13 QEMU-PASS GATE. A rung must have a RECORDED QEMU pass before it costs a boot.
  # 2026-08-07: a build was staged and run while its QEMU oracle line said NO RESULT -- the line was
  # printed and not acted on. Whether or not that build was sound, nothing should reach the board
  # unverified. `.qemu-pass` is written by the verify step next to the oracle; its absence blocks.
  # A stale pass is worse than none, so the marker must be NEWER than the .dom it vouches for.
  for r in $RUNGS; do
    [[ -f "$OVERLAY/$r.dom" ]] || continue
    if [[ ! -f "$ORACLES/$r.qemu-pass" ]]; then
      bad "$r has no recorded QEMU pass ($ORACLES/$r.qemu-pass) -- verify under QEMU before spending a boot"
    else
      # An mtime comparison is not proof: the markers can be (and were) `touch`ed retroactively,
      # zero-byte, minutes after the boot they supposedly vouched for. Bind the marker to the
      # artifact's CONTENT instead -- it must record the sha256 of the exact .dom being staged.
      _want=$(sha256sum "$OVERLAY/$r.dom" | cut -d" " -f1)
      _have=$(grep -oE "[0-9a-f]{64}" "$ORACLES/$r.qemu-pass" 2>/dev/null | head -1)
      if [[ -z "$_have" ]]; then
        bad "$r.qemu-pass records no image hash -- an empty marker cannot distinguish verified from touched"
      elif [[ "$_have" != "$_want" ]]; then
        bad "$r.qemu-pass vouches for a DIFFERENT image (has ${_have:0:12}, staged ${_want:0:12})"
      fi
    fi
  done

  # Count DISTINCT RUNG NAMES, not slots. Repeating one rung in a boot is a documented and
  # valuable practice here -- results on this board have flipped between runs, so a probe is
  # meant to be run more than once per boot -- and scoring repeats as "identical images" turns
  # a deliberate determinism check into a BLOCK. What this gate is actually for is catching
  # images that were supposed to DIFFER and came out byte-identical because the flags never
  # reached the compile, so it must compare one hash per distinct name.
  _names=$(printf "%s\n" $RUNGS | sort -u | grep -c .)
  u=$(printf "$shas" | sort -u | grep -c .); t=$(printf "$shas" | grep -c .)
  if [[ "$u" -eq "$_names" ]]; then
    if [[ "$t" -ne "$_names" ]]; then
      ok "$_names distinct images over $t slots (repeats are intentional)"
    else
      ok "$t staged images, all distinct"
    fi
  else
    bad "only $u distinct images among $_names distinct rungs -- images that should differ are byte-identical, so the flags did not reach the compile"
  fi
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
