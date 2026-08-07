#!/usr/bin/env bash
# Make the board image contain EXACTLY the domains this run needs, and nothing else.
#
# WHY THIS EXISTS. The overlay was being ACCUMULATED -- every probe staged, none retired --
# and pruned by hand whenever someone noticed. That reached 111 files / 35 MB in one session,
# cost 30.6 minutes of pure JTAG upload across 39 boots, and lost one boot outright to an
# HTTP 413 rejection that arrives only AFTER the board is locked and power-cycled.
#
# The preflight gates that were added afterwards do not actually close this. Measured against
# the very next session's accumulation (26 files, 1.91 MB):
#   * 25 of the 26 were under C9's 256 KB block threshold, so they only produced a warning;
#   * the 26th was q31.dom at 1.5 MB, which sits on C9's permanent keep-always exemption.
# Net: C9 would have blocked on NOTHING. The prune that happened was manual.
#
# So the fix is not a bigger warning, it is to stop accumulating. The overlay is DERIVED here:
# given the rung list, this makes overlay+target hold exactly {controller} u {package files} u
# {named rungs}, and moves everything else to a dated attic. Nothing is deleted -- a .dom is
# cheap to keep and expensive to regenerate once the flags that built it are forgotten.
#
# Usage:
#   bash capstone/tests/stage-board-domains.sh gp0 gp16 gp32 k800        # dry run, prints plan
#   bash capstone/tests/stage-board-domains.sh --apply gp0 gp16 gp32 k800
#   BAKED_RUNGS="gp0 gp16 gp32 k800" bash capstone/tests/stage-board-domains.sh --apply
#
# Accepts the same spellings the runners use: bare rung names ("gp16"), "x.dom", and
# SQLITE_STAGE_DOMS forms ("/test-domains/q31.dom", "host args|path:selector").
set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/..
ROOT=$(cd -- "$ROOT" && pwd)
OVERLAY=${PREFLIGHT_OVERLAY:-$ROOT/capstone/caplifive-system/sw/buildroot/overlay/test-domains}
TARGET=${STAGE_TARGET:-$ROOT/capstone/caplifive-system/sw/buildroot/build/target/test-domains}
ATTIC=${STAGE_ATTIC:-/tmp/capstone/overlay-attic}

# The controller binary the runners invoke; without it every rung fails to launch.
KEEP_OVERLAY="lpc"
# Installed into the target by the buildroot package, NOT by us. Never attic these: a prefix
# glob once deleted the package-installed sbi.dom and the next build silently shipped without it.
KEEP_TARGET="lpc sbi.dom sbi.smode smode.dom smode.smode thread.dom fib.dom"

APPLY=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) ARGS+=("$a") ;;
  esac
done
[[ ${#ARGS[@]} -eq 0 ]] && read -r -a ARGS <<<"${BAKED_RUNGS:-} $(tr ',' ' ' <<<"${SQLITE_STAGE_DOMS:-}")"

WANTED=" "
for d in "${ARGS[@]:-}"; do
  [[ -n "$d" ]] || continue
  b="$(basename "${d##*|}" | cut -d: -f1)"      # strip "host args|" and ":selector"
  WANTED+="${b%.dom}.dom $b "
done
[[ "$WANTED" == " " ]] && { echo "no rungs given (pass names, or set BAKED_RUNGS)" >&2; exit 1; }

echo "=== stage-board-domains ==="
echo "  wanted: $(tr ' ' '\n' <<<"$WANTED" | grep -c '\.dom$' ) domain name(s)"
[[ "$APPLY" -eq 1 ]] || echo "  DRY RUN -- nothing will move. Re-run with --apply to act."

plan_dir() {  # $1=dir  $2=space-delimited keep-list  $3=label
  local dir="$1" keep=" $2 " label="$3" f sz tot=0 n=0
  [[ -d "$dir" ]] || { echo "  [$label] missing: $dir"; return; }
  local stale=() keptb=0
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    sz=$(stat -c%s "$dir/$f" 2>/dev/null || echo 0)
    if [[ "$keep" == *" $f "* || "$WANTED" == *" $f "* ]]; then
      keptb=$(( keptb + sz )); n=$(( n + 1 ))
    else
      stale+=("$f"); tot=$(( tot + sz ))
    fi
  done < <(ls -1 "$dir" 2>/dev/null)
  printf "  [%s] keep %d file(s) = %d B | retire %d file(s) = %d B (~%.1fs of JTAG per boot)\n" \
    "$label" "$n" "$keptb" "${#stale[@]}" "$tot" "$(awk "BEGIN{print $tot/131072}")"
  if (( ${#stale[@]} > 0 )); then
    printf "        %s\n" "${stale[*]}"
    if [[ "$APPLY" -eq 1 ]]; then
      mkdir -p "$ATTIC"
      local moved=0
      for f in "${stale[@]}"; do            # BY EXPLICIT NAME, never a glob
        mv -f -- "$dir/$f" "$ATTIC/$f" 2>/dev/null && moved=$(( moved + 1 ))
      done
      echo "        -> moved $moved to $ATTIC"
    fi
  fi
}

plan_dir "$OVERLAY" "$KEEP_OVERLAY" "overlay"
plan_dir "$TARGET"  "$KEEP_TARGET"  "target "

# The target dir is what buildroot actually packs. Overlay-clean while target is dirty ships the
# stale file anyway, and until now nothing checked target at all -- so report any divergence.
if [[ -d "$OVERLAY" && -d "$TARGET" ]]; then
  miss=()
  for w in $WANTED; do
    [[ "$w" == *.dom ]] || continue
    [[ -f "$OVERLAY/$w" ]] || continue
    [[ -f "$TARGET/$w" ]] || miss+=("$w")
  done
  if (( ${#miss[@]} > 0 )); then
    echo "  WARN: in overlay but NOT in target (buildroot packs target): ${miss[*]}"
    [[ "$APPLY" -eq 1 ]] && { for m in "${miss[@]}"; do cp -f "$OVERLAY/$m" "$TARGET/$m"; done; echo "        -> copied across"; }
  fi
fi

if [[ "$APPLY" -eq 1 ]]; then
  cat <<'EOF'

  Now rebuild BOTH, in this order -- linux-rebuild is NOT optional. Buildroot does not track
  overlay/ into the cpio, so an OpenSBI-only relink leaves the firmware byte-identical and the
  retirement silently does nothing:
    cd capstone/caplifive-system/sw/buildroot
    make build LINUX_PAYLOAD=1 A=linux-rebuild  CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
    make build LINUX_PAYLOAD=1 A=opensbi-rebuild CAPSTONE_CC_PATH="$(realpath ../../../capstone-c)"
  Then confirm the firmware HASH changed. Never the size: buildroot pads in 2 MiB steps, so two
  different images routinely have identical sizes.
EOF
fi
