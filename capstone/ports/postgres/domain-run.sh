# What both replay runners do, sourced by each of them.
#
# run-pg-replay.sh boots the unprotected arm and run-pg-sublet.sh the
# protected one. They differ in which image they build, which regions the host
# is told to make, and what the guest command asks for. The rest was the same
# file twice: sizing the trace region from the file, sizing cma from the
# regions, staging the three files into the shared directory, and booting the
# guest once.
#
# Globals the caller sets before calling, and the names are the ones the build
# scripts use too:
#
#   SCRIPT_DIR OUT SHARE_DIR LOG_FILE
#   PG_PAYLOAD PG_ARENA PG_TRACE  and PG_SCRATCH for the protected arm
#   PYTHON
#
# pgdom_trace_region sizes the trace region from the file, so one build serves
# traces of different lengths without a rebuild.
pgdom_trace_region() {
  local trace=$1
  TRACE_BYTES=$(stat -c %s "$trace")
  PG_TRACE=${PG_TRACE:-$(( (TRACE_BYTES + 1048575) / 1048576 * 1048576 + 1048576 ))}
}

# cma= has to cover every region the host makes and the kernel's own use of the
# area, because a region above four megabytes comes from there rather than from
# the buddy allocator.
pgdom_cma() {
  local total=0 r
  for r in "$@"; do total=$(( total + r )); done
  PG_CMA=${PG_CMA:-$(( total / 1048576 + 64 ))M}
}

# Whatever a previous run left, before anything is built. A build that fails
# must not leave a stale image behind for the boot to pick up, which is why
# this is separate and comes first.
pgdom_clean() {
  local image=$1
  mkdir -p "$OUT" "$SHARE_DIR"
  rm -f "$SHARE_DIR/$image" "$SHARE_DIR/pg_host.user" "$SHARE_DIR/trace.a11"
}

# One boot, with the shared directory as it stands. The caller passes the
# guest command and the marker that means the run finished.
pgdom_boot() {
  local guest_cmd=$1 marker=$2

  "$PYTHON" "$CAPSTONE_REPO_ROOT/capstone/tests/runtime-qemu/run-domain-smoke.py" \
    --share-dir "$SHARE_DIR" \
    --log-file "$LOG_FILE" \
    --timeout-multiplier "${PG_TIMEOUT_MULTIPLIER:-40}" \
    --kernel-arg "cma=$PG_CMA" \
    --guest-command "$guest_cmd" \
    --success-marker "$marker"
}

# The guest host, built with the region sizes the domain was built against, so
# the two halves cannot disagree about them.
pgdom_host() {
  local extra=${1:-}
  PG_EXTRA_DEFS="-DPG_REPLAY_PAYLOAD_SIZE=${PG_PAYLOAD}UL -DPG_REPLAY_ARENA_SIZE=${PG_ARENA}UL -DPG_REPLAY_TRACE_SIZE=${PG_TRACE}UL $extra" \
    OUT="$OUT" OUT_HOST="$SHARE_DIR/pg_host.user" \
    bash "$SCRIPT_DIR/build-pg-host.sh"
}

# What the domain said, between its two markers, and where the rest is.
pgdom_report() {
  echo
  echo "== what the domain reported"
  sed -n '/__CAPSTONE_PG_REPLAY_ENTRY__/,/__CAPSTONE_PG_REPLAY_DONE__/p' "$LOG_FILE" || true
  echo "full serial log: $LOG_FILE"
}
