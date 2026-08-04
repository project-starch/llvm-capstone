#!/usr/bin/env bash
# Board-session watchdog. Run it ALONGSIDE a board runner, as a second process.
#
# WHY THIS EXISTS. A board runner can sit for its whole per-domain budget with the FPGA doing
# nothing -- a wedged domain emits no UART at all, and a runner that died leaves a waiter
# spinning on a file that will never be written. Both look identical from outside: "still
# running". On 2026-08-02 that cost a stretch of wall-clock where nothing was on the board at
# all, while stale `tail -f` processes on finished logs made it look busy.
#
# The runner's own timeouts are not enough: they are per-domain and silent until they expire,
# so a 480 s budget spends 480 s before saying anything. This reports every interval.
#
# LIVENESS IS BY PID, NEVER BY PROCESS NAME. `pgrep -f <pattern>` matches the watchdog's OWN
# command line, because the pattern is one of its arguments -- so it reports the runner alive
# forever and the watchdog never fires. That is the self-matching-pgrep trap this project has
# hit three times; the first draft of THIS script hit it too and a unit test caught it.
#
# LIVE ENTRY-STALL DETECTION. R-16 itself was FIXED IN SILICON on 2026-08-04
# (`caplifive_fixed_forward.bit`, capability operand forwarding), so this should now fire only
# on a bitstream lacking that fix -- which is exactly why the detection stays: it is the cheap
# early warning that the board was reflashed to one. See capstone/tests/fpga-repros/R16-entry-stall/.
# `SHA5` means the monitor handed off to the domain; `SHA6`
# means the domain came back. `SHA5` with no following `SHA6` and no further UART is an entry
# stall: the domain's code never ran, so THIS run carries no information and every later domain
# in the same boot is lost anyway (a wedged domain takes the core). Waiting out the per-domain
# timeouts after that is pure waste -- 3 domains x 200 s = 600 s spent to learn nothing.
# With ABORT_ON_ENTRY_STALL=1 the watchdog kills the runner as soon as the pattern is stable,
# turning a 600 s dead run into ~30 s.
#
# Usage:
#   bash board-watchdog.sh <uart-log> [idle-limit-s] [runner-pid]
#   ABORT_ON_ENTRY_STALL=1 ENTRY_STALL_S=260 bash board-watchdog.sh ...
#   (runner-pid omitted -> liveness is not checked, only UART idle is reported)
#
# Emits ONE LINE PER CHECK on stdout, so it can be piped into a Monitor:
#   ALIVE   <elapsed>s  +<bytes> since last check
#   QUIET   <elapsed>s  no UART for <idle>s  (limit <limit>s)
#   STALE   <elapsed>s  no UART for <idle>s  -- EXCEEDED LIMIT
#   GONE    runner process no longer exists (log last grew <idle>s ago)
#   ENDED   runner finished
set -uo pipefail

LOG=${1:?usage: board-watchdog.sh <uart-log> [idle-limit-s] [runner-pattern]}
LIMIT=${2:-180}
RUNNER_PID=${3:-}
INTERVAL=${WATCHDOG_INTERVAL:-15}
ABORT_ON_ENTRY_STALL=${ABORT_ON_ENTRY_STALL:-0}
# MEASURED, not guessed: the JTAG upload runs at ~130 KiB/s, so a 17.4 MB firmware is ~133 s
# of completely legitimate UART silence, and larger images have taken 227 s. Anything below
# that aborts healthy runs DURING THE UPLOAD -- which is what produced a whole session of
# false "the board will not boot" / "cyclic boot" / "the firmware is broken" diagnoses on
# 2026-08-02. The earlier comment here said "healthy boots have gone 120 s silent"; that
# figure predated the upload measurement and was itself an underestimate. Do not lower this.
ENTRY_STALL_S=${ENTRY_STALL_S:-260}

start=$SECONDS
# byte offset of the log when we started: everything before it is replayed scrollback
START_SIZE=$( [ -f "$LOG" ] && echo $(( $(stat -c%s "$LOG") + 1 )) || echo 1 )
last_size=-1
last_change=$SECONDS

while true; do
  sleep "$INTERVAL"
  now_elapsed=$(( SECONDS - start ))
  size=$( [ -f "$LOG" ] && stat -c%s "$LOG" 2>/dev/null || echo 0 )

  # kill -0 tests existence of a specific process. No name matching, so nothing can match
  # this script itself. With no PID given, assume alive and rely on the idle limit alone.
  runner_alive=1
  if [ -n "$RUNNER_PID" ]; then
    kill -0 "$RUNNER_PID" 2>/dev/null || runner_alive=0
  fi

  if [ "$size" -ne "$last_size" ]; then
    delta=$(( size - (last_size < 0 ? size : last_size) ))
    last_size=$size
    last_change=$SECONDS
    echo "ALIVE   ${now_elapsed}s  +${delta}B"
  else
    idle=$(( SECONDS - last_change ))
    # NO-BOOT: load_image issued but the board never printed a boot banner. Seen when the
    # JTAG transfer silently moves nothing -- the log shows load_image, set $pc, continue, and
    # then dead air, with no "downloaded N bytes" line. The runner would otherwise burn its
    # whole per-domain budget on a board that has no image in DDR.
    if [ "$ABORT_ON_ENTRY_STALL" = "1" ] && [ "$idle" -ge "$ENTRY_STALL_S" ] && [ -f "$LOG" ]; then
      # RUN-SCOPED, same reason as the entry-stall scan below: `tail -c +$START_SIZE` was
      # effectively the whole log (START_SIZE is ~1 because callers rm the log first), so the
      # replayed scrollback's own `buildroot login` matched the healthy case and this check
      # could never fire. That direction is a false NEGATIVE -- dead boots ran to full timeout.
      scan0=$(awk '/emit gdb_input .*monitor load_image/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG" 2>/dev/null)
      case "$scan0" in
        *load_image*)
          case "$scan0" in
            # HEALTH SIGNAL IS THE LOGIN PROMPT, NOT THE BOOTROM BANNER. Measured:
            # healthy runs show ZERO "Hello World" after their own load_image (the
            # banner prints at power-on, before the load), but DO show `buildroot
            # login` and tens of KB of kernel output. A dead boot shows ~900 bytes
            # and neither. Keying on the banner would never have fired.
            *"buildroot login"*|*"SQ: "*) ;;        # userspace reached: fine
            *)
              echo "NO-BOOT ${now_elapsed}s  load_image issued, no boot banner and no download for ${idle}s"
              echo "  -> JTAG transfer moved nothing; this boot is dead. Aborting runner."
              [ -n "$RUNNER_PID" ] && kill -0 "$RUNNER_PID" 2>/dev/null && kill -TERM "$RUNNER_PID" 2>/dev/null
              echo "ENDED   ${now_elapsed}s"; exit 0 ;;
          esac ;;
      esac
    fi

    # Entry stall: the LAST capability-share marker in the log is SHA5 with no SHA6 after it.
    if [ "$ABORT_ON_ENTRY_STALL" = "1" ] && [ "$idle" -ge "$ENTRY_STALL_S" ] && [ -f "$LOG" ]; then
      # ONLY look at bytes written AFTER this watchdog started, and only after the board has
      # actually been given the image. The console REPLAYS the previous boot's scrollback
      # (~548 KB) when the driver connects, so grepping the whole log finds a SHA5 from a
      # PREVIOUS boot and aborts a perfectly healthy run before it has even booted. That bug
      # killed 24 runs over ~87 minutes on 2026-08-02 and produced the false conclusion that
      # "the board stopped accepting images".
      # SCAN FROM THIS RUN'S OWN load_image EMIT, not from a byte offset captured at start.
      # START_SIZE was always ~1 because every caller does `rm -f "$LOG"` before launching the
      # runner, so the log is empty when the watchdog starts and the console's ~548 KB of
      # replayed scrollback lands INSIDE the window. Measured post-"fix": iglog-234029 had its
      # last SHA marker at offset 492011 and its own load_image at 548318, and the abort still
      # fired at 255 s. Only the 45->180 s threshold bump was actually helping.
      # The driver's `emit gdb_input <- {'text': 'monitor load_image` line is emitted by THIS
      # run, so everything after its last occurrence is unambiguously ours.
      scan=$(awk '/emit gdb_input .*monitor load_image/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' "$LOG" 2>/dev/null)
      # require the image to have been handed over in THIS run before any stall verdict
      case "$scan" in *load_image*) ;; *) lastmark=""; scan="";; esac
      lastmark=$(printf '%s' "$scan" | sed -n 's/.*\(SHA[56]:[0-9A-F]*\).*/\1/p' | tail -1)
      case "$lastmark" in
        SHA5:*)
          if [ -n "$RUNNER_PID" ] && ! kill -0 "$RUNNER_PID" 2>/dev/null; then
            echo "GONE    ${now_elapsed}s  runner already exited; not sending a stray TERM"
            echo "ENDED   ${now_elapsed}s"; exit 0
          fi
          echo "ENTRY-STALL ${now_elapsed}s  last share marker=$lastmark, no SHA6 for ${idle}s"
          echo "  -> domain never ran; the rest of this boot is worthless. Aborting runner."
          [ -n "$RUNNER_PID" ] && kill -TERM "$RUNNER_PID" 2>/dev/null
          echo "ENDED   ${now_elapsed}s"
          exit 0 ;;
      esac
    fi
    if [ "$runner_alive" -eq 0 ]; then
      echo "GONE    ${now_elapsed}s  runner not running; log last grew ${idle}s ago"
      echo "ENDED   ${now_elapsed}s"
      exit 0
    fi
    if [ "$idle" -ge "$LIMIT" ]; then
      echo "STALE   ${now_elapsed}s  no UART for ${idle}s -- EXCEEDED LIMIT ${LIMIT}s"
    else
      echo "QUIET   ${now_elapsed}s  no UART for ${idle}s (limit ${LIMIT}s)"
    fi
  fi

  if [ "$runner_alive" -eq 0 ]; then
    echo "GONE    ${now_elapsed}s  runner PID ${RUNNER_PID:-?} is no longer running"
    echo "ENDED   ${now_elapsed}s"
    exit 0
  fi
done
