#!/bin/sh
# Guest-side: run a program and, every 20 s, sum the resident pages of its
# mappings by owner (procstat -v; RES is in 4 KiB pages).
#   run-maps.sh OUT PROGRAM ARGS...
# OUT receives one line per sample:
#   MAPS t=<epoch> total=<pages> jemalloc=<pages> mrs=<pages> anon=<pages> file=<pages> stack=<pages>
# followed by the program's own output and time -l.  The program runs with
# MRS's default (revocation on, asynchronous) and mqstat.so, nothing else.
OUT=$1; shift
rm -f "$OUT"
env -i PATH=/bin:/usr/bin HOME=/root _RUNTIME_REVOCATION_ENABLE=1 _RUNTIME_REVOCATION_ASYNC_REVOKE=1 \
  LD_PRELOAD=/root/mq/mqstat.so /usr/bin/time -l "$@" > "$OUT.prog" 2>&1 &
sleep 2
P=$(pgrep -x "$(basename "$1")" | head -1)
echo "MAPS-PID $P" >> "$OUT"
while kill -0 "$P" 2>/dev/null; do
  procstat -v "$P" 2>/dev/null | awk -v t="$(date +%s)" '
    NR > 1 {
      n = $11
      if (n == "") n = "anon"
      else if (n ~ /^mrs:/) n = "mrs"
      else if (n ~ /^jemalloc:/) n = "jemalloc"
      else if (n ~ /^\//) n = "file"
      else n = "other"
      r[n] += $5; tot += $5
    }
    END { printf "MAPS t=%d total=%d jemalloc=%d mrs=%d anon=%d file=%d other=%d\n", t, tot, r["jemalloc"], r["mrs"], r["anon"], r["file"], r["other"] }' >> "$OUT"
  sleep 20
done
wait
cat "$OUT.prog" >> "$OUT"
echo "MAPS-DONE" >> "$OUT"
