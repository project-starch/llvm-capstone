#!/bin/sh
# Runs INSIDE the CheriBSD guest.  usage: guest-run-probe.sh
#
# ONE ssh SESSION FOR THE WHOLE SWEEP. The earlier host-side driver opened two or
# three connections per case; at ~34 runs that is a hundred logins into a guest whose
# sshd is MaxStartups 10:30:100 behind a hostfwd with a backlog of 1. This project has
# wedged that guest once already by reconnecting too often, so the loop lives here and
# the host reads one stream.
#
# Each binary carries one marker at the sqlite3.c line where the host ASan oracle --
# same case source, same flags -- reported the memory error. LB_SITE arms exactly one.
#   LB_SUMMARY hits=0   the defect site never executed; says nothing about CHERI
#   hits>0, wit=0       the site executed, this access could not be shown to be stale
#   LB_WITNESS          the defective access happened (UAF / OOB past the REQUESTED
#                       size / CAP-OOB). With no fault: reached and not detected.
P=/root/corpus-probe
cd /root/corpus || exit 1
for f in *.db; do [ -f "$f.orig" ] || cp "$f" "$f.orig"; done

run_one() {   # bin site nomem5
  bin=$1; site=$2; nm=$3
  if [ ! -x "$P/$bin" ]; then
    printf '%-20s %-4s %-14s %s\n' "$bin" "$site" NOBIN ""
    return
  fi
  img=`echo "$bin" | sed 's/_sys$//'`
  [ -f "$img.db.orig" ] && cp -f "$img.db.orig" "$img.db"
  if [ "$nm" = 1 ]; then
    env NOMEM5=1 LB_SITE=$site timeout 300 "$P/$bin" > /tmp/pr.txt 2>&1
  else
    env LB_SITE=$site timeout 300 "$P/$bin" > /tmp/pr.txt 2>&1
  fi
  rc=$?
  cp /tmp/pr.txt "/root/probe-logs/$bin.log"
  armed=`grep -c "^LB_ARMED site=$site\$" /tmp/pr.txt`
  hits=`grep '^LB_SUMMARY' /tmp/pr.txt | sed -n 's/.*hits=\([0-9]*\).*/\1/p' | head -1`
  wit=`grep -c '^LB_WITNESS ' /tmp/pr.txt`
  kinds=`grep -o 'kind=[A-Z-]*' /tmp/pr.txt | sort -u | tr '\n' ',' | sed 's/,$//'`
  code=`grep -o 'si_code=[0-9]* ([^)]*)' /tmp/pr.txt | head -1`
  if [ "$nm" = 1 ] && [ "$bin" != "fts3destroyoom_sys" ] && ! grep -q 'allocator=system' /tmp/pr.txt; then
    v=BADCONFIG
  elif [ "$armed" -ne 1 ]; then v=NOT-ARMED
  elif [ "$rc" = 124 ]; then v=TIMEOUT
  elif [ -n "$code" ]; then v=FAULT
  elif [ -z "$hits" ]; then v=NO-SUMMARY
  elif [ "$hits" = 0 ]; then v=NOT-REACHED
  elif [ "$wit" -gt 0 ]; then v=WITNESS
  else v=REACHED; fi
  printf '%-20s %-4s %-14s rc=%-4s hits=%-7s wit=%-4s %s %s\n' \
    "$bin" "$site" "$v" "$rc" "${hits:-?}" "$wit" "$kinds" "$code"
}

rm -rf /root/probe-logs; mkdir -p /root/probe-logs
echo "revocation knobs: `sysctl -n security.cheri.runtime_revocation_default security.cheri.runtime_revocation_every_free_default | tr '\n' '/'`"
printf '%-20s %-4s %-14s %s\n' CASE SITE VERDICT DETAIL
# memsys5 arm
run_one mem5design       1  0
run_one blobclose        2  0
run_one fts5rank         3  0
run_one staticbind       4  0
run_one rtreestatic      4  0
run_one wschema          5  0
run_one fz09             6  0
run_one jsoneachroot     7  0
run_one jsoneachstatic   7  0
run_one a783931794_0     8  0
run_one fts3destroyoom   9  0
run_one c7def600bd_0    10  0
run_one 2c7a73eaea_0    11  0
run_one fts3snipor      12  0
run_one fts5vocabeof    13  0
run_one fts5structwrite 14  0
run_one fts5inplace     15  0
run_one 634ac14488_0    16  0
run_one 33cf194218_0    17  0
run_one bfe33f80dd_0    18  0
run_one 8f5b14a5c2_0    19  0
run_one 415540ddaa_0    20  0
run_one adfb203a7d_0    20  0
run_one expertrem       21  0
# system-allocator arm: on purecap each allocation carries its own bounds, so the
# probe can answer spatial questions here that the arena makes unanswerable
run_one mem5design_sys       1  1
run_one blobclose_sys        2  1
run_one fts5rank_sys         3  1
run_one staticbind_sys       4  1
run_one rtreestatic_sys      4  1
run_one wschema_sys          5  1
run_one fz09_sys             6  1
run_one jsoneachroot_sys     7  1
run_one jsoneachstatic_sys   7  1
run_one a783931794_0_sys     8  1
run_one fts3destroyoom_sys   9  1
run_one c7def600bd_0_sys    10  1
run_one 2c7a73eaea_0_sys    11  1
run_one fts3snipor_sys      12  1
run_one fts5vocabeof_sys    13  1
run_one fts5structwrite_sys 14  1
run_one fts5inplace_sys     15  1
run_one 634ac14488_0_sys    16  1
run_one 33cf194218_0_sys    17  1
run_one bfe33f80dd_0_sys    18  1
run_one 8f5b14a5c2_0_sys    19  1
run_one 415540ddaa_0_sys    20  1
run_one adfb203a7d_0_sys    20  1
run_one expertrem_sys       21  1
echo "DONE"
