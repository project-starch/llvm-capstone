#!/bin/bash
# Slot census of the two web servers the heap temporal-safety papers measure, in their published configurations:
#   nginx  $NGINX, 16 worker processes, wrk -t16 -c256 -d30s on a 64-byte file (DangZero, CCS 2022, sec. 6.5)
#   httpd  Apache $HTTPD (event MPM, its default), ApacheBench -n 100000 -c 128 (128 concurrent requests as CRCount,
#          NDSS 2019, and 100,000 requests as pSweeper, CCS 2018), on the same 64-byte file
# Both servers allocate per connection and per request from pools they carve from malloc'd blocks (nginx's ngx_pool,
# Apache's APR pools), so the census counts pool blocks and the other mallocs: the objects the allocator below sees
# with the servers unmodified ("hidden" in report.py).
#
# usage: run-webservers.sh OUT_DIR
# Downloads nginx, PCRE 8.45 (nginx 1.20 predates PCRE2; nginx compiles PCRE in with --with-pcre), wrk $WRK, Apache
# httpd, APR and APR-util into $SLOTCENSUS_CACHE (default ~/slot-census/cache) and records their sha256. Each server
# starts under the tracer (master and every worker or child it forks report: a forked process starts at the live set
# it inherits), takes the load, and is stopped gracefully so its processes exit through exit(). A cell is valid when
# the load generator reports no errors or non-2xx responses and the server stops cleanly.
set -u
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=$1
NGINX=1.20.2 PCRE=8.45 WRK=4.2.0 HTTPD=2.4.58 APR=1.7.4 APRUTIL=1.6.3
PORT_N=${SLOTCENSUS_NGINX_PORT:-18080} PORT_A=${SLOTCENSUS_HTTPD_PORT:-18081}
. "$HERE/tracer.sh"
start_pass

fetch() {   # file url: downloaded once, unpacked into $TOOL
  cached "$1" "$2"
  tar xzf "$CACHE/$1" -C "$TOOL" || exit 1
}
fetch nginx-$NGINX.tar.gz https://nginx.org/download/nginx-$NGINX.tar.gz
fetch pcre-$PCRE.tar.gz https://sourceforge.net/projects/pcre/files/pcre/$PCRE/pcre-$PCRE.tar.gz/download
fetch wrk-$WRK.tar.gz https://github.com/wg/wrk/archive/refs/tags/$WRK.tar.gz
fetch httpd-$HTTPD.tar.gz https://archive.apache.org/dist/httpd/httpd-$HTTPD.tar.gz
fetch apr-$APR.tar.gz https://archive.apache.org/dist/apr/apr-$APR.tar.gz
fetch apr-util-$APRUTIL.tar.gz https://archive.apache.org/dist/apr/apr-util-$APRUTIL.tar.gz
B=$OUT/logs/build.log
(cd "$TOOL/nginx-$NGINX" && ./configure --prefix="$TOOL/nginx" --with-pcre="$TOOL/pcre-$PCRE" > "$B" 2>&1 &&
 make -j16 >> "$B" 2>&1 && make install >> "$B" 2>&1) || { echo "run-webservers: nginx build failed" >&2; exit 1; }
make -C "$TOOL/wrk-$WRK" -j16 >> "$B" 2>&1 || { echo "run-webservers: wrk build failed" >&2; exit 1; }
# httpd needs an installed PCRE (pcre-config): the same PCRE 8.45, built once more into a private prefix
mkdir -p "$TOOL/pcre-build" && tar xzf "$CACHE/pcre-$PCRE.tar.gz" -C "$TOOL/pcre-build" --strip-components=1 &&
(cd "$TOOL/pcre-build" && ./configure --prefix="$TOOL/pcre" --disable-shared >> "$B" 2>&1 &&
 make -j16 install >> "$B" 2>&1) \
  || { echo "run-webservers: pcre build failed" >&2; exit 1; }
mv "$TOOL/apr-$APR" "$TOOL/httpd-$HTTPD/srclib/apr" && mv "$TOOL/apr-util-$APRUTIL" "$TOOL/httpd-$HTTPD/srclib/apr-util"
(cd "$TOOL/httpd-$HTTPD" &&
 ./configure --prefix="$TOOL/httpd" --with-included-apr --with-pcre="$TOOL/pcre/bin/pcre-config" >> "$B" 2>&1 &&
 make -j16 >> "$B" 2>&1 && make install >> "$B" 2>&1) \
  || { echo "run-webservers: httpd build failed" >&2; exit 1; }
{
  provenance_tracer
  (cd "$CACHE" && sha256sum nginx-$NGINX.tar.gz pcre-$PCRE.tar.gz wrk-$WRK.tar.gz httpd-$HTTPD.tar.gz apr-$APR.tar.gz \
     apr-util-$APRUTIL.tar.gz)
  echo "nginx $("$TOOL/nginx/sbin/nginx" -v 2>&1) --with-pcre; 16 workers; wrk $WRK -t16 -c256 -d30s, 64-byte file"
  echo "httpd $("$TOOL/httpd/bin/httpd" -v | head -1) $("$TOOL/httpd/bin/httpd" -V | grep -i 'Server MPM')"
  echo "ab -n 100000 -c 128, 64-byte file"
} > "$OUT/provenance.txt"
head -c 64 /dev/zero | tr '\0' 'x' > "$TOOL/file64"

: > "$OUT/cells.tsv"
# nginx: its stock configuration with 16 workers, the port, and the 64-byte file as the document root's index
N=$TOOL/nginx
cp "$TOOL/file64" "$N/html/file64"
sed -i -e 's/^worker_processes .*/worker_processes  16;/' -e "s/listen       80;/listen       $PORT_N;/" \
  "$N/conf/nginx.conf"
grep -q "worker_processes  16;" "$N/conf/nginx.conf" && grep -q "listen       $PORT_N;" "$N/conf/nginx.conf" \
  || { echo "run-webservers: nginx.conf changed" >&2; exit 1; }
t=$(date +%s)
measured "$OUT/nginx.jsonl" "$N/sbin/nginx" > "$OUT/logs/nginx.server" 2>&1
sleep 1
"$TOOL/wrk-$WRK/wrk" -t16 -c256 -d30s "http://127.0.0.1:$PORT_N/file64" > "$OUT/nginx.out" 2>&1; load=$?
"$N/sbin/nginx" -s quit; for _ in $(seq 1 100); do [ -e "$N/logs/nginx.pid" ] || break; sleep 0.2; done
stopped=1; [ -e "$N/logs/nginx.pid" ] || stopped=0
# all 16 workers must have reported (a census missing them would read as a valid, tiny peak)
workers=$(grep -c '"cmdline":"nginx: worker process' "$OUT/nginx.jsonl")
v=INVALID
[ $load = 0 ] && [ $stopped = 0 ] && [ "$workers" -ge 16 ] && ! grep -qE "Socket errors|Non-2xx" "$OUT/nginx.out" \
  && v=valid
record_cell nginx 're:^nginx: (master|worker) process' "$v" \
  "wrk=$load stopped=$stopped workers=$workers $(took $t)"

# httpd: its stock configuration with the port and the 64-byte file
H=$TOOL/httpd
cp "$TOOL/file64" "$H/htdocs/file64"
sed -i "s/^Listen 80$/Listen $PORT_A/" "$H/conf/httpd.conf"
grep -q "^Listen $PORT_A$" "$H/conf/httpd.conf" || { echo "run-webservers: httpd.conf changed" >&2; exit 1; }
t=$(date +%s)
measured "$OUT/httpd.jsonl" "$H/bin/apachectl" -k start \
  > "$OUT/logs/httpd.server" 2>&1
sleep 1
"$H/bin/ab" -n 100000 -c 128 "http://127.0.0.1:$PORT_A/file64" > "$OUT/httpd.out" 2>&1; load=$?
"$H/bin/apachectl" -k graceful-stop; for _ in $(seq 1 150); do [ -e "$H/logs/httpd.pid" ] || break; sleep 0.2; done
stopped=1; [ -e "$H/logs/httpd.pid" ] || stopped=0
failed=$(sed -n 's/^Failed requests: *//p' "$OUT/httpd.out"); non2xx=$(grep -c "Non-2xx" "$OUT/httpd.out")
v=INVALID
[ $load = 0 ] && [ $stopped = 0 ] && [ "${failed:-x}" = 0 ] && [ "$non2xx" = 0 ] && v=valid
record_cell httpd httpd "$v" "ab=$load failed=${failed:-?} stopped=$stopped $(took $t)"
finish_pass
