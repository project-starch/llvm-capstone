# Sourced by run-ycsb.sh: build libevent $LIBEVENT static into $TOOL/libevent (memcached needs it; the measuring
# hosts have no libevent headers). Needs OUT, TOOL and CACHE; sets LIBEVENT_SHA.

LIBEVENT=2.1.12-stable

build_libevent() {   # into $TOOL/libevent, static; sets LIBEVENT_SHA
  local E=libevent-$LIBEVENT.tar.gz
  [ -s "$CACHE/$E" ] || curl -fsSL -o "$CACHE/$E" \
    "https://github.com/libevent/libevent/releases/download/release-$LIBEVENT/$E" || exit 1
  tar xzf "$CACHE/$E" -C "$TOOL" || exit 1
  (cd "$TOOL/libevent-$LIBEVENT" && ./configure --prefix="$TOOL/libevent" --disable-shared --disable-openssl \
     > "$OUT/libevent-build.log" 2>&1 && make -j16 install >> "$OUT/libevent-build.log" 2>&1) \
    || { echo "libevent: build failed" >&2; exit 1; }
  LIBEVENT_SHA=$(cd "$CACHE" && sha256sum "$E" | cut -c1-64)
}
