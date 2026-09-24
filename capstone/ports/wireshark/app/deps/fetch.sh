# Sourced: ts_fetch <name> -> prints the verified tarball's path. The digest is deps.json's,
# which records where it was read; a mismatch is an error, never a warning.
ts_fetch() {
  local name=$1 url sha ver tar
  read -r url sha ver < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1]))[sys.argv[2]]; print(d["url"], d["sha256"], d["version"])' \
                           "$TS_DEPS_DIR/deps.json" "$name") || return 1
  tar="$TS_DEPS_SRC/$(basename "$url")"
  if [[ ! -f "$tar" ]]; then
    curl -sSfL --retry 5 --retry-all-errors --retry-delay 3 -o "$tar.part" "$url" || return 1
    mv "$tar.part" "$tar"
  fi
  echo "$sha  $tar" | sha256sum -c --quiet - >&2 || { echo "ts_fetch: $name: sha256 MISMATCH for $tar" >&2; return 1; }
  echo "$tar"
}
