#!/usr/bin/env bash
# Build the toolchain for tarantool #7657.
#
# Tarantool is a full DBMS with a bundled LuaJIT + many submodules; a from-source
# build is large and slow. But this bug is a plain SIGSEGV (not an ASan-only
# report), so it reproduces on the OFFICIAL prebuilt release images — no source
# build needed. We pull one VULNERABLE tag (2.8.3, pre-#7664) and one FIXED tag
# (2.11, which carries the #7664 backport). Both ship the builtin `merger`
# module. Reproduced on 2026-08-03.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
W="$HERE/.work"; mkdir -p "$W"

VULN_IMG=tarantool/tarantool:2.8.3      # 2.8.3-0-g01023dbc2 — pre-fix, vulnerable
FIX_IMG=tarantool/tarantool:2.11        # 2.11.5-0-g12a9ceb870 — has #7664 backport

command -v docker >/dev/null || { echo "docker required" >&2; exit 2; }

for img in "$VULN_IMG" "$FIX_IMG"; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "present: $img"
  else
    echo "== pulling $img =="
    docker pull "$img"
  fi
done

# Sanity: both images must expose the builtin merger module.
for img in "$VULN_IMG" "$FIX_IMG"; do
  docker run --rm --entrypoint tarantool "$img" \
    -e 'local ok,m=pcall(require,"merger"); assert(ok and m.new_table_source, "no merger"); print("merger OK"); os.exit()' \
    >/dev/null || { echo "merger module missing in $img" >&2; exit 1; }
done
echo "built: vuln=$VULN_IMG fixed=$FIX_IMG (prebuilt release images, merger present)"
