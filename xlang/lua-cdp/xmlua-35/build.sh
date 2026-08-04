#!/usr/bin/env bash
# xmlua #35 is pure Lua + LuaJIT-FFI (no compile). Ensure luajit/libxml2/valgrind
# and clone xmlua + its luacs dependency at HEAD (the issue is OPEN -> HEAD is the
# vulnerable tree).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); W="$HERE/.work"; mkdir -p "$W"
command -v luajit >/dev/null && command -v valgrind >/dev/null && pkg-config --exists libxml-2.0 \
  || sudo apt-get install -y luajit libxml2-dev valgrind
[ -d "$W/xmlua" ] || git clone https://github.com/clear-code/xmlua "$W/xmlua"
[ -d "$W/luacs" ] || git clone https://github.com/clear-code/luacs "$W/luacs"
echo "ready: xmlua + luacs cloned"
