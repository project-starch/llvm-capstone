#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$here/../../../tests/capstone-test-env.sh"
work=$(mktemp -d "$CAPSTONE_TMP_ROOT/sublet-blocks.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
python3 - "$here/../../patches/0040-riscv-sublet-blocks.patch" "$work" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text().splitlines()
name = 'arch/riscv/mm/sublet-block-state.h'
i = text.index('+++ b/' + name) + 2
lines = []
while i < len(text) and not text[i].startswith('--- '):
    if text[i].startswith('+'):
        lines.append(text[i][1:])
    i += 1
header = '\n'.join(lines) + '\n'
assert 'if (r->depth != 1)' in header
out = Path(sys.argv[2])
(out / 'sublet-block-state.h').write_text(header)
(out / 'mutant').mkdir()
(out / 'mutant/sublet-block-state.h').write_text(
    header.replace('if (r->depth != 1)', 'if (false)'))
PY
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -I"$work" "$here/state.c" -o "$work/state"
"$work/state"
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -I"$work/mutant" "$here/state.c" -o "$work/mutant/state"
if "$work/mutant/state" > "$work/control.log" 2>&1; then
    echo 'FAIL: removing the depth-1 condition escaped the oracle' >&2
    exit 1
fi
cat "$work/control.log"
echo 'PASS: depth-2 whole-block unmap negative control failed as required'
