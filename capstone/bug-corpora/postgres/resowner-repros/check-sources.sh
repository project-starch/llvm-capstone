#!/usr/bin/env bash
# The ten files this corpus COMPILES, by sha256 at the pin.
#
#   check-sources.sh <tree>
#
# resowner.c is the subject of both cases, so "which release" is not a detail
# here: the gate is that the bytes compiled are the pin's bytes.  It fails on
# any mismatch rather than warning, because a warning about the subject of a
# corpus is not a gate.
#
# All ten are byte-identical in REL_17_0 and REL_17_5 (checked 2026-11-10), so
# this one list covers both: the defect pin (17.5) and the memory-context
# port's allocator pin (17.0).  That is why these cases can be built by
# ports/postgres/memory-contexts unchanged.
set -uo pipefail
TREE=${1:-}
if [ -z "$TREE" ] || [ ! -d "$TREE" ]; then
    echo "usage: check-sources.sh <postgresql source tree>" >&2
    exit 75
fi

read -r -d '' SUMS <<'EOF'
0fe4cf4ff97f70ce31c984b568fe3d2433c52d8da538b062e6793a12a1f9999c  src/backend/utils/mmgr/aset.c
db80bd6c58acb0ab9cdc45ded17bb32afbbd29fec5f1471b82e4c60c1fe9a671  src/backend/utils/mmgr/mcxt.c
24dc37b6dcc99f510515b3334b361b7d68c805a1e435634717733760cc880571  src/backend/utils/mmgr/generation.c
7e00aeaf86d31c8a28975bafff91c9baceb00d4124dd1416e090cac0f6d7ea05  src/backend/utils/mmgr/slab.c
a0316ed75811049178fa6cb3365568968389800fa983af8b706c6086bed53722  src/backend/utils/mmgr/bump.c
a2d1d8512a24469fb259f46a2dbdd028bdb676036fa913d210278fea4eeb8170  src/backend/utils/mmgr/alignedalloc.c
4d7a4a17e1bdf4575bc69f4e2a55060df2193cb764bad830ca9fb3bd4e6cc01a  src/backend/utils/mmgr/memdebug.c
63ebe04cf07b2af3f050c8e3fd99501b59b55565b338a8869fd5c75457224d09  src/backend/utils/resowner/resowner.c
3d9c31b6df37e104846fb435ed8035cd916dc314dfe256d09967d39b3acfb7d8  src/common/hashfn.c
0ccc09db9a7adf0e9842bea024e5ac5ad070b8ce4788d9991204abf759d23e42  src/port/qsort.c
EOF

bad=0
while read -r want path; do
    [ -z "${path:-}" ] && continue
    if [ ! -f "$TREE/$path" ]; then
        printf '%-46s MISSING\n' "$path"; bad=1; continue
    fi
    got=$(sha256sum -- "$TREE/$path"); got=${got%% *}
    if [ "$got" = "$want" ]; then
        printf '%-46s ok\n' "$path"
    else
        printf '%-46s MISMATCH (got %s)\n' "$path" "${got:0:16}..."; bad=1
    fi
done <<< "$SUMS"

if [ $bad -ne 0 ]; then
    echo "check-sources: this tree is not the pin for the files the corpus compiles" >&2
    exit 1
fi
echo "check-sources: all ten files match the pin"
