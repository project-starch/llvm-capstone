#!/usr/bin/env python3
"""Turn the Sublet memsys5 port (already patched, usually after sublet-block-bounds.py) spatial-only.

    sublet-spatial-only.py SOURCE      (the patched sqlite3-capstone.c, edited in place)

A comparison arm, not the port: the control for "Sublet spatial + temporal" in the reuse-distance
comparison with PoisonCap's own off/on switch (NESTED_TEMPORAL). Every block memsys5 hands out and
every lookaside slot is still a CDERIVE child bounded to its block or slot (spatial), but sqlite3_free
and the lookaside release no longer revoke it (CREVOKE): a stale pointer keeps working (no temporal
protection). The revocations of the whole pool at shutdown and of a connection's lookaside generation
stay, as do the rounding and every other line. Each edit must apply exactly once.

On the block port (sublet-3220000-memsys5-block.patch, recognised by its memsys5Renew), the renewal
of a realloc that keeps its block no longer revokes the old child either; it still derives the new
one, so the arm makes the same calls as the port. The lookaside renewal goes through
lookasideRelease and needs no edit of its own.

Revocation moves no memsys5 address, so this arm's reuse distances are expected to equal the
temporal arm's; measuring that is its purpose. Without per-object revocation every hand-out keeps its
lifetime node, and a reissued block is derived while its previous child is still live.
"""
import sys

EDITS = [
    ("  /* End the caller's block (CREVOKE); faults on a pointer that is not live. */\n"
     "  capstone_cap_revoke_child(mem5.zPool, pOld);\n",
     "  /* spatial-only arm: the caller's block is NOT revoked; its child stays live. */\n"),
    ("static LookasideSlot *lookasideRelease(sqlite3 *db, void *p){\n"
     "  capstone_cap_revoke_child(db->lookaside.pLife, p);\n",
     "static LookasideSlot *lookasideRelease(sqlite3 *db, void *p){\n"
     "  /* spatial-only arm: the slot is NOT revoked. */\n"),
]

BLOCK_PORT_EDITS = [
    ("  iSize = memsys5Size(pPrior);\n"
     "  capstone_cap_revoke_child(mem5.zPool, pPrior);\n",
     "  iSize = memsys5Size(pPrior);\n"
     "  /* spatial-only arm: the old child is NOT revoked; it stays live. */\n"),
]


def main():
    path = sys.argv[1]
    text = open(path).read()
    if 'static void *memsys5Renew(' in text:
        edits = EDITS + BLOCK_PORT_EDITS
    else:
        edits = EDITS
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f'{path}: expected exactly one {old.strip()[:60]!r}, found {text.count(old)}')
        text = text.replace(old, new)
    open(path, 'w').write(text)
    print(f'spatial-only: {len(edits)} edits applied to {path}')


if __name__ == '__main__':
    main()
