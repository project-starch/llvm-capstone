#!/usr/bin/env python3
"""Turn the Sublet memsys5 port (already patched, usually after sublet-block-bounds.py) spatial-only.

    sublet-spatial-only.py SOURCE      (the patched sqlite3-capstone.c, edited in place)

A comparison arm, not the port: the control for "Sublet spatial + temporal" in the reuse-distance
comparison with PoisonCap's own off/on switch (NESTED_TEMPORAL). Every block memsys5 hands out and
every lookaside slot is still a CDERIVE child bounded to its block or slot (spatial), but sqlite3_free
and the lookaside release no longer revoke it (CREVOKE): a stale pointer keeps working (no temporal
protection). The revocations of the whole pool at shutdown and of a connection's lookaside generation
stay, as do the rounding and every other line. Each edit must apply exactly once.

On the block port (sublet-3220000-memsys5-block.patch, recognised by its memsys5Renew), a realloc
that keeps its block or its slot no longer renews the lifetime (CRENEW) either: the old child stays
live, and the block or slot goes out again as one more child (CDERIVE) with the bounds the port
gives it.

Revocation moves no memsys5 address, so this arm's reuse distances are expected to equal the
temporal arm's; measuring that is its purpose. Without per-object revocation every hand-out keeps its
lifetime node, and a reissued block is derived while its previous child is still live.
"""
import sys

EDITS = [
    ("  /* All address calculations precede the lifetime end. */\n"
     "  capstone_cap_release(pOld);\n",
     "  /* spatial-only arm: the caller's block is NOT revoked; its child stays live. */\n"),
    ("  capstone_cap_release(p);\n",
     "  /* spatial-only arm: the slot is NOT revoked. */\n"),
]

BLOCK_PORT_EDITS = [
    # memsys5Renew: the same block as a new child of the pool.
    ("  return capstone_cap_renew(pPrior);\n",
     "  /* spatial-only arm: the old child is NOT ended; the block goes out again as a new child. */\n"
     "  int i, iSize;\n"
     "  void *p;\n"
     "  memsys5Enter();\n"
     "  i = (int)(((u8 *)pPrior-mem5.zPool)/mem5.szAtom);\n"
     "  iSize = memsys5Size(pPrior);\n"
     "  p = capstone_cap_derive(mem5.zPool, (unsigned long)i*mem5.szAtom, iSize);\n"
     "  memsys5Leave();\n"
     "  return p;\n"),
    # lookasideRenew: the same slot as a new child of pLife, which the connection holds;
    # lookasideRelease no longer ends the old one.
    ("static void *lookasideRenew(void *p){\n"
     "#ifdef SUBLET_REALLOC_KEEPS_LIFETIME\n"
     "  return p;\n"
     "#else\n"
     "  return capstone_cap_renew(p);\n",
     "static void *lookasideRenew(sqlite3 *db, void *p){\n"
     "#ifdef SUBLET_REALLOC_KEEPS_LIFETIME\n"
     "  return p;\n"
     "#else\n"
     "  return lookasideIssue(db, lookasideRelease(db, p));\n"),
    ("    void *pNew = lookasideRenew(p);\n",
     "    void *pNew = lookasideRenew(db, p);\n"),
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
