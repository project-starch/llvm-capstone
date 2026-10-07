#!/usr/bin/env python3
"""Apply the CHERI-required source fixes to the pristine SQLite 3.22.0 amalgamation.

These are the four rewrites from the Capstone arm's adapt-sqlite-322.sh that are
pure purecap-ABI requirements, not Capstone quirks: SQLite 3.22.0 predates CHERI and
assumes 8-byte pointers, so buffers that receive capabilities end up 8-byte aligned
and structs holding them are undersized. Without these the pristine source SIGBUSes
as soon as a database is opened.

Deliberately NOT applied: the `sqlite3_filename` typedef backport. That exists only so
the Capstone tree's shared VFS (written against SQLite 3.41+) compiles; this arm uses
CheriBSD's real unix VFS.
"""
import os
import pathlib, sys

# The pinned 3.22.0 amalgamation is shared by all three arms and is not
# repository content; override with SQLITE_AMALGAMATION if it lives elsewhere.
_AMALG = pathlib.Path(os.environ.get(
    'SQLITE_AMALGAMATION', '/home/zephyr/arms/sqlite/shared/amalgamation'))
SRC = _AMALG / 'sqlite3.c'
DST = _AMALG / 'sqlite3-cheri.c'

REWRITES = [
    # 1. The Parse tail copied into saveBuf contains capabilities; 16-byte align it.
    ('  char saveBuf[PARSE_TAIL_SZ];',
     '  char saveBuf[PARSE_TAIL_SZ] __attribute__((aligned(16)));'),
    # 2+3. allocateCursor rounds the VdbeCursor+aType area with ROUND8; capabilities
    #      stored after it need 16.
    ('ROUND8(sizeof(VdbeCursor)) + 2*sizeof(u32)*nField + ',
     '((ROUND8(sizeof(VdbeCursor)) + 2*sizeof(u32)*nField + 15)&~15) + '),
    ('&pMem->z[ROUND8(sizeof(VdbeCursor))+2*sizeof(u32)*nField]',
     '&pMem->z[(ROUND8(sizeof(VdbeCursor))+2*sizeof(u32)*nField+15)&~15]'),
    # 4. RowSet allocation hardcodes 64 bytes; with 16-byte pointers the struct is larger.
    ('pMem->zMalloc = sqlite3DbMallocRawNN(db, 64);',
     'pMem->zMalloc = sqlite3DbMallocRawNN(db, ROUND8(sizeof(RowSet)));'),
]

s = SRC.read_text()
if 'saveBuf[PARSE_TAIL_SZ] __attribute__' in s:
    sys.exit('refusing to adapt an already-adapted source')
for old, new in REWRITES:
    n = s.count(old)
    if n == 0:
        sys.exit(f'rewrite not found (source is not the official 3.22.0 amalgamation?): {old[:60]}')
    s = s.replace(old, new)
    print(f'  applied x{n}: {old[:62]}')
DST.write_text(s)
# assert each landed
for _, new in REWRITES:
    assert new in s, new
print(f'wrote {DST} ({len(s.splitlines())} lines)')
