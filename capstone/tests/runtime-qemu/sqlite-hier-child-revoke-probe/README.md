# SQLite hierarchical derived-child revoke probe (use-after-close)

The Stage-2 **"after"** for the use-after-close class (cve-repros rows
4/5/7/8/9/10/12, HIERARCHICAL-REVOKE): a statement/value pointer that lives
*inside* a SQLite connection, dereferenced after `sqlite3_close(connection)`. The
proposal's **H** primitive — closing the connection invalidates every pointer
beneath it.

This is the **positive** counterpart to `sqlite-hier-revoke-probe` (which showed
that two independent `create_region()`s are independent rev-tree roots, so a
parent revoke does **not** cascade — NO-CASCADE). Here the child is **derived**
from the parent with a new monitor op, so the cascade holds.

## What the new monitor op does

`share_child_region(dom, parent_id, offset, len, perm)` (SBI fid `0xd`,
`IOCTL_REGION_SHARE_CHILD`):

1. `__mrev(parent)` mints a **senior** revocation handle, retained by the engine
   under `parent_id` (so `revoke_region(parent)` acts on it); the parent cap drops
   to depth `d+1` (junior).
2. `__split` carves the child `[base+offset, base+offset+len)` out of the parent's
   own cap. The split products stay at depth `d+1`, **junior** to the senior
   handle. Head/tail fragments are retained in `regions[]` (like `split_out_cap`).
3. The child is `__tighten`ed (read) and shared to the domain last, so the
   borrower's `REGION_COUNT-1` query resolves to it.
4. `revoke_region(parent)` → `__revoke(parent_rev)` then walks the junior run and
   invalidates the derived child.

## Flow

- Engine writes the column value **inside** the connection at `CHILD_OFFSET`, then
  `share_child_region` lends the child sub-window.
- Round 1: host reads the child (`0xC01A0DED…`) and caches the pointer.
- Engine `revoke_region(parent)` == `sqlite3_close`.
- Round 2: host re-reads the cached child pointer = the use-after-close.

## Result (2026-07-07) — TRAPPED (cascade validated)

```
sqlite-hier-child: statement value (child) derived + borrowed from connection
sqlite-hier-child: round 1 retval = 0xc01a0dedc01a0ded
sqlite-hier-child: host read statement value OK before close
sqlite-hier-child: close revoked the connection (parent)
sqlite-hier-child: round 2 returned 0x000000000fa017ed
sqlite-hier-child: use-after-close TRAPPED via hierarchical cascade ...
```

`round 2 == 0x0FA017ED` (fault sentinel) == the parent revoke cascaded to the
derived child; the monitor terminated the domain on the use-after-close read.

## Control

Scoping (a parent revoke does **not** reach unrelated regions) is established by
the sibling probe `sqlite-hier-revoke-probe` (independent rev roots, NO-CASCADE).
Together the two probes show the cascade is **precisely scoped to the parent's rev
lineage**: derived children are invalidated, independent regions are not.

Build/run: `../build-sqlite-hier-child-revoke-probe.sh`,
`../run-sqlite-hier-child-revoke-probe.sh`.

Requires firmware rebuilt with the `share_child_region` op (monitor fid `0xd` +
`IOCTL_REGION_SHARE_CHILD`); see the dated history note
`agent-handoff/history/07-07-2026_*_hierarchical-cascade-*`.
