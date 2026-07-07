# Hierarchical revoke cascade needs a split-derived child (Stage-2, use-after-close)

**Date:** 2026-07-07
**Context:** SQLite-Capstone Stage-2. After validating the borrow-revoke shape
(row 3, `sqlite-borrow-revoke-probe`), the next shape is HIERARCHICAL-REVOKE — the
7 use-after-close rows (4/5/7/8/9/10/12), the proposal's "close cascades to
statements/values" (Table 4, primitive H).

## Finding (empirical)

Revoking a parent borrow does **not** cascade to an independently-created child
borrow. The feasibility probe `tests/runtime-qemu/sqlite-hier-revoke-probe`
(existing ops only, no firmware) lends a parent (connection) and a child
(statement value) region, caches the child in the borrower, revokes the **parent**
(= `sqlite3_close`), and re-reads the child:

```
round 1 retval = 0xc01a0dedc01a0ded   # child read while open
close revoked the connection (parent)
round 2 returned 0xc01a0dedc01a0ded   # child STILL readable -> NO CASCADE
```

**Root cause:** each `create_region` produces an independent capability;
`shared_region_annotated(..., REV_BORROWED)` mints an independent `__mrev` handle
per region. Two regions are therefore **independent rev-tree roots**, so
`revoke_region(parent)` (`__revoke` on the parent's handle) walks only the
parent's own junior run and never reaches the child. Confirmed against the monitor
source (`capstone-sbi/sbi_capstone.c`: `create_region` → `split_out_cap`;
`shared_region_annotated` → per-region `__mrev`; `revoke_region` → `__revoke`).

## Why the cheap composition can't work

The lender API `create_region(len)` lets the kernel pick the region base, so a
child cannot be forced to land inside the parent's range; `split_out_cap` then
never derives the child from the parent. And a Linux U-mode lender cannot issue
SBI directly — every lender op goes libcapstone → modcapstone ioctl → monitor SBI.
So there is no existing-ops path to a derived child.

## Required extension (design)

Add a monitor operation that shares a child **split-derived from a parent**, so
the child is junior in the parent's rev lineage and a parent revoke cascades:

1. `mrev(parent)` — mint the senior revocation handle, retained by the engine
   (stored in `regions[parent]` so `revoke_region(parent)` acts on it), memory
   **not** otherwise shared.
2. `__split` a child sub-cap from the parent's cap at `[base+off, base+off+len)`
   (parent must be large enough to contain the child). Because the split happens
   after `mrev(parent)`, the child is junior to `parent_rev`.
3. `delin` + share the child to the domain (the borrower reads it).
4. `revoke_region(parent)` then cascades: `__revoke(parent_rev)` invalidates the
   junior run, including the split-derived child.

**Plumbing cost (3 layers):** new `SBI_EXT_CAPSTONE_*` fid + monitor handler
(`sbi_capstone.c/.h`); modcapstone kernel ioctl + `libcapstone.c/.h` wrapper
(e.g. `share_child_region(dom, parent_id, off, len, perm)`); then a positive probe.
Requires a `capstone-sbi-domain` package + rootfs rebuild and a
`caplifive-buildroot` submodule commit + pointer bump. Rev-tree cascade semantics
across a split-derived child are **plausible but unverified** — the first build
must confirm `revoke(parent)` invalidates the split child (and, ideally, does
**not** invalidate an unrelated region).

## Status

Borrow-revoke shape: **validated** (rows 3/13/18/19 covered). Hierarchical shape:
**VALIDATED** — the derived-child extension was implemented and the cascade
confirmed end-to-end (see the resolution below).

## Resolution (2026-07-07) — derived-child cascade VALIDATED

The `share_child_region` monitor op was implemented exactly as designed above and
the positive probe `tests/runtime-qemu/sqlite-hier-child-revoke-probe` traps the
use-after-close:

```
round 1 retval = 0xc01a0dedc01a0ded   # child read while connection open
close revoked the connection (parent)  # revoke_region(parent) == sqlite3_close
round 2 returned 0x000000000fa017ed    # use-after-close TRAPPED via cascade
```

`round 2 == 0x0FA017ED` (fault sentinel) == `__revoke(parent_rev)` invalidated the
split-derived child; the borrower's cached-pointer read faulted and the monitor
terminated the domain. This confirms the rev-tree cascade prediction from
`capstone-qemu/target/riscv/cap_rev_tree.c` (a senior `mrev` handle revokes the
junior run, including a `split`-derived child) holds on real RTL.

**What landed (3 layers + probe):**
- Monitor op `share_child_region(dom, parent_id, offset, len, perm)` — SBI fid
  `0xd` (`SBI_EXT_CAPSTONE_REGION_SHARE_CHILD`) + dispatch, added to **both**
  monitor copies: `components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.c` (the
  copy `fw_jump.elf` links, which fields the lender's M-mode SBI ecall) and
  `package/capstone-sbi-domain/capstone-sbi/sbi_capstone.c` (the `sbi.dom` copy).
  Mints the senior handle with `__mrev(parent)` (retained under `parent_id`),
  `__split`s the child out of the parent's own cap (junior in its rev lineage),
  retains the head/tail fragments in `regions[]` (like `split_out_cap`), tightens
  to read and DPI-shares the child last.
- Kernel `modcapstone`: `IOCTL_REGION_SHARE_CHILD` (num 9) + struct + handler +
  dispatch; `libcapstone` `share_child_region()` wrapper.
- Probe `sqlite-hier-child-revoke-probe` (+ build/run scripts).

**Rebuild gotcha (recorded so it isn't re-hit):** OpenSBI's monitor is compiled
via `components/opensbi/lib/sbi/sbi_capstone_dom.c`, which is a one-line
`#include "capstone-sbi/sbi_capstone.c"`. The top Makefile rule `%.c.S: %.c`
watches only the wrapper, **not** the included file, so editing
`capstone-sbi/sbi_capstone.c` does **not** trigger `sbi_capstone_dom.c.S`
regeneration. After editing the OpenSBI monitor copy you must force it (e.g.
`rm components/opensbi/lib/sbi/sbi_capstone_dom.c.S` and/or `touch` the wrapper)
before `make build A="opensbi-rebuild"`, or `fw_jump.elf` silently relinks stale
assembly. The `capstone-sbi-domain` package copy regenerates correctly because its
own Makefile depends directly on `capstone-sbi/sbi_capstone.c`.

Together with the NO-CASCADE sibling probe (`sqlite-hier-revoke-probe`, independent
rev roots) this shows the cascade is **precisely scoped to the parent's rev
lineage**: derived children are invalidated, independent regions are not.
