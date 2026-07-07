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
**needs the derived-child extension above** — a real firmware spike, not a clone.
Decision on whether to invest now vs. use a coarser single-arena approximation is
pending (see the Stage-2 strategy note in
`plans/sqlite-capstone-cve-corpus-plan.md`).
