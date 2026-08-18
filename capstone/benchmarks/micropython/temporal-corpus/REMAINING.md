# What the last 6 need

Ten of the sixteen certain temporal cases are measured in the domain. These six
are not. Two of them are now one backport away; the other four need a different
domain, and that claim is checked below rather than asserted.

## 1. DONE except a parent build: `MPY-T14`, `MPY-T15`

CORRECTED TWICE, both on 2026-08-18. First this section said a compiler crash
blocked `lib/oofatfs`; the probe behind that had simply omitted `+m`, which the
domain build always passes. Then it said the remaining work included "a block
device for the domain to mount, which is where the actual design question sits".
That was reading the issues rather than their PoCs: both reproductions define
their block device in Python.

The build work is done and measured. In a pure-capability domain under QEMU:

    0000-00_sanity.py.actual     SANE 2
    0001-01_vfs_smoke.py.actual  VFS hello 32768

a FAT filesystem formatted, mounted, written and read back over a 32 KiB RAM block
device written in Python. Behind `MPY_VFS=1`; with it unset the default image is
byte-identical (md5 `bc1f806a8179ca94` before and after). Full account, including
the capability bug and the typedef collision it turned up, in
`evidence/vfs-runs-in-the-domain-2026-08-18.txt`.

**What is left:** a parent build. Both rows are fixed at the pin -- the
reproduction raises `ValueError: lhs and rhs should be compatible` there, verified
on the host -- so the domain has to be built from `64f0394d80ca^` = `4e6dc0b569`
(2026-05-09). That is the procedure `backport-2024/` already established for
`MPY-T02` and `MPY-T05`, against a gap of three months rather than two years.

**Worth it because** these two would be genuine upstream reproductions, not
reconstructions.

## 2. Done: `MPY-T29`

Measured 2026-08-18, `retval 0x29007701`, as a reconstruction behind
`MPY_T29_HIDDEN_ROOT`. See its case directory.

## 3. Needs a different domain, and here is the check: `MPY-T01`, `MPY-T04`, `MPY-T06`, `MPY-T19`

This section previously asserted these four away. Given that the section above it
had to be corrected twice, each claim was re-checked against the source.

- **`MPY-T01`, `MPY-T04`** -- VERIFIED BLOCKED. The fix, `8b24aa36ba97`, is
  `m_renew` on `poll_set->pollfds` while `poll_obj_t->pollfd` points into it: a
  dangling interior pointer after a realloc that moves. `poll_set_add_fd` and the
  whole `struct pollfd` half sit inside `#if MICROPY_PY_SELECT_POSIX_OPTIMISATIONS`
  (`extmod/modselect.c:115`--`211`), and the `#else` arm has no pollfds array at
  all, so there is no non-POSIX path carrying the defect. It needs real OS file
  descriptors.
  Worth knowing anyway: the SHAPE -- an interior pointer surviving a reallocation
  -- is already measured three times over, by `MPY-T09`, `MPY-T10` and `MPY-T13`.
  A reconstruction here would add a fourth instance of a mechanism, not a new one,
  which is what separates it from `MPY-T29`.

- **`MPY-T06`** -- VERIFIED BLOCKED, and more cheaply than expected: `lib/berkeley-db-1.xx`
  is not even checked out in the pinned tree. Beyond that, the property that makes
  this row worth having is that berkeley-db allocates OUTSIDE the GC heap, which is
  why it is the only row here that crashes on stock while its GC-managed twins stay
  silent. Porting it into the domain heap would remove exactly that property.

- **`MPY-T19`** -- VERIFIED BLOCKED for the upstream program: it needs NimBLE.
  But note what its mechanism is -- "gc.collect() collects a BLE object still
  referenced only from C", i.e. a live object invisible to the root scan. That is
  the same mechanism `MPY-T29` now measures. So this row is blocked as a
  reproduction and covered as a finding.

A domain with file descriptors and a real libc is a Linux process under Capstone,
which is a different experiment from a confined domain, not a harder version of
this one.

## Summary

| what | unlocks | kind of work |
|---|---|---|
| ~~VFS build plumbing~~ DONE; a parent build at `4e6dc0b569` remains | `T14`, `T15` | backport, as for `T02`/`T05` |
| ~~hidden-root variant behind an `#ifdef`~~ | `T29` | DONE, measured 0x29007701 |
| a domain with an OS | `T01`, `T04`, `T06`, `T19` | out of scope for this domain |

Ceiling with the current domain: **12 of 16**, not 16.
