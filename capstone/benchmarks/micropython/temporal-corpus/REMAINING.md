# What the last 7 need

Ten of the sixteen certain temporal cases are measured in the domain. These six
are not, and they are blocked by three different things. Only ONE of the three is
ours to fix.

## 1. Build plumbing, not a compiler fix: `MPY-T14`, `MPY-T15`

CORRECTED 2026-08-18. This section previously said the compiler could not build
`lib/oofatfs` and that a codegen fix was the blocker. That was wrong: the probe
behind it omitted `-Xclang -target-feature -Xclang +m`, which the domain build
always passes (`build-micropython-silicon.sh:149`). With the build's own flags
`ff.c` compiles, 69400 bytes, and so does the rest of the stack -- `ffunicode.c`,
`vfs.c`, `vfs_fat_diskio.c`, `vfs_fat_file.c`, `vfs_reader.c`, `vfs_blockdev.c`.
`vfs_fat.c` stops at `#error "MICROPY_VFS_FAT requires MICROPY_VFS"`, which is a
configuration switch. Full account in
`evidence/oofatfs-backend-crash-2026-08-18.txt`.

**Needed, all of it build work:** `MICROPY_VFS=1` and `MICROPY_VFS_FAT=1` in
`port/mpconfigport.h`; `extmod/vfs*.c` plus `lib/oofatfs` into the amalgamation,
which needs a mechanism for `lib/` sources beyond libm and a way past the
hardcoded `PORT` in `build-micropython-silicon.sh`; and a block device for the
domain to mount, which is where the actual design question sits.

**Not needed:** the i128 shift work. A fix for it landed anyway
(`CapstoneISelLowering.cpp`, `srl(mul(zext,zext), XLen)` -> `mulhu`/`mulh`, and a
clean diagnostic instead of an assertion on the forms that cannot be lowered),
but it changes no FatFs codegen: with `+m` the object is byte-identical with and
without it.

**Worth it because** these two would be genuine upstream reproductions, not
reconstructions.

## 2. Reachable as a reconstruction: `MPY-T29`

`#4705`'s fix is `unix/gccollect: Make sure stack/regs get captured properly for
GC`, and it is port-specific, so there is no upstream program to run here.

But our port has the same shape of gap. `mpy_domain.c:102`:

    void gc_collect(void) {
        gc_collect_start();
        gc_collect_root(sp_now, (top - sp_now) / sizeof(void *));   /* stack only */
        gc_collect_end();
    }

Only the C stack is scanned; registers are not captured. At `-O0`, which is what
the domain builds at, almost everything is spilled, so this is unlikely to bite by
accident, and no claim is made that it does.

**Needed:** the `MPY-T07`/`MPY-T16` treatment. A variant behind its own `#ifdef`
that deliberately hides the only reference to a live object from the root scan,
forces a collect, and then uses the pointer. Small, and it lands as a
reconstruction like the other two.

## 3. Needs a different domain, not more work: `MPY-T01`, `MPY-T04`, `MPY-T06`, `MPY-T19`

- `MPY-T01`, `MPY-T04`: the defect is at `modselect.c:151`, inside
  `#if MICROPY_PY_SELECT_POSIX_OPTIMISATIONS`, built on `poll()` and
  `struct pollfd`. It needs real OS file descriptors. The fix touches only that
  block, so there is no non-POSIX path carrying the same defect.
- `MPY-T06`: needs berkeley-db compiled freestanding. And the property that makes
  this row valuable is that berkeley-db mallocs OUTSIDE the GC heap, which is why
  it is the only row here that crashes on stock while its GC-managed twins stay
  silent. Porting it into the domain would change exactly that.
- `MPY-T19`: needs the NimBLE stack.

A domain with file descriptors and a real libc is a Linux process under Capstone,
which is a different experiment from a confined domain, not a harder version of
this one.

## Summary

| what | unlocks | kind of work |
|---|---|---|
| VFS build plumbing: config, `lib/` sources, a block device | `T14`, `T15` | build work we own; NOT a compiler fix |
| ~~hidden-root variant behind an `#ifdef`~~ | `T29` | DONE, measured 0x29007701 |
| a domain with an OS | `T01`, `T04`, `T06`, `T19` | out of scope for this domain |

Ceiling with the current domain: **12 of 16**, not 16.
