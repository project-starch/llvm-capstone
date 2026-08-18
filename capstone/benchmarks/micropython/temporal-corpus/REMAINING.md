# What the last 7 need

Nine of the sixteen certain temporal cases are measured in the domain. These seven
are not, and they are blocked by three different things. Only ONE of the three is
ours to fix.

## 1. A compiler fix we own: `MPY-T14`, `MPY-T15`

Blocked by a crash, not by effort. `MICROPY_VFS` needs `lib/oofatfs`, and the
compiler cannot build it:

    Assertion `VT.isVector() && "Unable to legalize non-vector shift"' failed
    in SelectionDAGLegalize::ExpandNode, on function '@f_mkfs'

Minimised to two lines in `evidence/i128-shift-crash-repro.c`: a variable-count
shift on `unsigned __int128` crashes, the same shift on `u64` does not. Verified
pre-existing by reverting all nine changed `llvm/` files to
`origin/capstone-bootstrap` and rebuilding: identical assertion, identical
function.

Same root cause as everything in `i128-capability-fixes.md`. On `capstone64` i128
is legal because it carries a capability, so a genuine 128-bit shift is never
expanded by the generic legaliser and never lowered by the target either. Round 1
covered shift-by-XLen-or-more, which produced a wrong result; this form has no
legalisation at all.

**Needed:** a lowering or expansion for variable-count i128 shifts, in the same
place as the existing i128 logical lowering. Then, separately, the VFS build work:
`extmod/vfs*.c` plus `lib/oofatfs` into the amalgamation, which also needs a
mechanism for `lib/` sources beyond libm and a way past the hardcoded `PORT` in
`build-micropython-silicon.sh`.

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
| i128 variable-shift legalisation, then VFS build | `T14`, `T15` | compiler fix we own, then build plumbing |
| hidden-root variant behind an `#ifdef` | `T29` | small, reconstruction |
| a domain with an OS | `T01`, `T04`, `T06`, `T19` | out of scope for this domain |

Ceiling with the current domain: **12 of 16**, not 16.
