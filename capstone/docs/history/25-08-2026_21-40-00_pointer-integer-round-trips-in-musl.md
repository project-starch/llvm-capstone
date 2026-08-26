# The pointer/integer round trips musl's own compile hides

The survey compiles with `-w -Wno-int-conversion` and reports 1355 of 1361 files
OK. Both suppressions are there for a documented reason, and both hide a class of
defect that compiles cleanly and faults only when the code runs.

`survey-musl-capstone.py --strict-ptr` measures it. No QEMU, seconds to run.

## Why a single total would be misleading

On this target `sizeof(void*)` is 16 and `sizeof(uintptr_t)` is 8, both measured.
An address is cheap; authority is a separate thing. So the two directions are not
equally serious:

| direction | what happens | verdict |
|---|---|---|
| int -> ptr | fabricates an UNTAGGED capability | intended in the syscall path, where `syscall_arg_t` is `void *` and fd/count/flags ride in the cursor |
| ptr -> int | narrows 128 to 64: address kept, capability dropped | fine for `(uintptr_t)p & 15`, FATAL if cast back |

1675 int->ptr lines in 543 files is almost entirely the syscall ABI. Reporting
that as a finding would be alarming and wrong.

What actually destroys a capability is the ROUND TRIP, and it is bounded from
both sides. Both directions somewhere in a file is an UPPER bound (a syscall
wrapper can do each in a different function and never round-trip). Both on the
SAME LINE is a LOWER bound.

## The numbers

    surveyed        1361 files
    no conversion    795 files are free of both
    ptr -> int       114 lines in  57 files
    int -> ptr      1675 lines in 543 files
    round trip, upper bound   34 files
    round trip, lower bound   14 files

## The 14, and they are only four problems

**Gratuitous, delete the casts.** `src/exit/atexit.c:73,78` round-trips a
FUNCTION pointer through `uintptr_t` to park it in a `void *`. Here `void *` is
16 bytes and holds it directly; the integer detour exists only to satisfy a
pedantic C rule. Calling the result on this target dereferences an untagged code
capability.

**Mechanical, and the rewrite is a rule.** `src/malloc/oldmalloc/aligned_alloc.c:27`
aligns a pointer up through an integer. Any

    (void *)((uintptr_t)p OP X)

becomes

    p + (((uintptr_t)p OP X) - (uintptr_t)p)

which is pointer arithmetic and carries the provenance.

**A real bug in an argument that must be a pointer.** `src/select/pselect.c:12`:

    syscall_arg_t data[2] = { (uintptr_t)mask, _NSIG/8 };

`mask` is a `sigset_t *` the kernel dereferences, not a scalar riding in the
cursor. This is NOT the intended syscall pattern; it is a destroyed capability in
an argument that has to be one. It became visible only after the diagnostics were
demoted to warnings (see "the instrument" below).

**Needs a design, not a fix.** The five timer files (`timer_create.c:125`,
`timer_delete.c:8`, `timer_getoverrun.c:8`, `timer_gettime.c:8`,
`timer_settime.c:10`) pack a `pthread_t` into a `timer_t` with `>>1` and the sign
bit. A capability cannot be shifted and recovered; this wants a side table.
Threads are unsupported here, so it is not urgent. Note `timer_create.c` is the
same file that carried a real defect found by hand earlier the same day.

**And the six mallocng files, all at one line of `meta.h`:**

    const struct meta_area *area = (void *)((uintptr_t)meta & -4096);
    assert(area->check == ctx.secret);

## Why this matters more than the list

A capability-port sketch for mallocng was written the same day and MISSED that
line. All six files compiled without a diagnostic, because `-w` hides it, and the
port would have faulted at `area->check` the first time anything was freed.

The successful compile said nothing about it. The scan found it in seconds.

## The instrument, and how it was wrong twice

Both mistakes were caught by controls before any number was reported, which is
the only reason the numbers above are worth anything.

1. **Wrong diagnostic names.** clang has SEPARATE diagnostics for `void *` and
   for typed pointers (`-Wvoid-pointer-to-int-cast`, `-Wint-to-void-pointer-cast`).
   The first pattern matched only the typed ones and the positive control
   reported zero, which stopped the run.
2. **Errors truncate the file.** `int-conversion` is an ERROR in current clang,
   so removing the suppression makes the compile STOP at the first site and every
   later one in that file is never emitted. Demoting all six to warnings changed
   the measurement from 391 files to 543, and raised the round-trip lower bound
   from 13 to 14: `pselect.c` was hidden behind the abort.

`--strict-ptr` therefore carries both controls and refuses to print a number if
either misbehaves:

- POSITIVE: `(void *)(uintptr_t)p` on one line MUST be detected as a round trip.
- NEGATIVE: the two directions in one file on DIFFERENT lines must NOT be.

A detector that has never fired is unproven; one that fires on everything is
worse than none.
