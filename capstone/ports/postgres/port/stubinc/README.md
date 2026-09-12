# The headers a freestanding build resolves instead of the host's

PostgreSQL's `c.h` includes fourteen system headers before it declares
anything of its own. A freestanding build has none of them, and a build that
found the host's would be compiling against the host's word sizes and the
host's types, which is the one thing a cross build must never do.

These are the declarations the seven files of the memory manager actually
reach, and nothing else. They are small on purpose: every line here is a line
that could disagree with the platform the code will run on, so a stub that
declares more than the manager uses is a liability, not a convenience.

`-nostdlibinc` and not `-nostdinc`: the compiler's own `stddef.h`, `stdarg.h`,
`stdint.h` and `stdbool.h` are correct for the target by construction and must
keep being found.

`setjmp.h` declares a `sigjmp_buf` because `elog.h` does, and nothing here ever
jumps: the manager's only use of the error path ends the process.
