# Can all 16 run on Capstone, and what would each need

Short answer: **7 already do. 4 more are reachable with real but bounded work.
5 are not reachable without changing what this domain is.**

The domain is the constraint, not the compiler. It has no OS, no filesystem
(`MICROPY_VFS=0`), no threads, no sockets, and its extmod set is
binascii, deflate, framebuf, hashlib, heapq, json, random, re, uctypes.
Building an old tree is no longer a blocker: `backport-2024/` documents the
recipe, and gcc-12 plus three shims got the 2024 tree building as a domain.

## Done: 7

`MPY-T02`, `MPY-T05`, `MPY-T09`, `MPY-T10`, `MPY-T11`, `MPY-T12`, `MPY-T13`.
Two faulted, both on `cause 24`; none was caught because an object was dead.

## Reachable, in rough order of cost

**`MPY-T14` and `MPY-T15`, issues 19060 and 17848.** Need `MICROPY_VFS=1` plus
`extmod/vfs*.c` and `lib/oofatfs` in the domain build. Nothing here needs an OS:
the reporter's block device is written in Python and `VfsFat` runs over it, so
this is a build-configuration job, not a porting one. Their parent is
`64f0394d80ca^` from 2026-05, only three months before the pin, and 18 of our 20
portability patches already apply to it cleanly.

**`MPY-T07`, issue 4128.** The defect is in CALLER code, not the interpreter:
`mp_parse()` frees the lexer and the next line reads `lex->source_name`. Our own
`mpy_domain.c` can make exactly that call sequence, and `mp_parse` still frees the
lexer at the pin, so no parent build is needed at all.

  Caveat worth stating in the result: that would be the DEFECT reproduced in our
  glue, not the upstream program. It is the same API misuse against the same
  interpreter, but it is a reconstruction and must be labelled as one.

**`MPY-T16`, issue 5487.** Same shape: a port deinit hook that touches GC memory
after `gc_sweep_all()`. Our domain owns its teardown, so this is a C-level variant
of `mpy_domain.c`, and again a reconstruction rather than the ESP32 program.

## Expensive: 1

**`MPY-T06`, issue 12543.** Needs `modbtree`, which needs berkeley-db compiled
freestanding for `capstone64`. That is a real C library port, and the interesting
part of the defect is precisely that berkeley-db uses its OWN malloc outside the
GC heap, which is why it is the one row in this corpus that crashes on stock while
its GC-managed twins stay silent. Porting it would change that property.

## Not reachable without changing the domain: 5

**`MPY-T01` and `MPY-T04`, CVE-2023-7152 and issue 12887.** The defect is at
`modselect.c:151`, inside `#if MICROPY_PY_SELECT_POSIX_OPTIMISATIONS`. That block
is built on `poll()` and `struct pollfd` and needs real OS file descriptors. There
is no non-POSIX path with the same defect: the fix touches only that block.

**`MPY-T19`, issue 5226.** Needs the NimBLE stack.

**`MPY-T29`, issue 4705.** The fix is `ports/unix/gccollect.c`, making the unix
port capture stack and registers properly. Our domain has its own root capture, so
there is nothing here to reproduce; an analogous test would be a defect we
invented.

**`MPY-T21` is not in this list** because it was demoted to uncertain, see
`EXCLUDED.md`.

## What that adds up to

Running everything reachable would take the domain-measured count from 7 to 11 of
16. The remaining 5 are blocked by what the domain is, not by effort, and two of
those (`T07`, `T16`) would arrive as reconstructions rather than upstream
reproductions even when they do run.
