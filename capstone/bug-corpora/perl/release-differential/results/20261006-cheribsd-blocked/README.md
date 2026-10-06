# cheribsd-revocation, 2026-10-06: attempted and BLOCKED

This arm is **not measured**. It is recorded here rather than left out, because an
absent arm with no explanation reads like an arm nobody tried.

The two ways to run Perl 5.36.3 purecap both fail, and each fails for its own
reason:

| image | platform | what happens |
|---|---|---|
| dynamically linked | stock CheriBSD (the stock CheriBSD output tree) | the loader refuses it before `main`: `ld-elf.so.1: Traditional TLS not supported`. Measured on two independent builds (the 2026-09-28 qualification build and the recipe build), with and without the `poisoncap-retire-payload-fix` libc preloaded |
| dynamically linked | the PoisonCap platform, where it *does* run | the guest's own `sshd-session` dies on a poison exception part way through a file copy — `serial.log`: `poison exception ... pid N (sshd-session) ... exited on signal 34`, and `scp` reports `Connection to 127.0.0.1 closed by remote host`. It ran a 17-section smoke there in September; it cannot carry a corpus |
| statically linked (`PERL_CHERI_STATIC=1`, added in this change) | stock CheriBSD | accepted by the kernel, then **SIGPROT (signal 34), `In-address space security exception`, on every invocation including `perl -v`** |

## The static fault is localised, and it is not the platform

One boot, ascending stages, each returning a marker:

| stage | result |
|---|---|
| a static purecap C `hello` built with the same `cc` wrapper | `HELLO 3`, rc 0 |
| **mruby, static purecap, on this same image** — the positive control | `42`, rc 0 |
| a probe calling only `PERL_SYS_INIT3`, then `perl_alloc` | both reached; `perl_alloc` returns `0x40819000` |
| the same probe's next call, `perl_construct` | **SIGPROT** |
| the full interpreter, `perl -e 'print 6*7'` | SIGPROT |

So the toolchain, the C runtime and this image's handling of `__cap_relocs` are all
sound — a static purecap binary runs, and mruby's does on this very image. The fault
is inside Perl's interpreter construction. `-Wl,-E`, which perl's `ccdlflags` adds
even under `-Uusedl`, is **not** the cause: a relink without it faults identically.

## What would settle it

Bisect `perl_construct` for the first global whose capability is not the one the
code expects — printf stages through `perl_construct`, or a core dump read with the
SDK's debugger. The dynamic build works, so the difference between the two link
modes is where the defect lives. This belongs in the Perl port's own issue trail,
not in this corpus.

## Consequence for the corpus

`corpus.json` requires three arms, not four, and every case's `status` says why the
fourth is absent. The three Capstone arms are complete and each passed its own two
controls; the host differential is complete in both directions. Comparison with the
mruby corpus, whose fourth arm *is* measured, has to stop at the three.
