# perl in a Capstone musl domain

perl, pinned at 5.36.3, cross-built with perl-cross as a Capstone domain on musl
with this repo's port runtime, and the same release built natively as the
reference its output is compared with. Why perl, which release, and what its four
nested allocators look like: `docs/design/perl-sublet-port-evaluation.md`.

## Result (2026-09-26)

**`scripts/smoke.pl` runs in a domain and its output is byte-identical to the
native reference** built from the same release
(`results/2026-09-26/smoke.txt`): 17 checks covering arrays, hashes, string
building, SV churn (20,000 hashes of arrays, made and dropped), recursion,
`eval`/`die`, numeric formatting and bignums, `sort`, regex match, substitution
and a global count, a closure, references, a blessed object with method calls,
file write and read back, a directory listing, and `pack`/`sprintf`. Exit status
0, and the only unserved syscalls are `rt_sigaction` and `rt_sigprocmask`, which a
domain has no signals for.

## Build and run

```sh
source capstone/tests/capstone-test-env.sh
CAPSTONE_LLVM_BUILD_DIR=<llvm build> RUNTIME_REPO=<llvm-capstone tree> \
  bash capstone/ports/perl/musl/build-perl-domain.sh
# Copy the resulting interpreter into the existing VM's host share.
capstone-vm --state "$CAPSTONE_TMP_ROOT/dev-vm" run /mnt/host/perl.dom -e 'print 1'
```

The build recipe retains the pinned upstream sources, cross configuration,
patches and native reference. CMake now builds the [shared application SDK](../../../runtime/applications.md)
under `$PERLD_ROOT/runtime`; its `capstone-cc` supplies the CRT and linker rules.
There is no Perl-specific entry adapter, argument file, compiler wrapper or VM
runner. The old `run-perl-domain.sh` interface has been removed. Use ordinary
argv, `run --cwd DIR -e NAME=value`, or `capstone-exec` in the Linux guest shell.

`PERLD_HEAP=level0` remains the default; `PERLD_HEAP=sublet` selects the common
revoking heap with `PERLD_HEAP_LOG`. The image declares its additional heap grant;
the launcher supplies it and the monitor reclaims it, including after SIGKILL.
No caller-side `PERLD_HEAP_REGION_BYTES` is needed.

An upstream TAP harness can run several tests without rebooting. Place the
upstream `t/` and `lib/` on the share, install the host CLI, then from the host's
copy of `t/`:

```sh
prove --exec 'capstone-vm --state /tmp/capstone/dev-vm run --cwd /mnt/host/perl-tests/t -e PERL5LIB=/mnt/host/perl-tests/lib /mnt/host/perl.dom' \
  base/if.t base/cond.t base/num.t
```

The migrated recipe was rebuilt from the pinned tarball and these three files
pass. The [complete `t/base` run](results/2026-09-26/base-tests.txt) has six
passing files and three failing files: `base/term.t` fails one of seven tests
because target fork/clone is unserved; `base/lex.t` and `base/rs.t` each end in
SIGSEGV before producing TAP output. `prove` reports 9 files, 332 emitted
assertions and exit status 1. The same Linux VM continues after both faults.
The complete upstream Perl suite has not been run. The historical 17-check smoke
result above belongs to its recorded build and is not a new claim about this
upstream subset. The common runtime separately verifies real Perl argv and a
stdin/stdout filter, alongside mruby and the lifecycle contract programs.

## Why perl-cross

perl's own `Configure` answers its questions by **running** target programs, which
a cross build cannot do. perl-cross answers them by compiling only -- a size comes
from the ELF symbol table through `readelf` (`cnf/configure_type.sh`, `checksize`)
-- and builds `miniperl` with the **host** compiler (its `Makefile`, the `miniperl`
rule), so the build never executes target code. That is the host/target split the
PostgreSQL port uses for its build tools. It carries a diff for every perl5
release from 5.22.3 to 5.44.0.

One build step in perl does run target code: `dist/Time-HiRes/Makefile.PL`
compiles a probe and executes it. That extension is disabled, together with
`PerlIO/mmap`, which needs a symbol `-Ud_mmap` removes.

## What configure cannot know about a domain

The `linux` hints answer from what musl's libc *contains*, and a domain serves
less than musl exposes. Each of these is given explicitly, with its reason in the
build script:

| Given | Why |
|---|---|
| `-Dalignbytes=16` | a capability's alignment; the hints derive 8 from `long` |
| `-Ud_mmap` | a domain has no mmap. Left defined, perl reads the page size at startup for its mmap PerlIO layer, and a domain has no auxv either, so `sysconf(_SC_PAGESIZE)` is 0 and perl dies with `panic: bad pagesize 0` before running anything |
| `-Ud_nanosleep` | this release's configure leaves the variable unset and the `config.h` template then emits `# HAS_NANOSLEEP`, which is not a directive, so every compilation fails. The script now checks the generated `config.h` for such a line rather than reading 300 errors |
| `-Accflags=-D_GNU_SOURCE` | musl declares `memrchr`, `setresuid`, `setresgid` and `eaccess` only under it while the symbols are in libc either way, so configure's link tests find them and the compile would not |

`d_fork` is left as configured: musl has `fork`, the runtime does not serve it, and
a program that forks gets an error at run time. Nothing on the path so far forks.

## What perl needed (`patches/5.36.3/`)

Each patch's header carries its evidence. 0001-0005 are conditional -- on
`PTRSIZE > IVSIZE`, `PTRSIZE > UVSIZE`, `PTRSIZE == 16` or
`__CAPSTONE_PURECAP__` -- so no other platform is affected; 0006 fixes a defect
that is simply latent elsewhere, and is unconditional.

| | File | Why |
|---|---|---|
| 0001 | `sv_inline.h` | `struct body_details` describes each SV body's size in a `U8`. At 16-byte pointers `XPVIO` is 256 bytes and truncates to **0**, `regexp` is 320 and truncates to **64**, so the arena would carve slots smaller than the bodies they are for. clang reported both and the build ignored the warnings; the build script now fails on a truncated constant |
| 0002 | `cv.h` | `PoisonPADLIST` has arms for 8- and 4-byte pointers and `#error` otherwise, which stops `ext/re`, the one extension built with `-DDEBUGGING` |
| 0003 | `gv.c` | the stash cache keeps a stash as `PTR2IV(stash)` in an SV's `IV` and reads it back with `INT2PTR`. An 8-byte `IV` cannot hold a capability, so what comes back is an address without authority: cause 24 at the head of `S_mro_get_linear_isa_dfs`, on every `perl -e`. The cache is transparent, so it is not used where a pointer is wider than an `IV` |
| 0004 | `perl.h` | two defects. `PTRV` picks the first integer whose size equals `PTRSIZE`; at 16 none matches and the fallthrough takes `unsigned`, so **every** `PTR2UV`/`PTR2IV`/`PTR2nat` truncated an address to 32 bits, silently. And `DPTR2FPTR`/`FPTR2DPTR` convert data to function pointers through an integer, which loses authority: the call through the result traps in `Perl_filter_read` on every `perl -e`. 39 sites use that pair |
| 0005 | `op.c` | `S_maybe_multideref` counts its arguments on a first pass by incrementing a pointer from `arg_buf`, which is `NULL` then, and takes `arg - arg_buf` as the count; every write through it is under `if (pass)`. Incrementing a null pointer is undefined in C and harmless elsewhere, and here it is capability arithmetic without a capability: cause 24, reached by any subscript chain. Measured at `-O2` **and** at `-O1`, which is what says it is the program and not the optimiser |
| 0006 | `doio.c` | `S_openn_setup` does `Zero(mode,sizeof(mode),char)` where `mode` is its own `char *` parameter, so it clears **a pointer's** size, not the buffer's. Both callers pass a `char[PERL_MODE_MAX]`, and PERL_MODE_MAX is 8, so wherever a pointer is 8 bytes the two agree by coincidence; at 16 it writes 16 bytes into an 8-byte stack array. A defect in perl rather than a porting difference, so the patch is unconditional and a no-op elsewhere. Present unchanged in 5.32.1, 5.36.3 and 5.38.2, and worth reporting upstream. Found at the script's first `open()` |

## Open, in the order they matter

1. **`INT2PTR` round trips that remain.** clang lists them as
   `-Wint-to-pointer-cast` on a build with warnings on; in `op.c` alone they are
   `CALL_BLOCK_HOOKS` (`op.h:837,839`), the custom-op table (`op.c:18548,18628`)
   and the COP identity cache (`op.c:1331,9598` -- that one only compares, so it
   is sound). None is on the path reached so far; each is a latent cause-24.
2. **The test suite.** 2823 `.t` files and `t/TEST` forks per file, which a domain
   cannot do. Host `prove --exec` now runs each file through the shared VM CLI;
   the complete `t/base` result above is the current upstream subset. Diagnose
   the `lex.t` and `rs.t` faults and the missing target fork/clone service before
   extending the suite.
3. **The library is staged only for development tests.** Set `PERL5LIB` to
   `/mnt/host/perl-tests/lib` for those tests. A rootfs installation and broader
   module loading coverage remain open; `scripts/smoke.pl` uses core builtins.
