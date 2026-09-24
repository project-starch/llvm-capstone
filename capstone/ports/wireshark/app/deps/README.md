# tshark's third-party libraries, cross-built for capstone64 (M-deps)

Each library tshark 4.6.8 needs is built by its own recipe, `build-<name>.sh`, from a pinned and
verified tarball (`deps.json` records each digest and where it was read). The libraries are
installed as static archives into `$TS_WORK/deps-cap` (`/tmp/capstone/tshark-app/deps-cap` by
default).

```bash
source capstone/tests/capstone-test-env.sh      # from a worktree: CAPSTONE_LLVM_BUILD_DIR and
                                                # CAPSTONE_BUILDROOT_DIR at the main clone
bash capstone/ports/wireshark/app/deps/build-zlib.sh     # then pcre2, c-ares, libgpg-error,
                                                         # libgcrypt, libxml2
```

## The tooling

- **`env.sh`** (sourced by every recipe). It builds, once per toolchain and runtime source:
  - this lane's own `libc-capstone.a`;
  - the domain runtime: start-musl, hostcall, tls, level0, the libc overrides from
    `runtime/libc_overrides.sh`, soft-float, and compiler-rt's 128-bit division;
  - `domain_entry.c`, the `capstone_main` → `main` adapter.

  It then checks `capstone-cc` in both directions: a call to `puts` links, while an undefined
  function and an unknown `-l` do not.
- **`capstone-cc`**, the CC every configure and make is given. It compiles with the capstone64
  clang against musl-capstone's headers, and links a real domain, so configure's `HAVE_*` answers
  are true of this libc. Modelled on CPython's `interpreter/toolchain/capstone-cc`.
  - `-l<name>` resolves to libraries built here.
  - `TS_CENSUS=1` turns on clang's pointer↔integer cast warnings.
  - `TS_CENSUS_LOG` collects them per compile, under a lock, so a parallel build cannot garble
    them.

## Every recipe's gate

1. The library's **own test suite passes natively**, in the same configuration and with the
   port's patches applied.
2. It **cross-builds** with the cast census on. Every pointer→integer ("lossy") and
   integer→pointer ("rebuilt") site is listed in `$TS_WORK/deps-build/<name>-logs/cast-sites.txt`,
   taken per compile under a lock, and classified by hand below.
3. Its **own tools or test programs link** as capstone64 domains.

## Results (2026-09-24)

All seven libraries tshark 4.6.8 requires pass their gates.


| library | native tests | cross | cast sites, unique (lossy / rebuilt) | provenance-losing, and what was done |
|---|---|---|---|---|
| zlib 1.3.2 | `make test` OK | `libz.a` | 1 / 0 | none: `crc32.c:647` is an alignment test |
| PCRE2 10.48 (8-bit, no JIT) | 3/3 | `libpcre2-8.a` | 0 / 0 (census shown to fire through libtool) | none |
| c-ares 1.34.8 | 1,093/1,093 (mock-server tests; Live* need a network) | `libcares.a` | 5 / 0 after the patch (10 before) | **5 "cast off const" round trips**, `(void *)((size_t)p)`, patched (`patches/c-ares-0001`); the 5 left hash addresses |
| libgpg-error 1.61 (no threads) | 13/13 (`t-poll` does not compile without threads, upstream) | `libgpg-error.a` | 1 / 0 | none: `%p` formatting |
| libgcrypt 1.12.4 (no asm) | 40 pass, 2 skipped (6 GB/256 GB hashes), `t-lock` fails as expected: it drives locks from many threads, and libgpg-error has none | `libgcrypt.a` | 56 / 0 after the patch (3 rebuilt before) | **3 patched** (`patches/libgcrypt-0002`): ChaCha20/Salsa20 selftests aligned by masking an address (now add an offset, as rijndael.c does); `sexp_null_cond`'s constant-time select through `uintptr_t` (now a branch, which gives up constant time). `fips.c`'s `__thread` (C-47) is plain static storage in a domain (`patches/libgcrypt-0001`) |
| libxml2 2.15.4 (no ICU, no threads) | 12/12 upstream checks: runtest 3,410, runsuite 1,471, … (`testModule` needs a shared build; `runxmlconf` has 0 tests, its suite is not in the tarball) | `libxml2.a` | 35 / 13 | none: attribute values are offsets rebuilt as `base + offset` (a real pointer plus an integer); the rest are integers carried in pointers and hashed addresses |
| GLib 2.80.5 (libglib only) | glib:glib suite 133 ok, 1 skipped, 0 failed | `libglib-2.0.a`, 0 failed objects; 9/9 GLib test programs link | 40 / 23 after the patches | see below |


**GLib**, cross-configured by its own meson (`build-glib.sh`), so `config.h`'s answers come from
this libc. That alone removed the M0 census's futex and wait-header failures: musl-capstone has
neither header, and GLib falls back.
- **The cross file** answers the few run-time probes (printf family, `va_list` copy, stack
  direction, `strlcpy`, `/proc`).
- **libffi** is a stub `.pc`, since only libglib is built, not GObject.

Seven patches, each under `__CAPSTONE__` except `gqsort`'s copy, whose native suite (`sort`, 4
subtests) covers the rewrite:
- `glib-0001`: `gintptr`/`guintptr` are `long`, the address, because no integer type is as wide as
  a capstone64 pointer and the compiler has no `__intcap_t`;
- `glib-0002`: once-init on a `gsize` uses `gsize` atomics (it did a 16-byte load of an 8-byte
  object), and `g_once_init_leave_pointer` no longer detours through `guintptr`;
- `glib-0003`: `g_atomic_pointer_and`/`or`/`xor`, `gdataset`'s flag macros and lock-and-get, and
  `gbitlock`'s mask move the capability's address, instead of doing integer atomics on its storage
  or rebuilding it from an integer;
- `glib-0004`: four static assertions that `gintptr` is pointer-wide become "large enough" or go;
- `glib-0005`: an aligned allocator built from `malloc`, since musl-capstone's heap has none;
- `glib-0006`: `gqsort`'s copy mode 2 moves pointer-sized words whole. Found by reading, not by
  the census, which cannot see a cast of a pointer's type.
- `glib-0007`: no "dynamic ASAN loading" in a domain. GLib declares the LeakSanitizer entry points
  weak and calls them when their address is not NULL; in a domain an undefined weak symbol's
  address is not NULL (ISSUES C-56's open half), so `g_ignore_leak()` would call the image base.
  Found by the tshark M0 link gate, not by GLib's suite. The recipe now refuses an archive with any
  undefined weak symbol; that check fired on the unpatched archive (`__lsan_enable`,
  `__lsan_ignore_object`) and passes with the patch, and none of the other six archives has one.

Of the 63 census lines left, the 23 rebuilt sites all carry real integers (quarks, fds, log
depths, error numbers, unichars) or are the `ghash` small-array code, dead with 16-byte pointers.
None rebuilds a pointer from an address.

**Also needed by the recipes:** the runtime carries three things the shared lists do not:
- compiler-rt's 128-bit integer division and four long-double conversions (libgcrypt's mpi, GLib's
  tests);
- `runtime/atomic_libcalls.c`, for 16-byte atomics.
