# php / cheribsd — PHP 5.0.0 for CheriBSD purecap

The CHERI baseline arm of [`bug-corpora/php/plain-temporal-repros`](../../../bug-corpora/php/plain-temporal-repros)
and [`bug-corpora/php/zend-alloc-repros`](../../../bug-corpora/php/zend-alloc-repros): the real
interpreter, so the 56 trigger scripts run here exactly as they run on the Capstone arms.

[`../../../tests/cheri-php-blind`](../../../tests/cheri-php-blind) measures CHERI against *rows*
that model PHP's allocator; its own Scope section says the real triggers need a full PHP 5.0.0
purecap build, gated separately. This is that gate.

## Build

    CHERI_SDK=~/cheriBSD/cheri/output/sdk \
    CHERI_SYSROOT=~/cheriBSD/cheri/output/rootfs-riscv64-purecap \
    ./build.sh /tmp/capstone/php-cheribsd

The script ends by asserting the ELF's flags say `cheriabi` and `capability mode`: a hybrid build
would compile, link, run, and measure nothing.

### What had to be pre-answered, and why

PHP 5.0.0's configure settles six questions by **running** a program, which a cross build cannot
do. Each answer below is a fact about the target that is readable without running anything.

| question | answer | where the answer comes from |
|---|---|---|
| `fopencookie` | present | CheriBSD `usr/include/stdio.h:436` |
| `funopen` | answered *no* | also present — but PHP checks it FIRST and then defines its own 2004 `fopencookie` wrapper, which collides with the native declaration (7 errors in `main/streams/cast.c`). `no` is how a 2004 configure is told to take the native route; `cast.c` is the only file that reads it |
| `getaddrinfo` | works | CheriBSD libc; the probe resolves a name to decide |
| reentrant time functions | POSIX | FreeBSD's `localtime_r`/`gmtime_r` |
| ZEND_MM alignment | 16, log2 4 | a purecap pointer is a 128-bit capability aligned to 16, so the probe's union is too. (ZEND_MM is not compiled at 5.0.0 anyway) |
| cookie seeker takes `off64_t` | yes | `cookie_seek_function_t` is `int (*)(void *, off64_t *, int)`, `stdio.h:428` |

Two host-toolchain notes that cost a build each:

- **`CPP` must carry no linker flag.** autoconf 2.13 reads *any* stderr output from the
  preprocessor as a failed header check, so one `clang: warning: argument unused ... '-fuse-ld=lld'`
  makes `<netinet/in.h>`, `<pwd.h>`, `<locale.h>`, `<sys/resource.h>` and `<dlfcn.h>` all answer
  "no" — and the build then fails in five files with missing `AF_INET`, `LC_ALL`, `RUSAGE_SELF`.
- **A configure-only `lex` stub** is used on a host without flex. PHP 5.0.0 ships its generated
  scanners and the timestamp guard keeps `make` from regenerating them; `AC_PROG_LEX` probes
  regardless.

## Run

    python3 run-cheribsd-cases.py --stage <dir> --run NAME=ARGV ... \
        --record <json> --log <serial.log> \
        --sysctl security.cheri.runtime_revocation_default=1

It speaks the same `--run NAME=ARGV` contract as `../app/run-virtual.py`, so
`bug-corpora/php/tools/run-php-corpus.py` judges either backend the same way. The corpus runner
drives it twice — revocation on, then off — because a purecap `SIGPROT` carries no symbol, so the
**pair** is what ties a fault to the defect.

Two things it refuses to do:

- **It never opens the shared image for writing.** It copies the image once per `--work` directory
  and boots a throwaway qcow2 overlay over the copy. `-snapshot` is not enough: it still takes a
  shared write lock, and a peer lane's guest has held that image for days.
- **It never sends the program over the console.** Files reach the guest over virtio-9p
  (`kldload virtio_p9fs; mount -t p9fs -o trans=virtio phpcorpus /mnt/php`), and the binary is
  copied to `/tmp` so that 56 runs do not each pay the transport.

`controls/bounds-malloc.c` is the configuration's declared control: a read far past a 32-byte
block, which must die by `SIGPROT` in the same boot. A silence scored in a run where it did not
fault is not a reading, and the judge refuses it.
