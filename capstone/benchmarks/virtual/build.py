#!/usr/bin/env python3
"""Build every microbenchmark image of the virtual profile for ONE configuration.

  build.py --mode capstone --sdk <sdk-dir> --out <dir>    measured images, <dir>/<name>.dom
  build.py --mode count    --sdk <sdk-dir> --out <dir>    counting images (QEMU functional pass only)
  build.py --mode native   --out <dir>                    riscv64 gcc -O2 -static references
  build.py --mode plain    --out <dir> --plain-musl <prefix> --plain-libgcc <libgcc.a>
                                                          the Plain arm: the SDKs' clang, plain
                                                          riscv64 (-march, default rv64gc, lp64d),
                                                          static musl 1.2.5 (mallocng), the harness
                                                          with user-mode rdcycle/rdinstret

Upstream sources are read from the fetched trees (/tmp/capstone/{coremark,beebs,rv8}-src by
default) and never copied into the repository. A program that needs a source adaptation gets a
patched COPY in <dir>/src/<name>/, made by exact substitutions whose match counts are asserted (a
substitution that does not match exactly as often as declared fails the build). Iteration sizes
come from sizes.tsv next to this script. Writes <dir>/manifest.tsv and <dir>/build.log; exits
non-zero if any program fails to build.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The native reference compiler: buildroot's riscv64 gcc (a worktree has no buildroot submodule, so
# point BENCH_NATIVE_CC at the main checkout's capstone/caplifive-buildroot/build/host/bin/).
NATIVE_CC = os.environ.get('BENCH_NATIVE_CC', os.path.join(os.environ.get('CAPSTONE_BUILDROOT_DIR', ''),
                           'build/host/bin/riscv64-buildroot-linux-gnu-gcc'))
COREMARK_COMMIT = 'cfa9ab377835911f23d9b0831c7be302ed1f58de'
BEEBS_COMMIT = '049ded9f3aeb5591f553879d3a0376b8614e9422'
RV8_COMMIT = 'd61b32f9e04bf04ce6dea88607ab93497fad95af'
# Old C that clang 22 and gcc 12 treat differently by default; the same list goes to both.
LEGACY = ['-Wno-implicit-function-declaration', '-Wno-implicit-int', '-Wno-int-conversion',
          '-Wno-incompatible-pointer-types', '-Wno-return-type', '-Wno-deprecated-non-prototype',
          '-Wno-unused-value', '-Wno-error']
ALLOCATORS = ('malloc', 'free', 'realloc', 'calloc')
PLAIN_K_MAX = 3   # Plain layout pads K = 0..3, 64 bytes apart (the Capstone allocator ladder's k_step)

# ---- BEEBS ---------------------------------------------------------------------------------------
# Sources and defines from each src/<name>/Makefile.am (headers dropped): the 80 programs upstream
# builds. src/Makefile.am's SUBDIRS has 85 entries: ctl, template and trio are support directories,
# and matmult and sglib-arraysort build nothing (their Makefile.am has only EXTRA_DIST). Their sources
# are cmp-identical to matmult-int's/matmult-float's and sglib-arrayquicksort's, so the physical
# port's "matmult" (-DMATMULT_INT) and "sglib-arraysort" (-DQUICK_SORT) were matmult-int and
# sglib-arrayquicksort under other names; they are not built twice here.
BEEBS = {
    'aha-compress': ['compress_test.c'], 'aha-mont64': ['mont64.c'], 'bs': ['libbs.c'],
    'bubblesort': ['libbubblesort.c'], 'cnt': ['cnt.c'], 'compress': ['libcompress.c'],
    'cover': ['libcover.c'], 'crc': ['libcrc.c'], 'crc32': ['crc_32.c'],
    'ctl-stack': ['ctl.c', '-DCTL_STACK'], 'ctl-string': ['string.c'],
    'ctl-vector': ['ctl.c', '-DCTL_VECTOR'], 'cubic': ['libcubic.c', 'basicmath_small.c'],
    'dijkstra': ['dijkstra_small.c'], 'dtoa': ['libdtoa.c'], 'duff': ['libduff.c'],
    'edn': ['libedn.c'], 'expint': ['libexpint.c'], 'fac': ['libfac.c'], 'fasta': ['libfasta.c'],
    'fdct': ['libfdct.c'], 'fibcall': ['libfibcall.c'], 'fir': ['libfir.c'], 'frac': ['libfrac.c'],
    'huffbench': ['libhuffbench.c'], 'insertsort': ['libinsertsort.c'],
    'janne_complex': ['libjanne_complex.c'], 'jfdctint': ['libjfdctint.c'],
    'lcdnum': ['liblcdnum.c'], 'levenshtein': ['liblevenshtein.c'], 'ludcmp': ['libludcmp.c'],
    'matmult-float': ['matmult.c', '-DMATMULT_FLOAT'], 'matmult-int': ['matmult.c', '-DMATMULT_INT'],
    'mergesort': ['libmergesort.c'],
    'miniz': ['miniz.c', 'miniz_b.c', '-fno-strict-aliasing'], 'minver': ['libminver.c'],
    'nbody': ['nbody.c'], 'ndes': ['libndes.c'], 'nettle-aes': ['nettle-aes.c'],
    'nettle-arcfour': ['arcfour.c'], 'nettle-cast128': ['cast128.c'], 'nettle-des': ['des.c'],
    'nettle-md5': ['md5.c'], 'nettle-sha256': ['nettle-sha256.c'], 'newlib-exp': ['ef_exp.c'],
    'newlib-log': ['ef_log.c'], 'newlib-mod': ['ef_mod.c'], 'newlib-sqrt': ['ef_sqrt.c'],
    'ns': ['libns.c'], 'nsichneu': ['libnsichneu.c'],
    'picojpeg': ['libpicojpeg.c', 'picojpeg_test.c'], 'prime': ['libprime.c'],
    'qrduino': ['qrtest.c', 'qrencode.c', 'qrframe.c'], 'qsort': ['libqsort.c'],
    'qurt': ['libqurt.c'], 'recursion': ['librecursion.c'], 'rijndael': ['aes.c', 'aesxam.c'],
    'select': ['libselect.c'], 'sglib-arraybinsearch': ['arraybinsearch.c'],
    'sglib-arrayheapsort': ['arraysort.c', '-DHEAP_SORT'],
    'sglib-arrayquicksort': ['arraysort.c', '-DQUICK_SORT'],
    'sglib-dllist': ['dllist.c'],
    'sglib-hashtable': ['hashtable.c'], 'sglib-listinsertsort': ['listinsertsort.c'],
    'sglib-listsort': ['listsort.c'], 'sglib-queue': ['queue.c'], 'sglib-rbtree': ['rbtree.c'],
    'slre': ['libslre.c'], 'sqrt': ['libsqrt.c'], 'st': ['libst.c'],
    'statemate': ['libstatemate.c'], 'stb_perlin': ['libstb_perlin.c'],
    'stringsearch1': ['stringsearch1.c', 'fast.fwd.inc.c', 'fast.rev.d12.c'],
    'strstr': ['libstrstr.c'], 'tarai': ['libtarai.c'],
    'trio-snprintf': ['trio.c', 'trio_test.c', 'triostr.c', '-DTRIO_EXTENSION=0',
                      '-DTRIO_DEPRECATED=0', '-DTRIO_MICROSOFT=0', '-DTRIO_ERRORS=0',
                      '-DTRIO_FEATURE_FLOAT=0', '-DTRIO_SNPRINTF', '-DTRIO_SNPRINTF_ONLY'],
    'trio-sscanf': ['trio.c', 'trio_test.c', 'triostr.c', '-DTRIO_SSCANF', '-DTRIO_EXTENSION=0',
                    '-DTRIO_DEPRECATED=0', '-DTRIO_MICROSOFT=0', '-DTRIO_ERRORS=0',
                    '-DTRIO_FEATURE_FLOAT=0', '-DTRIO_FEATURE_FILE=0', '-DTRIO_FEATURE_STDIO=0',
                    '-DTRIO_FEATURE_FD=0', '-DTRIO_FEATURE_DYNAMICSTRING=0',
                    '-DTRIO_FEATURE_CLOSURE=0', '-DTRIO_FEATURE_STRERR=0',
                    '-DTRIO_FEATURE_LOCALE=0', '-DTRIO_EMBED_NAN=1', '-DTRIO_EMBED_STRING=1'],
    'ud': ['libud.c'], 'whetstone': ['libwhetstone.c'], 'wikisort': ['libwikisort.c'],
}
# BEEBS's in-file bump allocator (malloc_beebs with an empty free_beebs), the file defining it,
# and the allocator entry points that file defines. Verified from source (build fails if not).
BUMP = {
    'ctl-stack': ('ctl.c', ('malloc', 'free')), 'ctl-string': ('string.c', ('malloc', 'free')),
    'ctl-vector': ('ctl.c', ('malloc', 'free')), 'dtoa': ('libdtoa.c', ('malloc', 'free')),
    'huffbench': ('libhuffbench.c', ('malloc', 'free')),
    'miniz': ('miniz.c', ('malloc', 'realloc', 'free')),
    'qrduino': ('qrframe.c', ('malloc', 'calloc', 'free')),
    'sglib-dllist': ('dllist.c', ('malloc', 'free')),
    'sglib-hashtable': ('hashtable.c', ('malloc', 'free')),
    'sglib-listinsertsort': ('listinsertsort.c', ('malloc', 'free')),
    'sglib-listsort': ('listsort.c', ('malloc', 'free')),
    'sglib-rbtree': ('rbtree.c', ('malloc', 'free')),
    'trio-snprintf': ('trio.c', ('malloc', 'realloc', 'free')),
    'trio-sscanf': ('trio.c', ('malloc', 'realloc', 'free')),
}
# Programs whose verify_benchmark returns -1: their check value is a state hash from
# beebs/state/<name>.c (appended to the file named here), or benchmark()'s result.
STATE_FILE = {
    'bs': 'libbs.c', 'compress': 'libcompress.c', 'crc': 'libcrc.c', 'cubic': 'basicmath_small.c',
    'fir': 'libfir.c', 'huffbench': 'libhuffbench.c', 'lcdnum': 'liblcdnum.c',
    'minver': 'libminver.c', 'newlib-exp': 'ef_exp.c', 'newlib-log': 'ef_log.c',
    'newlib-mod': 'ef_mod.c', 'ns': 'libns.c', 'picojpeg': 'libpicojpeg.c', 'qsort': 'libqsort.c',
    'qurt': 'libqurt.c', 'select': 'libselect.c', 'sglib-arrayheapsort': 'arraysort.c',
    'sglib-arrayquicksort': 'arraysort.c', 'ud': 'libud.c',
    'whetstone': 'libwhetstone.c', 'wikisort': 'libwikisort.c',
}
# Source fixes, applied to the build-directory copy for EVERY variant (both arms, -libc and native),
# each with the trap or failure that requires it. They are memory-safety or capability-alignment
# corrections that the exact bounds of the virtual profile enforce; none changes a checked result.
FIXES = {
    # 1-indexed Numerical Recipes sort of arr[1..20]: reads and writes arr[20], one past the
    # 20-element global. T1: cause=28 OOB at arr+80 (bounds 80 bytes); native: SIGSEGV at exit.
    'qsort': [('libqsort.c', [(r'(?m)^float arr\[20\] = \{', 'float arr[21] = {', 1)])],
    'select': [('libselect.c', [(r'(?m)^float arr\[20\] = \{', 'float arr[21] = {', 1)])],
    # aes.h's `word` "must be a 32-bit storage unit" but is unsigned long: word_in at block+12
    # reads 8 bytes of a 16-byte block (T1: cause=28, 8-byte load at offset 12 of 16). fillrand
    # stores an unsigned long through static char r[4] (8 bytes into 4).
    'rijndael': [('aes.h', [(r'typedef unsigned long   word;', 'typedef unsigned int    word;', 1)]),
                 ('aesxam.c', [(r'static char          r\[4\];', 'static char          r[8];', 1)])],
    # ctl Vector Delete moves s->size elements from `end` after shrinking s->size, i.e. reads
    # (begin) elements past the live end of the array (ctl-vector-libc T1: cause=28 at the end of
    # a 400-byte object). The moved tail is (new size - begin) elements.
    'ctl-vector': [('vector.h', [(r'memmove\(&s->value\[begin\],&s->value\[end\],s->size\*sizeof\(type\)\);',
                                  'memmove(&s->value[begin],&s->value[end],(s->size-begin)*sizeof(type));', 1)])],
    # dtoa's private Bigint pool hands out blocks in units of sizeof(double); a Bigint holds a
    # pointer, so a block that is 8 mod 16 traps on its first capability store (T1: cause=6,
    # unaligned capability store). Round each block to 16 bytes and align the pool.
    'dtoa': [('libdtoa.c', [(r'static double private_mem\[PRIVATE_mem\], \*pmem_next = private_mem;',
                             'static double private_mem[PRIVATE_mem] __attribute__((aligned(16))), *pmem_next = private_mem;', 1),
                            (r'(len = \(sizeof\(Bigint\) \+ \(x-1\)\*sizeof\(ULong\) \+ sizeof\(double\) - 1\)\n\t\t\t/sizeof\(double\);)',
                             r'\1 len = (len + 1) & ~1u;', 1)])],
}
# Not in the measured set. Built only when named with --only (to reproduce the reason).
EXCLUDED = {
    'qrduino-libc': 'upstream use-after-free exposed by a real free: benchmark() frees strinbuf '
                    '(freeecc) and verify_benchmark then reads it; QEMU T1 cause=25 (revoked), '
                    'native verify fails',
}
NOT_PORTED = {
    'matmult': 'same source and flags as matmult-int (upstream builds nothing in src/matmult)',
    'sglib-arraysort': 'same source and flags as sglib-arrayquicksort (upstream builds nothing in '
                       'src/sglib-arraysort)',
    'rv8-bigint': 'C++ needing the C++ standard library (<vector>, <string>, <iostream>, '
                  '<algorithm>; 25 std:: uses). capstone-cc compiles C++ but the virtual SDK ships '
                  'musl C headers only: `#include <vector>` fails "file not found". No libc++.',
}
UNVERIFIED_RESULT_ONLY = {'fasta', 'miniz', 'rijndael', 'sglib-dllist', 'sglib-hashtable',
                          'stb_perlin'}
BEEBS_DIRS = {}

# ---- RV8 -----------------------------------------------------------------------------------------
# (upstream file, substitutions on the build-directory copy); each substitution is
# (regex, replacement, exact match count).
RV8 = {
    'aes': ('aes.c', [(r'(?ms)^int main\(\)\n\{.*\Z', '', 1)]),
    'norx': ('norx.c', [(r'(?ms)^/\* main \*/\n\nint main\(\)\n\{.*\Z', '', 1)]),
    'qsort': ('qsort.c', [(r'(?ms)^int main\(\)\n\{.*\Z', '', 1)]),
    'primes': ('primes.c', [(r'(?ms)^int main\(\)\n\{.*\Z', '', 1)]),
    'sha512': ('sha512.c', [(r'(?ms)^int main\(\)\n\{.*\Z', '', 1)]),
    'miniz': ('miniz.c', [(r'(?ms)^static const size_t DATA_SIZE = 8 \* 1024 \* 1024;\n.*\Z', '', 1)]),
    'dhrystone': ('dhrystone.c', [(r'(?m)^#define LOOPS 10000000$', '#define LOOPS RV8_N', 1),
                                  (r'(?m)^int main\(\)\n\{\n\tProc0\(\);\n\texit\(0\);\n\}\n', '', 1)]),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def substitute(src, dst, subs, prefix=''):
    text = Path(src).read_text(errors='surrogateescape')
    for pattern, repl, count in subs:
        text, n = re.subn(pattern, repl, text)
        if n != count:
            raise RuntimeError(f'{src}: substitution {pattern!r} matched {n} times, expected {count}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(prefix + text, errors='surrogateescape')
    return dst


def read_sizes():
    sizes = {}
    for line in (HERE / 'sizes.tsv').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        name, value = line.split('\t')[:2]
        sizes[name] = int(value)
    return sizes


HEAP_MEMBER = '*libapplication-runtime.a:heap.c.obj'


def layout_script(link_ld, layout, k):
    """The SDK's link.ld with heap.c's object moved LAST in .text and .bss, each inside a region of
    fixed size (heap-layout.json), and its text shifted by k x 64 bytes inside the region."""
    text = Path(link_ld).read_text()
    subs = [
        (r'  \.text : ALIGN\(16\) \{\n    \*\(\.text\.entry\)\n    \*\(\.text \.text\.\*\)\n  \} :text',
         f"""  .text : ALIGN(16) {{
    *(.text.entry)
    *(EXCLUDE_FILE({HEAP_MEMBER}) .text .text.*)
    . = ALIGN({layout['k_step']});
    __bench_heap_text_region = .;
    . = . + BENCH_HEAP_SHIFT;
    {HEAP_MEMBER}(.text .text.*)
    __bench_heap_text_end = .;
    . = __bench_heap_text_region + BENCH_HEAP_TEXT_REGION;
  }} :text"""),
        (r'    \*\(\.bss \.bss\.\*\)\n    \*\(COMMON\)\n    \. = ALIGN\(16\);\n  \} :data',
         f"""    *(EXCLUDE_FILE({HEAP_MEMBER}) .bss .bss.*)
    *(COMMON)
    . = ALIGN({layout['k_step']});
    __bench_heap_bss_region = .;
    {HEAP_MEMBER}(.bss .bss.*)
    __bench_heap_bss_end = .;
    . = __bench_heap_bss_region + BENCH_HEAP_BSS_REGION;
    . = ALIGN(16);
  }} :data"""),
    ]
    for pattern, repl in subs:
        text, n = re.subn(pattern, lambda m: repl, text)
        if n != 1:
            raise RuntimeError(f'{link_ld}: layout substitution matched {n} times: {pattern!r}')
    head = (f"/* Generated by capstone/benchmarks/virtual/build.py from {link_ld}: heap.c last, K={k}. */\n"
            f"BENCH_HEAP_SHIFT = {k * layout['k_step']};\n"
            f"BENCH_HEAP_TEXT_REGION = {layout['text_region']};\n"
            f"BENCH_HEAP_BSS_REGION = {layout['bss_region']};\n")
    tail = ('\nASSERT(__bench_heap_text_end <= __bench_heap_text_region + BENCH_HEAP_TEXT_REGION, '
            '"heap.c text exceeds its layout region")\n'
            'ASSERT(__bench_heap_bss_end <= __bench_heap_bss_region + BENCH_HEAP_BSS_REGION, '
            '"heap.c bss exceeds its layout region")\n')
    return head + text + tail


class Builder:
    def __init__(self, a):
        self.a = a
        self.out = a.out.resolve()
        self.mode = a.mode
        self.sizes = read_sizes()
        if self.mode == 'plain':
            # The Plain arm: the same clang binary the SDKs compile with, targeting ordinary
            # riscv64 Linux, against a plain-riscv64 musl 1.2.5 built from the SDKs' tarball (its
            # own headers only, as capstone-cc does with the Capstone musl).
            musl = a.plain_musl.resolve()
            self.plain_link = [str(musl / 'lib' / 'crt1.o'), str(musl / 'lib' / 'crti.o')]
            self.plain_libs = [str(musl / 'lib' / 'libc.a'), str(a.plain_libgcc.resolve())]
            self.plain_crtn = str(musl / 'lib' / 'crtn.o')
            self.cc = [str(Path(a.plain_cc)), '--no-default-config', '--target=riscv64-unknown-linux-musl',
                       f'-march={a.plain_march}', '-mabi=lp64d', '-nostdinc', '-isystem', str(musl / 'include')]
            self.ld = str(Path(a.plain_cc).with_name('ld.lld'))
            # Layout pads: a Plain image differs from the Capstone images in every address, so the code-placement
            # effect the K ladder measured (up to ~8 % at identical instret) cannot be held equal; it is spread
            # instead. Every program is linked at K = 0..PLAIN_K_MAX: a retained, never-executed .text section of
            # K x 64 bytes right after musl's crt1/crti, so everything after it (the program, the harness, libc)
            # sits K x 64 bytes later and nothing else changes. The kit spreads a program's processes over them.
            self.plain_pads = []
            self.log, self.warnings = [], []
            (self.out / 'pad').mkdir(parents=True, exist_ok=True)
            for k in range(PLAIN_K_MAX + 1):
                s = self.out / 'pad' / f'pad-K{k}.S'
                s.write_text('\t.section .text.bench_plain_pad,"axR",@progbits\n\t.p2align 6\n'
                             f'\t.globl __bench_plain_pad\n__bench_plain_pad:\n\t.zero {k * 64}\n')
                o = self.out / 'pad' / f'pad-K{k}.o'
                self.run(self.cc + ['-c', str(s), '-o', str(o)])
                self.plain_pads.append(str(o))
        else:
            self.cc = [str(a.sdk.resolve() / 'capstone-cc')] if self.mode != 'native' else [NATIVE_CC]
        # -ffp-contract=off: BEEBS's own checks compare floats EXACTLY against values computed
        # without fused multiply-add; contracted, ludcmp, matmult-float and st fail on both
        # compilers (QEMU, 2026-10-26). Same flag for every arm and the native build.
        # -fmacro-prefix-map: __FILE__ (miniz's and sha512's asserts) would otherwise embed this
        # build's output directory, and the two arms' objects would differ by its name.
        self.base = (['-O2', '-ffp-contract=off', f'-fmacro-prefix-map={self.out}=BUILD'] + LEGACY +
                     [f'-I{HERE}'] + shlex.split(a.extra_cflags))
        if self.mode == 'native':
            self.base += ['-static']
        if self.mode == 'plain':
            self.base += shlex.split(a.plain_cflags)
        self.log = getattr(self, 'log', [])
        self.warnings = getattr(self, 'warnings', [])
        self.kset = set()
        if self.mode not in ('native', 'plain'):
            sdk = a.sdk.resolve()
            self.sdk_config = json.loads((sdk / 'sdk.json').read_text())
            # capstone-cc's own link-input checks, imported from the SDK (the link itself is done
            # here because capstone-cc refuses -T and -Map; its command line is reproduced exactly)
            self.ccmod = importlib.machinery.SourceFileLoader('capstone_cc', str(sdk / 'capstone-cc')).load_module()
            layout_path = a.heap_layout or sdk.parent.parent / 'heap-layout.json'
            self.layout = json.loads(Path(layout_path).read_text())
            spec = importlib.util.spec_from_file_location('heap_layout', HERE / 'heap-layout.py')
            hl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hl)
            heap_sha = hashlib.sha256(hl.archive_member(self.sdk_config['runtime'], 'heap.c.obj')).hexdigest()
            if heap_sha not in self.layout['objects']:
                raise SystemExit(f'{layout_path}: does not record this SDK\'s heap.c.obj ({heap_sha[:16]}); '
                                 're-run heap-layout.py over both arms\' SDKs')
            self.heap_sha = heap_sha
            (self.out / 'layout').mkdir(parents=True, exist_ok=True)
            for k in range(self.layout['k_max'] + 1):
                (self.out / 'layout' / f'layout-K{k}.ld').write_text(
                    layout_script(self.sdk_config['script'], self.layout, k))
            if self.mode == 'capstone' and (HERE / 'kvariants.txt').exists():
                self.kset = {l.split()[0] for l in (HERE / 'kvariants.txt').read_text().splitlines()
                             if l.strip() and not l.startswith('#')}

    def run(self, cmd, cwd=None):
        line = ' '.join(shlex.quote(c) for c in cmd)
        self.log.append(line)
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if r.returncode:
            raise RuntimeError(f'command failed ({r.returncode}): {line}\n{r.stdout[-4000:]}')
        if r.stdout.strip():
            self.warnings.append(f'### {line}\n{r.stdout}')
        return r.stdout

    def compile(self, name, sources, flags):
        """Compile each source to an object (same flags for all), then link with the harness."""
        obj = self.out / 'obj' / name
        obj.mkdir(parents=True, exist_ok=True)
        objs = []
        for i, src in enumerate(sources):
            if str(src).endswith('.o'):
                objs.append(str(src))
                continue
            o = obj / f'{i}-{Path(src).stem}.o'
            self.run(self.cc + self.base + flags + ['-c', str(src), '-o', str(o)])
            objs.append(str(o))
        harness = obj / 'bench-harness.o'
        hflags = (['-DBENCH_COUNT_ALLOCS'] if self.mode == 'count' else
                  ['-DBENCH_USER_COUNTERS'] if self.mode == 'plain' else [])
        self.run(self.cc + self.base + hflags + ['-c', str(HERE / 'bench-harness.c'), '-o', str(harness)])
        objs.append(str(harness))
        if self.mode == 'native':
            image = self.out / name
            self.run(self.cc + ['-O2', '-static'] + objs + ['-o', str(image), '-lm'])
            return [(name, image, 0)]
        if self.mode == 'plain':
            # ld.lld directly (static, --gc-sections like capstone-cc's link): musl's crt, the
            # objects, musl's libc.a (libm included) and the buildroot gcc's libgcc.a for the
            # compiler support routines (128-bit long double arithmetic).
            images = []
            for k, pad in enumerate(self.plain_pads):
                image = self.out / f'{name}.K{k}'
                self.run([self.ld, '-static', '--gc-sections', '-u', '__bench_plain_pad', '-o', str(image),
                          '-Map', str(image) + '.map'] + self.plain_link + [pad] + objs +
                         ['--start-group'] + self.plain_libs + ['--end-group', self.plain_crtn])
                images.append((f'{name}.K{k}', image, k))
            return images
        images = []
        for k in (range(self.layout['k_max'] + 1) if name in self.kset else [0]):
            image = self.out / f'{name}.K{k}.dom'
            self.link(objs, image, k)
            images.append((f'{name}.K{k}', image, k))
        return images

    def link(self, objs, image, k):
        """capstone-cc's link (compiler.py main: the same inputs, checks and ld.lld flags), with the
        layout script in place of the SDK's link.ld and a map written beside the image."""
        cfg = self.sdk_config
        files, linker_inputs = self.ccmod.link_files(list(objs))
        sdk = {key: str(Path(cfg[key]).resolve()) for key in ('runtime', 'libc', 'builtins')}
        self.ccmod.check_link_inputs([*files, Path(sdk['runtime']), Path(sdk['libc']), Path(sdk['builtins'])], cfg)
        wraps = [f'--wrap={f}' for f in ALLOCATORS] if self.mode == 'count' else []
        script = cfg['script'] if self.a.layout == 'default' else str(self.out / 'layout' / f'layout-K{k}.ld')
        self.run([cfg['ld'], '--no-dependent-libraries', '--gc-sections', '-T', script, '-o', str(image.absolute()),
                  '-Map', str(image.with_suffix('.map'))] + wraps +
                 ['--whole-archive', sdk['runtime'], '--no-whole-archive', '--start-group', *linker_inputs,
                  sdk['libc'], sdk['builtins'], '--end-group'])

    # -- suites --
    def beebs(self, name, libc=False):
        root = self.a.beebs.resolve()
        d = root / 'src' / BEEBS_DIRS.get(name, name)
        spec = BEEBS[name]
        files = [s for s in spec if not s.startswith('-')]
        flags = [s for s in spec if s.startswith('-')]
        prog = name + ('-libc' if libc else '')
        size = self.sizes[prog] if prog in self.sizes else self.sizes[name]
        flags += [f'-I{root / "support"}', f'-I{d}', f'-DBOARD_REPEAT_FACTOR={size}',
                  f'-DBENCH_ID="beebs/{prog}"']
        srcdir = self.out / 'src' / prog
        sources = []
        fixes = {f: subs for f, subs in FIXES.get(name, [])}
        if fixes:
            # every file of the program is copied, so a patched header shadows the original for
            # every translation unit (a quoted include searches the including file's directory)
            flags.insert(flags.index(f'-I{d}'), f'-I{srcdir}')
            for f, subs in fixes.items():
                if not f.endswith('.c'):
                    substitute(d / f, srcdir / f, subs)
        libc_prefix = '#include "beebs/beebs-libc-alloc.h"\n' if libc and name in BUMP else ''
        for f in files:
            src = d / f
            subs, prefix = list(fixes.get(f, [])), libc_prefix
            if name in BUMP and f == BUMP[name][0]:
                # Capability correctness of the bump heap: the arena must be 16-aligned and each
                # allocation a multiple of 16, or a pointer-holding object lands on a misaligned
                # address and its capability stores trap.
                subs.append((r'(?m)^static char heap\[HEAP_SIZE\];', 'static char heap[HEAP_SIZE] __attribute__((aligned(16)));', 1))
                text = src.read_text(errors='surrogateescape')
                bumps = len(re.findall(r'heap_ptr \+= size;', text))
                subs.append((r'heap_ptr \+= size;', 'heap_ptr += (size + 15) & ~(size_t)15;', bumps))
                if libc:
                    for fn in BUMP[name][1]:
                        subs.append((rf'(?m)^{fn}_beebs \(', f'bench_unused_bump_{fn}_beebs (', 1))
            if name in STATE_FILE and f == STATE_FILE[name]:
                dst = srcdir / f
                substitute(src, dst, subs, prefix)
                with dst.open('a') as s:
                    s.write(f'\n#include "beebs/state/{name}.c"\n')
                sources.append(dst)
            elif subs or prefix or fixes:
                # every TU of a -libc variant sees the allocator mapping (prefix)
                sources.append(substitute(src, srcdir / f, subs, prefix))
            else:
                sources.append(src)
        state = ['-DBEEBS_STATE_CHECK'] if name in STATE_FILE else []
        sources.append(HERE / 'beebs' / 'beebs-main.c')
        return prog, self.compile(prog, sources, flags + state), size

    def rv8(self, name):
        upstream, subs = RV8[name]
        src = self.a.rv8.resolve() / 'src' / upstream
        copy = substitute(src, self.out / 'src' / f'rv8-{name}' / upstream, subs)
        size = self.sizes[f'rv8-{name}']
        flags = [f'-DRV8_N={size}', f'-DRV8_UPSTREAM="{copy}"', f'-I{HERE / "rv8"}']
        if name == 'dhrystone':
            # Proc0's two gettimeofday calls are the bracket; its report printf (after the bracket,
            # once per iteration) is silenced: one UART line and a delegated write per iteration.
            flags += ['-Dgettimeofday=rv8_gettimeofday', '-Dprintf=rv8_silent_printf']
        if name == 'miniz':
            flags += ['-DMINIZ_NO_STDIO', '-DMINIZ_NO_ARCHIVE_APIS', '-DMINIZ_NO_TIME']
        return f'rv8-{name}', self.compile(f'rv8-{name}', [HERE / 'rv8' / f'{name}.c'], flags), size

    def ctl(self, name):
        flags = ['-DCTL_EMPTY'] if name == 'empty' else []
        return f'ctl-{name}', self.compile(f'ctl-{name}', [HERE / 'ctl' / 'loop.c'], flags), 0

    def coremark(self):
        root = self.a.coremark.resolve()
        size = self.sizes['coremark']
        # CM_LIST_NODE_BYTES: the list-node size core_list_init sizes the list with. 32 (a purecap
        # list_head) in EVERY build, native included, so the native reference has the same list
        # geometry and its CRCs are an independent check of the capstone ones; 16 is upstream's.
        node = 16 if self.a.coremark_unmodified else 32
        flags = [f'-I{HERE / "coremark"}', f'-I{root}', f'-DITERATIONS={size}',
                 '-DTOTAL_DATA_SIZE=2000', '-DPERFORMANCE_RUN=1', f'-DCM_LIST_NODE_BYTES={node}']
        if self.a.coremark_unmodified:
            flags.append('-DCM_REPRODUCE_OVERLAP')
        objs = []
        sources = [root / f for f in ('core_matrix.c', 'core_state.c', 'core_util.c')]
        if self.a.coremark_unmodified:
            sources.insert(0, root / 'core_list_join.c')
        else:
            # The one change to a core file (see README): core_list_init sizes the list for 16
            # bytes of pointers per node; a purecap list_head is 32, so unmodified the list runs
            # ~80% past its third of the data block into the matrix block, and the matrix init
            # overwrites live list pointers (T1: cause=24 loading a `next` that holds matrix data).
            sources.insert(0, substitute(root / 'core_list_join.c', self.out / 'src' / 'coremark' / 'core_list_join.c',
                                         [(r'ee_u32 per_item=16\+sizeof\(struct list_data_s\);',
                                           'ee_u32 per_item=CM_LIST_NODE_BYTES+sizeof(struct list_data_s);', 1)]))
        sources.append(HERE / 'coremark' / 'core_portme.c')
        # core_main.c is the only file with a per-file flag: main renamed, so the port's main
        # can run several iteration sets in one process.
        obj = self.out / 'obj' / 'coremark'
        obj.mkdir(parents=True, exist_ok=True)
        main_obj = obj / 'core_main.o'
        self.run(self.cc + self.base + flags + ['-Dmain=coremark_main', '-c', str(root / 'core_main.c'),
                                               '-o', str(main_obj)])
        return 'coremark', self.compile('coremark', sources + [main_obj], flags), size


def verify_trees(a):
    for path, commit in ((a.coremark, COREMARK_COMMIT), (a.beebs, BEEBS_COMMIT), (a.rv8, RV8_COMMIT)):
        head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
        if head != commit or dirty:
            sys.exit(f'{path}: at {head or "?"}{" (dirty)" if dirty else ""}, expected clean {commit}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=('capstone', 'count', 'native', 'plain'), required=True)
    p.add_argument('--plain-musl', type=Path, help='plain mode: install prefix of the riscv64 musl')
    p.add_argument('--plain-libgcc', type=Path, help='plain mode: a riscv64 lp64d libgcc.a')
    p.add_argument('--plain-cc', default=os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), 'clang'),
                   help='plain mode: the clang (default $CAPSTONE_LLVM_BIN/clang, the SDKs\' compiler)')
    p.add_argument('--plain-march', default='rv64gc')
    p.add_argument('--plain-cflags', default='', help='plain mode: appended to every compile')
    p.add_argument('--sdk', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--beebs', type=Path, default=Path('/tmp/capstone/beebs-src'))
    p.add_argument('--coremark', type=Path, default=Path('/tmp/capstone/coremark-src'))
    p.add_argument('--rv8', type=Path, default=Path('/tmp/capstone/rv8-src'))
    p.add_argument('--only', help='comma-separated program names (manifest names)')
    p.add_argument('--layout', choices=('controlled', 'default'), default='controlled',
                   help="default: the SDK's own link.ld (the gate's negative control only)")
    p.add_argument('--heap-layout', type=Path, help='heap-layout.json (default: <sdk>/../../heap-layout.json)')
    p.add_argument('--coremark-unmodified', action='store_true',
                   help='build core_list_join.c unmodified (reproduces the list/matrix overlap fault)')
    p.add_argument('--extra-cflags', default='', help='appended to every compile (experiments only)')
    p.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', '12')))
    a = p.parse_args()
    if a.mode in ('capstone', 'count') and not a.sdk:
        sys.exit('--sdk is required for capstone and count builds')
    if a.mode == 'plain' and not (a.plain_musl and a.plain_libgcc and Path(a.plain_cc).is_file()):
        sys.exit('plain mode needs --plain-musl, --plain-libgcc and a clang (--plain-cc or $CAPSTONE_LLVM_BIN)')
    if a.jobs > 12:
        sys.exit('at most -j12 on this host')
    verify_trees(a)
    a.out.mkdir(parents=True, exist_ok=True)
    b = Builder(a)
    jobs = [('ctl-empty', lambda: b.ctl('empty')), ('ctl-loop', lambda: b.ctl('loop')), ('coremark', b.coremark)]
    jobs += [(f'rv8-{n}', (lambda n=n: b.rv8(n))) for n in RV8]
    for n in BEEBS:
        jobs.append((n, (lambda n=n: b.beebs(n))))
        if n in BUMP:
            jobs.append((n + '-libc', (lambda n=n: b.beebs(n, libc=True))))
    if not a.only:
        jobs = [j for j in jobs if j[0] not in EXCLUDED]
    with (a.out / 'excluded.tsv').open('w') as f:
        for n, why in {**EXCLUDED, **NOT_PORTED}.items():
            f.write(f'{n}\t{why}\n')
    if a.only:
        want = set(a.only.split(','))
        unknown = want - {j[0] for j in jobs}
        if unknown:
            sys.exit(f'unknown program(s): {", ".join(sorted(unknown))}')
        jobs = [j for j in jobs if j[0] in want]
    rows, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        futures = {pool.submit(fn): name for name, fn in jobs}
        for f in concurrent.futures.as_completed(futures):
            name = futures[f]
            try:
                prog, images, size = f.result()
                for image_name, image, k in images:
                    rows.append((image_name, prog, k, sha256(image), size))
            except Exception as e:  # noqa: BLE001 - every failure is reported, none hidden
                failures.append((name, str(e)))
    (a.out / 'build.log').write_text('\n'.join(b.log) + '\n')
    (a.out / 'warnings.log').write_text('\n'.join(b.warnings) + '\n')
    order = {name: i for i, (name, _) in enumerate(jobs)}
    rows.sort(key=lambda r: (order[r[1]], r[2]))
    with (a.out / 'images.tsv').open('w') as m:
        m.write('image\tprogram\tK\timage_sha256\titerations\n')
        for r in rows:
            m.write('\t'.join(map(str, r)) + '\n')
    for name, err in sorted(failures):
        print(f'FAILED {name}: {err}', file=sys.stderr)
    print(f'built {len(rows)} of {len(jobs)} ({a.mode}) into {a.out}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
