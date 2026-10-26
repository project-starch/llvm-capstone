# Slot census: how many heap objects are live at once

How many lifetime-id slots does a process need? The decision rests on one number per program, input and dataset
size: the exact whole-run peak of simultaneously live heap objects in any one process. This directory measures it
natively on x86 with an `LD_PRELOAD` tracer, starting with SPEC CPU2017. Capstone is not involved, so the program's
own allocation behaviour is measured, not a port's.

    slotcensus.c     the tracer: an exact live counter at every malloc/calloc/realloc/free/memalign event (C++
                     new/delete reach it through libstdc++), the running peak, allocation and free counts, realloc
                     moves and in-place resizes, peak live usable bytes, a log2 histogram of requested sizes; one
                     JSON line per process at exit. A fork child starts at the live set it inherits.
    control.c        positive control: a known event sequence (peak, every entry point, both realloc classes,
                     realloc(NULL)/realloc(p, 0), a fork child) that prints the counts it must produce
    control_cxx.cc   positive control for operator new[]/delete[]
    check-control.py compares a control's EXPECT lines with its census lines; no census line is a FAIL
    tracer.sh        sourced by every runner: builds the tracer and the controls, runs the controls
    run-spec.sh      SPEC CPU2017 rate, C and C++ (16 benchmarks; no Fortran compiler on the measuring host)
    run-sqlite.sh    SQLite 3.22.0 speedtest1 across --size, lookaside on (the default) and off
    run-redis.sh     Redis 7.2 built with MALLOC=libc, 10^4..10^7 keys plus redis-benchmark's command mix
    run-memcached.sh memcached 1.6.45, 10^4..10^7 items (mc-load.py drives it); its slab allocator hides items
    summarize.py     per-invocation and per-benchmark peaks plus runcpu's validation verdicts
    pack.py          copies a pass into results/<stamp>/ with the host's paths removed

## Running

    bash run-spec.sh ~/spec2017/SPEC17 <pass-dir> [test|train|ref] [benchmark...]
    bash run-sqlite.sh <pass-dir> [size...]
    bash run-redis.sh <pass-dir> [keys...]
    bash run-memcached.sh <pass-dir> [keys...]
    python3 summarize.py <pass-dir> [summary.json]

`run-spec.sh` builds the tracer and the controls, then runs three controls: C (with its fork child) and C++ under
the tracer must pass, and C without the tracer (the negative control) must fail. Any other outcome exits 75 before
SPEC is touched. It then writes `config/slotcensus.cfg` from SPEC's `Example-gcc-linux-x86.cfg` (label `census`,
the system GCC, `-O2`, `-fcommon -fpermissive`, `-std=gnu++14` for C++, 502.gcc_r's
`-fgnu89-inline -D_FORTIFY_SOURCE=0`, 526.blender_r's `-std=gnu99`, without which glibc hides `struct timespec` from
its `threads.c`), builds every benchmark once, and runs each benchmark in its own `runcpu`, all
in parallel; the counts do not depend on timing. runcpu validates every output with SPEC's own tools, and
`summarize.py` refuses a benchmark without a `valid` verdict or without census lines.

The control is built with `-fno-builtin`: at `-O2` GCC deletes a `malloc` whose result is unused and a
`calloc`/`aligned_alloc`/`realloc(NULL)` that is freed at once. The first version of the control lost 3025 events
that way and its check failed, which is how this was found. The benchmarks are built normally, so an allocation the
compiler removes is not an object and is rightly not counted.

The server runners download their program's release from its project site, record its sha256, and write one cell
per dataset size: the census lines, the program's own output, and a `cells.tsv` row with the cell's verdict. Two
of them measure what a nested allocator hides: SQLite with its lookaside allocator on and off, and memcached, whose
own item count stands next to the census of the slab pages it mallocs.

What the tracer cannot see: objects inside a program's own allocator (a pool or arena carved from one malloc'd
block counts as one object), allocations made by `dlsym` while the tracer resolves the real functions (a 64 KiB
static buffer, not counted), and processes that end in `_exit` or a signal (no line).

## Results

`results/<stamp>/` holds a pass's raw lines with host paths removed (`pack.py`): the control files, each benchmark's
census lines (benchmark processes only: not the shells, specinvoke, specdiff or SPEC's image validators), the
generated config with the pass directory replaced by `<PASS>`, the provenance record, and runcpu's
`Success:`/`Error:` lines. runcpu's full logs carry the host's paths and stay on the host.

### Passes (2026-10-26, Xeon Gold 6230, GCC 15.2, glibc 2.43)

    2026-10-26-test-p12          SPEC test; 526.blender_r failed to build here (fixed after), see the next pass
    2026-10-26-test-blender-p12  SPEC test, 526.blender_r alone with -std=gnu99
    2026-10-26-train-p13         SPEC train
    2026-10-26-ref-p12           SPEC ref
    2026-10-26-sqlite-p13        speedtest1 sizes 25..800, lookaside on and off
    2026-10-26-redis-p13         10^4..10^7 populated keys
    2026-10-26-memcached-p13     10^4..10^7 items

Every pass ran its three controls as expected and every cell is valid (`python3 summarize.py results/<pass>`).
Peak live objects in one process, the maximum over a cell's processes:

    SPEC                 test        train          ref
    520.omnetpp_r   1,690,138    2,250,497    2,396,575
    510.parest_r    1,669,708    1,675,480    2,374,734
    523.xalancbmk_r    13,948    1,493,038    2,337,540
    500.perlbench_r   124,867      408,024    1,252,835
    502.gcc_r           1,549      815,610    1,069,976
    526.blender_r      18,421       35,630      328,463
    544.nab_r           1,600       14,344       80,151
    others: at most 26,926 at every size (lbm 5, deepsjeng 3, xz 21)

    servers              10^4         10^5         10^6         10^7   keys or items
    redis-server      175,721    1,452,966    8,498,473   48,923,899   (DBSIZE 30,003 .. 12,172,543)
    memcached             123          140          306        1,954   (curr_items 9,000 .. 9,000,000)
    speedtest1 1,102..1,164 at every size from 25 to 800, lookaside on or off

The ref peaks of 500.perlbench_r and 502.gcc_r agree with an earlier, separately written tracer on the same host
(1,252,835 and 1,069,978). memcached's census counts the slab pages it mallocs, not its items: the items are
invisible to the allocator below. speedtest1's peak does not grow with the data size (why is not measured here), and
lookaside off multiplies the allocation count by 8.5 to 11.7 but raises the peak by at most 45 objects.
